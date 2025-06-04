import argparse
import os
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from tqdm import trange
from transformers import AutoTokenizer

from fla.modules.l2norm import l2norm
from fla.ops.gated_delta_rule.chunk import chunk_gated_delta_rule


def recurrent_gated_delta_rule_ref(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    g: torch.Tensor,
    scale: float = None,
    initial_state: torch.Tensor = None,
    output_final_state: bool = False,
):
    q, k, v, beta, g = map(
        lambda x: x.transpose(1, 2).contiguous().to(torch.float32), [q, k, v, beta, g]
    )
    B, H, T, K, V = *k.shape, v.shape[-1]
    o = torch.zeros(B, H, T, V).to(v)
    h = torch.zeros(B, H, K, V).to(v)
    if initial_state is not None:
        h = initial_state
    if scale is None:
        scale = 1 / (q.shape[-1] ** 0.5)
    q = q * scale

    effective_rank_list = []
    for i in trange(T):
        effective_rank_list.append(
            [
                effective_rank(h[0, head_idx].cpu().numpy())
                for head_idx in range(h.shape[1])
            ]
        )

        b_q = q[:, :, i]
        b_k = k[:, :, i]
        b_v = v[:, :, i].clone()
        h = h.clone() * g[:, :, i].exp()[..., None, None]
        b_beta = beta[:, :, i]
        b_v = b_v - (h.clone() * b_k[..., None]).sum(-2)
        b_v = b_v * b_beta[..., None]
        h = h.clone() + b_k.unsqueeze(-1) * b_v.unsqueeze(-2)
        o[:, :, i] = torch.einsum("bhd,bhdm->bhm", b_q, h)

    if not output_final_state:
        h = None
    o = o.transpose(1, 2).contiguous()
    return o, h, effective_rank_list


def chunk_gated_delta_rule_ref(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    chunk_size: int = 64,
    scale: float = None,
    initial_state: torch.Tensor = None,
    output_final_state: bool = False,
):
    BT = chunk_size
    if scale is None:
        scale = 1 / (q.shape[-1] ** 0.5)
    # Calculate padding needed to make T a multiple of BT
    q, k, v, beta, g = map(
        lambda x: x.transpose(1, 2).contiguous().to(torch.float32), [q, k, v, beta, g]
    )

    T = q.shape[-2]
    pad_len = (BT - (T % BT)) % BT
    if pad_len > 0:
        # Pad all tensors
        q = F.pad(q, (0, 0, 0, pad_len))
        k = F.pad(k, (0, 0, 0, pad_len))
        v = F.pad(v, (0, 0, 0, pad_len))
        beta = F.pad(beta, (0, pad_len))
        g = F.pad(g, (0, pad_len))
    q, k, v, beta, g = map(lambda x: x.to(torch.float32), [q, k, v, beta, g])
    decay = g
    chunk_size = BT
    b, h, l, d_k = q.shape
    d_v = v.shape[-1]
    q = q * scale
    v = v * beta[..., None]
    k_beta = k * beta[..., None]
    assert l % chunk_size == 0
    # note that diagonal is masked.
    mask = torch.triu(
        torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=q.device),
        diagonal=0,
    )
    q, k, v, k_beta, decay = map(
        lambda x: rearrange(x, "b h (n c) d -> b h n c d", c=chunk_size),
        [q, k, v, k_beta, decay.unsqueeze(-1)],
    )
    decay = decay.squeeze(-1).cumsum(-1)
    L_mask = ((decay.unsqueeze(-1) - decay.unsqueeze(-2)).tril().exp().float()).tril()
    attn = -((k_beta @ k.transpose(-1, -2)) * L_mask).masked_fill(mask, 0)
    for i in range(1, chunk_size):
        attn[..., i, :i] = attn[..., i, :i].clone() + (
            attn[..., i, :i, None].clone() * attn[..., :i, :i].clone()
        ).sum(-2)
    attn = attn + torch.eye(chunk_size, dtype=torch.float, device=q.device)
    attn = attn
    k_cumsum = attn @ v

    attn = -(k_beta @ k.transpose(-1, -2)).masked_fill(mask, 0)
    for i in range(1, chunk_size):
        attn[..., i, :i] = attn[..., i, :i].clone() + (
            attn[..., i, :i, None].clone() * attn[..., :i, :i].clone()
        ).sum(-2)
    attn = attn + torch.eye(chunk_size, dtype=torch.float, device=q.device)
    attn = attn
    k_cumdecay = attn @ k_beta
    v = k_cumsum
    S = k.new_zeros(b, h, d_k, d_v)
    if initial_state is not None:
        S = initial_state
    o = torch.zeros_like(v)
    mask = torch.triu(
        torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=q.device),
        diagonal=1,
    )
    for i in range(0, l // chunk_size):
        q_i, k_i, v_i = q[:, :, i], k[:, :, i], v[:, :, i]
        attn = (q_i @ k_i.transpose(-1, -2) * L_mask[:, :, i]).masked_fill_(mask, 0)
        v_prime = (k_cumdecay[:, :, i] * decay[:, :, i, :, None].exp()) @ S
        v_new = v_i - v_prime
        o_inter = (q_i * decay[:, :, i, :, None].exp()) @ S
        o[:, :, i] = o_inter + attn @ v_new
        S = (
            S * decay[:, :, i, -1, None, None].exp()
            + (
                k_i * (decay[:, :, i, -1, None] - decay[:, :, i]).exp()[..., None]
            ).transpose(-1, -2)
            @ v_new
        )
    if not output_final_state:
        S = None
    # unpad
    o = rearrange(o, "b h n c d -> b h (n c) d")
    o = o[:, :, :T]
    o = o.transpose(1, 2)
    return o, S


def effective_rank(matrix):
    """
    Computes the effective rank of a matrix based on the definition by Roy and Vetterli.

    Args:
        matrix (numpy.ndarray): The input matrix.

    Returns:
        float: The effective rank of the matrix.
    """
    # Step 1: Compute singular values
    singular_values = np.linalg.svd(matrix, compute_uv=False)

    # Handle cases with zero singular values for numerical stability
    singular_values = singular_values[
        singular_values > np.finfo(float).eps
    ]  # Use a small epsilon

    if len(singular_values) == 0:
        return 0  # Or handle as per convention for a zero matrix

    # Step 2: Calculate the singular value distribution (p_k)
    norm_s1 = np.sum(singular_values)
    if norm_s1 == 0:
        # This case implies all singular values were zero or very close to zero
        return 0  # Effective rank of a zero matrix can be considered 0
        # or handle as per paper's convention for all-zero matrix
        # The paper states "non-all-zero matrix A" [cite: 1]

    p_k = singular_values / norm_s1

    # Step 3: Calculate Shannon Entropy (H)
    # Handle p_k = 0 using the convention 0 * log(0) = 0
    # np.log(0) is -inf. p_k * np.log(p_k) would be 0 * -inf which is nan.
    # We can filter out p_k == 0 before calculating entropy terms
    # However, p_k calculated from non-zero singular_values should not be zero.
    # For robustness against extremely small p_k values that might cause issues:
    entropy_terms = -p_k * np.log(p_k)
    entropy = np.sum(entropy_terms)

    # Step 4: Compute Effective Rank
    erank = np.exp(entropy)

    return erank


if __name__ == "__main__":
    import logging

    import git

    # Find repository root
    repo = git.Repo(search_parent_directories=True)
    repo_root = repo.working_tree_dir

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("main")
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer", type=int, default=10)
    parser.add_argument("--dataset", type=str, default="trivia_qa")
    parser.add_argument("--seq_len", type=int, default=16384)
    parser.add_argument("--train_context_len", type=int, default=2048)
    parser.add_argument("--path", type=str, default="")
    parser.add_argument("--model_name", type=str, default="dp3")
    parser.add_argument("--num_householder", type=int, default=3)
    parser.add_argument("--recompute", action="store_true")
    args = parser.parse_args()

    layer = args.layer
    dataset = args.dataset
    seq_len = args.seq_len
    model_name = args.model_name
    num_householder = args.num_householder
    recompute = args.recompute

    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")

    if (
        Path(
            os.path.join(
                repo_root,
                f"data/ranks/effective_rank_list_{dataset}_{seq_len}_layer_{layer}_{model_name}.pt",
            )
        ).exists()
        and not recompute
    ):
        logger.info(
            f"Loading effective rank list for {dataset} with sequence length {seq_len} and layer {layer}"
        )
        effective_rank_list = torch.load(
            os.path.join(
                repo_root,
                f"data/ranks/effective_rank_list_{dataset}_{seq_len}_layer_{layer}_{model_name}.pt",
            ),
            weights_only=False,
        )

        logger.info(f"Loading activations for {dataset} with sequence length {seq_len}")
        with open(
            os.path.join(
                repo_root,
                f"data/activations/data_{dataset}_{seq_len}_{model_name}.pkl",
            ),
            "rb",
        ) as file:
            data = pickle.load(file)
    else:
        logger.info(f"Loading activations for {dataset} with sequence length {seq_len}")
        with open(
            os.path.join(
                repo_root,
                f"data/activations/data_{dataset}_{seq_len}_{model_name}.pkl",
            ),
            "rb",
        ) as file:
            data = pickle.load(file)

        ks = torch.tensor(data[f"model.layers.{layer}.attn.k_id"]).to("cuda")
        vs = torch.tensor(data[f"model.layers.{layer}.attn.v_id"]).to("cuda")
        qs = torch.tensor(data[f"model.layers.{layer}.attn.q_id"]).to("cuda")
        betas = torch.tensor(data[f"model.layers.{layer}.attn.beta_id"]).to("cuda")
        gs = data.get(f"model.layers.{layer}.attn.g_id", None)
        if gs is not None:
            gs = torch.tensor(gs).to("cuda")

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            ks = l2norm(ks)
            qs = l2norm(qs)

            logger.info(f"qs.shape: {qs.shape}")
            logger.info(f"ks.shape: {ks.shape}")
            logger.info(f"vs.shape: {vs.shape}")
            logger.info(f"betas.shape: {betas.shape}")
            logger.info(f"gs.shape: {gs.shape if gs is not None else 'None'}")
            logger.info("Computing effective rank list with recurrent gated delta rule")
            o, S, effective_rank_list = recurrent_gated_delta_rule_ref(
                q=qs,
                k=ks,
                v=vs,
                beta=betas,
                g=gs if gs is not None else torch.zeros_like(betas),
                initial_state=None,
                output_final_state=True,
            )
            # Save effective rank list
            torch.save(
                effective_rank_list,
                os.path.join(
                    repo_root,
                    f"data/ranks/effective_rank_list_{dataset}_{seq_len}_layer_{layer}_{model_name}.pt",
                ),
            )

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                o_chunk, S_chunk = chunk_gated_delta_rule(
                    q=qs.to(torch.bfloat16),
                    k=ks.to(torch.bfloat16),
                    v=vs.to(torch.bfloat16),
                    g=gs.to(torch.bfloat16)
                    if gs is not None
                    else torch.zeros_like(betas).to(torch.bfloat16),
                    beta=betas.to(torch.bfloat16),
                    # chunk_size=64,
                    # scale=1 / (qs.shape[-1] ** 0.5),
                    initial_state=None,
                    output_final_state=True,
                )

            logger.info(
                "(S - S_chunk).abs().mean(dim=-1): {:.4f}".format(
                    (S - S_chunk.float()).abs().mean().item()
                )
            )

    num_heads = len(effective_rank_list[0])
    seq_length = len(effective_rank_list) // num_householder

    plt.figure(figsize=(4, 4))
    for head in range(num_heads):
        plt.plot(
            [x / num_householder for x in range(0, seq_length * num_householder)],
            [
                effective_rank_list[i][head]
                for i in range(0, seq_length * num_householder)
            ],
            label=f"Head {head}",
            linewidth=0.8,
        )

    for idx, token_idx in enumerate(data["s_start_idx"]):
        plt.axvline(
            x=token_idx,
            color="red",
            linestyle="--",
            alpha=0.5,
            label="BOS" if idx == 0 else "",
        )

    plt.axvline(
        x=args.train_context_len,
        color="green",
        linestyle="--",
        alpha=1,
        label="Train seq. length",
    )
    plt.xlabel("Sequence Position")
    plt.ylabel("Effective Rank")
    plt.title(f"{dataset} {seq_len} Layer {layer}")
    # plt.xlim(0, 4096)
    # plt.yscale("log")
    plt.legend(fontsize=6)
    plt.grid(True)
    plt.savefig(
        os.path.join(
            repo_root,
            f"plots/effective_rank_plot_{dataset}_{seq_len}_layer_{layer}_{model_name}.pdf",
        ),
        dpi=500,
    )
