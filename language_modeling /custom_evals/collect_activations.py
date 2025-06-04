import argparse
import os

import fla  # noqa
import git
import torch
from custom_evals.nethook import TraceDict
from custom_evals.ssmworkbench.text_datamodule import TextArrowFileModule
from transformers import AutoModelForCausalLM


@torch.no_grad()
def collect_activations(
    model,
    data_module,
    device,
    max_length,
    test,
    dataset_name: str,
    seq_len: int,
    model_name: str,
    repo_root: str,
):
    model.eval()

    if len(data_module.val_dataloader()) == 0:
        data_loader = data_module.train_dataloader()
    else:
        data_loader = data_module.val_dataloader()

    batch = next(iter(data_loader))
    input_ids: torch.Tensor = batch["src_seq"].to(device)

    layers = [f"model.layers.{m}.attn.k_id" for m in range(24)]
    layers += [f"model.layers.{m}.attn.v_id" for m in range(24)]
    layers += [f"model.layers.{m}.attn.q_id" for m in range(24)]
    layers += [f"model.layers.{m}.attn.beta_id" for m in range(24)]
    layers += (
        [f"model.layers.{m}.attn.g_id" for m in range(24)]
        if model_name.startswith("G")
        else []
    )
    with (
        TraceDict(
            model,
            layers=layers,
            retain_input=False,
            retain_output=True,
        ) as td,
        torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True),
    ):
        _ = model(input_ids)  # shape: (B, L, vocab_size)

    import pickle

    with open(
        os.path.join(
            repo_root,
            f"data/activations/data_{dataset_name}_{seq_len}_{model_name}.pkl",
        ),
        "wb",
    ) as file:
        s_start_idx = [
            i
            for i, id in enumerate(input_ids.cpu().detach().numpy().flatten().tolist())
            if id == 1
        ]  # 1 is <s> token id

        data = {k: v.output.float().cpu().detach().numpy() for k, v in td.items()}

        data["input_ids"] = input_ids.cpu().detach().numpy()
        data["s_start_idx"] = s_start_idx
        pickle.dump(
            data,
            file,
        )


def main():
    # Find repository root
    repo = git.Repo(search_parent_directories=True)
    repo_root = repo.working_tree_dir

    parser = argparse.ArgumentParser(
        description="Evaluate trained models on LM benchmarks"
    )
    parser.add_argument("--max_len", type=int, default=16384)
    parser.add_argument(
        "-p",
        "--path",
        type=str,
        default=os.path.join(
            repo_root,
            "<path to checkpoint>",
        ),
    )
    parser.add_argument("-d", "--data", type=str, default="trivia_qa")
    parser.add_argument("-s", "--split", type=str, default="train")
    parser.add_argument("--model_name", type=str, default="gdn")
    args = parser.parse_args()
    device = "cuda"
    dtype = torch.float
    torch.manual_seed(0)

    model = AutoModelForCausalLM.from_pretrained(
        args.path, device_map={"": device}, torch_dtype=dtype
    )
    model.eval()

    data_module = TextArrowFileModule(
        tokenizer=args.path,
        dataset_name=args.data,
        batch_size=1,
        num_cpu_worker=8,
        max_sample_len=args.max_len,
        data_dir="<path to huggingface data directory>",
        cache_dir="<path to huggingface cache directory>",
        val_ratio=0.0005,
        val_split_seed=2357,
        seed=42,
    )
    collect_activations(
        model,
        data_module,
        device,
        args.max_len,
        test=False,
        dataset_name=args.data,
        seq_len=args.max_len,
        model_name=args.model_name,
        repo_root=repo_root,
    )


if __name__ == "__main__":
    main()
