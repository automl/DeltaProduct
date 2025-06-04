import argparse
import json
import logging
import os

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM

import fla  # noqa
from custom_evals.ssmworkbench.text_datamodule import TextArrowFileModule


@torch.inference_mode()
def evaluate_length_extrapolation(model, data_module, device, max_length):
    model.eval()
    total_loss_sum = torch.zeros(max_length - 1, device=device)
    total_accuracy_sum = torch.zeros(max_length - 1, device=device)
    total_count = 0
    per_token_losses = []

    if len(data_module.val_dataloader()) == 0:
        data_loader = data_module.train_dataloader()
    else:
        data_loader = data_module.val_dataloader()

    for idx, batch in tqdm(enumerate(data_loader), desc="Evaluating lengths"):
        # if idx > 20:
        # break
        src_seq = batch["src_seq"].to(device)
        trg_seq = batch["trg_seq"].to(device)

        with torch.autocast(dtype=torch.bfloat16, device_type="cuda"):
            outputs = model(src_seq)
        logits = outputs.logits

        # Compute loss for all tokens
        loss = torch.nn.functional.cross_entropy(
            logits.transpose(1, 2).float(), trg_seq, reduction="none"
        )

        # Compute accuracy for all tokens
        predictions = torch.argmax(logits, dim=-1)
        correct_predictions = predictions == trg_seq

        # Compute cumulative metrics
        cum_loss = torch.cumsum(loss, dim=1)
        cum_correct = torch.cumsum(correct_predictions.float(), dim=1)

        # Calculate running averages
        token_positions = torch.arange(1, cum_loss.size(1) + 1, device=device)
        avg_cum_loss = cum_loss / token_positions
        avg_cum_accuracy = cum_correct / token_positions

        # Sum across batch dimension
        total_loss_sum += avg_cum_loss.sum(dim=0)
        total_accuracy_sum += avg_cum_accuracy.sum(dim=0)
        total_count += src_seq.size(0)
        per_token_losses.append(loss.cpu().float().numpy())

    # Compute mean metrics for each length

    mean_losses = (total_loss_sum / total_count).cpu().numpy()
    mean_accuracies = (total_accuracy_sum / total_count).cpu().numpy()
    per_token_losses_avg = np.concatenate(per_token_losses).mean(axis=0)

    # Calculate perplexities
    perplexities = np.exp(mean_losses)

    return {
        # "lengths": lengths.tolist(),
        "perplexities": perplexities.tolist(),
        "accuracies": mean_accuracies.tolist(),
        "token_losses": per_token_losses_avg.tolist(),
    }


def main():
    logging.basicConfig(level=logging.INFO)
    import git

    # Find repository root
    repo_root = git.Repo(search_parent_directories=True).working_tree_dir

    parser = argparse.ArgumentParser(
        description="Evaluate trained models on LM benchmarks"
    )
    parser.add_argument("--max_len", type=int, default=4096)
    parser.add_argument(
        "-p",
        "--path",
        type=str,
        default=os.path.join(
            repo_root,
            "<path to checkpoint>",
        ),
    )
    parser.add_argument("--data", type=str, default="codeparrot")
    parser.add_argument("--model_name", type=str, default="dp3")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_cpu_workers", type=int, default=8)
    args = parser.parse_args()

    if os.path.exists(
        os.path.join(
            repo_root,
            f"data/length_extrapolation/{args.data}_{args.max_len}_{args.model_name}.json",
        )
    ):
        logging.info(f"Skipping {args.data}_{args.max_len}_{args.model_name}.json")
        return

    logging.info(f"Model Checkpoint: {args.path}")
    device = "cuda"
    dtype = torch.float
    torch.manual_seed(0)

    logging.info(f"Loading model {args.path}")
    model = AutoModelForCausalLM.from_pretrained(
        args.path, device_map={"": device}, torch_dtype=dtype
    )
    model.eval()
    logging.info(f"{model}")
    logging.info(f"Model Checkpoint: {args.path}")
    device = "cuda"
    dtype = torch.float
    torch.manual_seed(0)

    model.eval()

    data_module = TextArrowFileModule(
        tokenizer=args.path,
        dataset_name=args.data,
        batch_size=args.batch_size,
        num_cpu_worker=args.num_cpu_workers,
        max_sample_len=args.max_len,
        data_dir="<hf-data-dir>",
        cache_dir="<hf-cache-dir>",
        val_ratio=0.0005,
        val_split_seed=2357,
        seed=42,
    )
    results = evaluate_length_extrapolation(
        model, data_module, device, args.max_len, test=False
    )

    # Save results as json
    with open(
        os.path.join(
            repo_root,
            f"data/length_extrapolation/{args.data}_{args.max_len}_{args.model_name}.json",
        ),
        "w",
    ) as f:
        json.dump({"results": results}, f)


if __name__ == "__main__":
    main()
