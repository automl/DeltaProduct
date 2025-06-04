import os
import pathlib

import torch
from datasets import load_dataset, load_from_disk
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from custom_evals.ssmworkbench.collators import DataCollator
from custom_evals.ssmworkbench.detokenizer import wikitext_detokenize

IGNORE_INDEX = -100
ACCESS_TOKEN = "hf_XXX"


class IndexDataset(torch.utils.data.Dataset):
    """
    Wrapper class to hold arrow file dataset indices
    """

    def __init__(self, dataset_indices):
        self.dataset_indices = dataset_indices

    def __getitem__(self, index):
        return self.dataset_indices[index]

    def __len__(self):
        return len(self.dataset_indices)


class TextArrowFileModule:
    """
    Datamodule to perform pretraining
    based on 1 train arrow file, 1 val arrow file
    Assumes that pre-processed indices exist
    """

    def __init__(
        self,
        tokenizer,
        dataset_name,
        num_cpu_worker,
        max_sample_len,
        seed,
        batch_size,
        data_dir,
        cache_dir,
        val_ratio,
        val_split_seed,
        logger=None,
        dev_set=False,
    ):
        super().__init__()

        if logger is None:
            import logging

            logger = logging.getLogger(__name__)
            logger.setLevel(logging.INFO)
            logger.addHandler(logging.StreamHandler())
        self.logger = logger
        self.num_cpu_worker = num_cpu_worker
        self.resume_index = None  # TODO not implemented yet
        self.dataset_name = dataset_name
        self.data_dir = pathlib.Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir = pathlib.Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        if dev_set:
            self.dev_set = True
            self.data_dir = self.data_dir / "dev"
            self.data_dir.mkdir(parents=True, exist_ok=True)
            self.logger.warning("Running in dev mode")
        else:
            self.dev_set = False

        self.batch_size = batch_size
        self.max_sample_len = max_sample_len
        self.seed = seed

        self.load_tokenizer(tokenizer)
        self.ignore_index = IGNORE_INDEX
        self.global_rank = 0
        self.collator = DataCollator(
            pad_token_id=self.tokenizer.pad_token_id,
            ignore_index=self.ignore_index,
        )
        tokenized_dataset = self.load_tokenized_dataset(val_ratio, val_split_seed)
        self.chunked_dataset = self.chunk_single_dataset(
            tokenized_dataset, max_sample_len
        )

    def load_tokenizer(self, path):
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.tokenizer_name = "mistral"
        self.vocab_size = len(self.tokenizer)

    def load_dataset_from_hf(self, val_ratio, val_split_seed):
        if self.dataset_name == "slimpajama_6b":
            all_samples = load_dataset(
                "DKYoon/SlimPajama-6B",
                cache_dir=self.cache_dir.absolute().as_posix(),
            )
        elif self.dataset_name == "codeparrot":
            all_samples = load_dataset(
                "codeparrot/codeparrot-train-v2-near-dedup",
                cache_dir=self.cache_dir.absolute().as_posix(),
            )
            valid_samples = load_dataset(
                "codeparrot/codeparrot-valid-v2-near-dedup",
                cache_dir=self.cache_dir.absolute().as_posix(),
            )
            all_samples["validation"] = valid_samples["train"]

        elif self.dataset_name == "openwebtext":
            all_samples = load_dataset(
                "Skylion007/openwebtext",
                cache_dir=self.cache_dir.absolute().as_posix(),
            )
        elif self.dataset_name == "fineweb_100b":
            all_samples = load_dataset(
                "HuggingFaceFW/fineweb",
                "sample-100BT",
                cache_dir=self.cache_dir.absolute().as_posix(),
            )
        elif self.dataset_name == "fineweb_10b":
            all_samples = load_dataset(
                "HuggingFaceFW/fineweb",
                "sample-10BT",
                cache_dir=self.cache_dir.absolute().as_posix(),
            )
        elif self.dataset_name == "fineweb_edu":
            all_samples = load_dataset(
                "HuggingFaceFW/fineweb-edu",
                "sample-10BT",
                cache_dir=self.cache_dir.absolute().as_posix(),
            )
        elif self.dataset_name == "math":
            all_samples = load_dataset("lighteval/MATH", "all")
            all_samples = all_samples.map(
                lambda example: {
                    "text": f"Problem: {example['problem']}, Solution: {example['solution']}"
                },
                num_proc=max(self.num_cpu_worker, 1),
                desc="Mapping math dataset to text",
            )
        elif self.dataset_name == "math_hard":
            all_samples = load_dataset("lighteval/MATH-Hard", "default")
            all_samples = all_samples.map(
                lambda example: {
                    "text": f"Problem: {example['problem']}, Solution: {example['solution']}"
                },
                num_proc=max(self.num_cpu_worker, 1),
                desc="Mapping math dataset to text",
            )
        elif self.dataset_name == "gsm8k":
            all_samples = load_dataset("openai/gsm8k", "socratic")
            all_samples = all_samples.map(
                lambda example: {
                    "text": f"Problem: {example['question']}, Solution: {example['answer']}"
                },
                num_proc=max(self.num_cpu_worker, 1),
                desc="Mapping math dataset to text",
            )
        elif self.dataset_name == "trivia_qa":
            all_samples = load_dataset("sentence-transformers/trivia-qa")
            all_samples = all_samples.map(
                lambda example: {
                    "text": f"Question: {example['query']}, Answer: {example['answer']}"
                },
                num_proc=max(self.num_cpu_worker, 1),
                desc="Mapping trivia dataset to text",
            )
        elif self.dataset_name == "camel_ai_math":
            all_samples = load_dataset(
                "camel-ai/math", "default", num_proc=max(self.num_cpu_worker, 1)
            )
            all_samples = all_samples.map(
                lambda example: {
                    "text": f"Problem: {example['message_1']}, Solution: {example['message_2']}"
                },
                num_proc=max(self.num_cpu_worker, 1),
                desc="Mapping camel_ai_math dataset to text",
            )
        elif self.dataset_name == "deepmind_math_dataset":
            all_samples = load_dataset(
                "deepmind/math_dataset", "algebra__linear_2d_composed"
            )
            all_samples = all_samples.map(
                lambda example: {
                    "text": f"Problem: {example['question']}, Solution: {example['answer']}"
                },
                num_proc=max(self.num_cpu_worker, 1),
                desc="Mapping deepmind_math_dataset dataset to text",
            )
        elif self.dataset_name == "openthoughts-114k-math":
            all_samples = load_dataset("open-r1/openthoughts-114k-math")
            all_samples = all_samples.map(
                lambda example: {
                    "text": f"Problem: {example['problem']}, Solution: {example['solution']}"
                },
                num_proc=max(self.num_cpu_worker, 1),
                desc=f"Mapping {self.dataset_name} dataset to text",
            )
        elif self.dataset_name == "openthoughts-114k-math-correct":
            all_samples = load_dataset("open-r1/openthoughts-114k-math")
            all_samples = all_samples.filter(
                lambda example: example.get("correct", False)
            )
            all_samples = all_samples.map(
                lambda example: {
                    "text": f"Problem: {example['problem']}, Solution: {example['solution']}"
                },
                num_proc=max(self.num_cpu_worker, 1),
                desc=f"Mapping {self.dataset_name} dataset to text",
            )
        elif self.dataset_name == "openthoughts-114k-math-incorrect":
            all_samples = load_dataset("open-r1/openthoughts-114k-math")
            all_samples = all_samples.filter(
                lambda example: not example.get("correct", False)
            )
            all_samples = all_samples.map(
                lambda example: {
                    "text": f"Problem: {example['problem']}, Solution: {example['solution']}"
                },
                num_proc=max(self.num_cpu_worker, 1),
                desc=f"Mapping {self.dataset_name} dataset to text",
            )
        elif self.dataset_name == "wikitext":
            all_samples = load_dataset(
                "wikitext",
                "wikitext-103-v1",
                cache_dir=self.cache_dir.absolute().as_posix(),
            )

            # [2021-12-25] TD: Running the detokenizer on wikitext-103 makes ppl worse
            # (GPT2-small val ppl after 10 epochs ~22 -> ~25)
            # However, it's useful for zero-shot transfer from Openwebtext,
            # as after detokenization it's closer to Openwebtext's format.
            # https://github.com/stanford-crfm/mistral/issues/12
            all_samples = all_samples.map(
                lambda example: {"text": wikitext_detokenize(example["text"])},
                num_proc=max(self.num_cpu_worker, 1),
                desc="Running detokenizer on dataset",
            )
        else:
            raise UserWarning(f"dataset name unknown: {self.dataset_name}")

        if "validation" not in all_samples and self.dataset_name != "math":
            all_samples = all_samples["train"].train_test_split(
                test_size=val_ratio,
                seed=val_split_seed,
                shuffle=True,  # Otherwise test will be at the end of the dataset
            )
            all_samples["validation"] = all_samples["test"]
            del all_samples["test"]
        return all_samples

    def tokenize_dataset(self, all_samples):
        def encode(examples):
            if self.dataset_name == "codeparrot":
                content = examples["content"]
            else:
                content = examples["text"]
            return {"input_ids": self.tokenizer(content)["input_ids"]}

        self.logger.info(
            f"Start tokenizing dataset {self.dataset_name} with {self.tokenizer_name}"
        )
        tokenized_dataset = all_samples.map(
            encode,
            batched=True,
            remove_columns=list(all_samples["train"].features.keys()),
            desc=f"Running {self.tokenizer_name} tokenizer on {self.dataset_name} dataset",
            num_proc=self.num_cpu_worker,
        )

        return tokenized_dataset

    def load_tokenized_dataset(self, val_ratio, val_split_seed):
        tokenized_dataset_dir = (
            self.data_dir / f"{self.dataset_name}_{self.tokenizer_name}_all"
        )

        if os.path.exists(tokenized_dataset_dir / "successful_processed.txt"):
            tokenized_dataset = load_from_disk(tokenized_dataset_dir)
            self.logger.info(f"Loaded tokenized dataset from {tokenized_dataset_dir}")

        else:
            all_samples = self.load_dataset_from_hf(val_ratio, val_split_seed)

            if self.dev_set:
                all_samples["train"] = (
                    all_samples["train"].select(range(10000)).to_iterable_dataset()
                )
                all_samples["validation"] = (
                    all_samples["validation"].select(range(100)).to_iterable_dataset()
                )

            tokenized_dataset = self.tokenize_dataset(all_samples)

            os.makedirs(tokenized_dataset_dir, exist_ok=True)
            tokenized_dataset.save_to_disk(tokenized_dataset_dir, max_shard_size="10GB")
            with open(tokenized_dataset_dir / "successful_processed.txt", "w") as f:
                f.write(f"processed {self.dataset_name} with {self.tokenizer_name}")
                f.write(f"vocab size: {self.vocab_size}")
                f.write(f"train samples: {len(all_samples['train'])}")
                f.write(f"validation samples: {len(all_samples['validation'])}")

            self.logger.info(f"Saved tokenized dataset to {tokenized_dataset_dir}")

        return tokenized_dataset

    def chunk_single_dataset(self, tokenized_dataset, context_length):
        chunked_dataset_dir = (
            self.data_dir
            / f"{self.dataset_name}_{self.tokenizer_name}_chunked_{context_length}"
        )

        if os.path.exists(chunked_dataset_dir / "successful_processed.txt"):
            chunked_dataset = load_from_disk(chunked_dataset_dir)
            self.logger.info(f"Loaded chunked dataset from {chunked_dataset_dir}")

        else:
            # chunk dataset
            def group_texts(examples):
                chunks = []
                remaining = []
                for _idx, sentence in enumerate(examples["input_ids"]):
                    sentence = remaining + sentence
                    length = len(sentence)
                    if length < context_length:
                        remaining = sentence
                        continue
                    chunked_length = (length // context_length) * context_length
                    chunks += [
                        sentence[i : i + context_length]
                        for i in range(0, chunked_length, context_length)
                    ]
                    remaining = sentence[chunked_length:]
                return {"input_ids": chunks}

            chunked_dataset = tokenized_dataset.map(
                group_texts,
                # with_indices=list(range(10000)) if self.dev_set else False,
                batched=True,
                num_proc=self.num_cpu_worker,
                desc=f"Grouping texts in chunks of {context_length}",
            )

            chunked_dataset.save_to_disk(
                chunked_dataset_dir,
                max_shard_size="10GB",
                num_proc=1,
            )
            with open(chunked_dataset_dir / "successful_processed.txt", "w") as f:
                f.write(f"processed {self.dataset_name} with {self.tokenizer_name}")
                f.write(f"vocab size: {self.vocab_size}")
                f.write(f"train samples: {len(chunked_dataset['train'])}")
                f.write(f"validation samples: {len(chunked_dataset['validation'])}")

            self.logger.info(f"Saved chunked dataset to {chunked_dataset_dir}")

        return chunked_dataset

    def train_dataloader(
        self,
        current_epoch=0,
        local_rank=0,
    ):
        loader = DataLoader(
            self.chunked_dataset["train"],
            batch_size=self.batch_size,
            collate_fn=self.collator,
            num_workers=self.num_cpu_worker,
            shuffle=True,
            pin_memory=True,
            drop_last=True,
            prefetch_factor=4,
        )
        self.logger.info("Finished loading training data")
        return loader

    def val_dataloader(
        self,
        current_epoch=0,
        local_rank=0,
    ):
        loader = DataLoader(
            self.chunked_dataset["validation"],
            batch_size=self.batch_size,
            collate_fn=self.collator,
            num_workers=self.num_cpu_worker,
            pin_memory=True,
            drop_last=False,
            prefetch_factor=4,
        )
        self.logger.info("Finished loading val data")
        return loader

    def old_train_dataloader(
        self,
        current_epoch=0,
        local_rank=0,
    ):
        train_set_size = self._train_dataset.num_record_batches
        train_indexes = list(range(train_set_size))
        train_indexes = self.rng.permutation(train_indexes)

        # min_num_samples = torch.LongTensor([train_set_size]).to(local_rank)
        min_num_samples = torch.tensor(train_set_size, device=f"cuda:{local_rank}")
        if torch.distributed.is_initialized():
            torch.distributed.all_reduce(
                min_num_samples, op=torch.distributed.ReduceOp.MIN
            )
        min_num_samples = min_num_samples.item()
        train_indexes = train_indexes[:min_num_samples]

        self.logger.info(
            f"### load train set with size {min_num_samples} from {train_set_size} samples on rank {self.global_rank}"
        )

        # shuffle the indices for every epoch other than 0.
        # the loaded indices are already shuffled
        if current_epoch > 0:
            seed = self.seed + current_epoch + self.global_rank
            tmp_rng = np.random.default_rng(seed)
            train_indexes = tmp_rng.permutation(train_indexes)

        if self.resume_index is not None:
            train_indexes = train_indexes[self.resume_index :]
            self.resume_index = None  # reset to avoid next-epoch issues

        train_index_dataset = IndexDataset(train_indexes)

        def train_pl_collate_fn(indices):
            raw_samples = [
                self._train_dataset.get_record_batch(i)["text"].to_pylist()[0]
                for i in indices
            ]
            return self.collator(raw_samples)

        loader = DataLoader(
            train_index_dataset,
            batch_size=self.batch_size,
            collate_fn=train_pl_collate_fn,
            num_workers=self.num_cpu_worker,
            pin_memory=True,
            drop_last=False,
        )
        self.logger.info("Finished loading training data")
        return loader

    def old_val_dataloader(self, local_rank=0):
        valid_set_size = self._valid_dataset.num_record_batches
        valid_indexes = list(range(valid_set_size))

        min_num_samples = torch.tensor(valid_set_size, device=f"cuda:{local_rank}")

        if torch.distributed.is_initialized():
            torch.distributed.all_reduce(
                min_num_samples, op=torch.distributed.ReduceOp.MIN
            )
        min_num_samples = min_num_samples.item()
        valid_indexes = valid_indexes[:min_num_samples]

        valid_index_dataset = IndexDataset(valid_indexes)

        print(
            f"### load valid set with size {min_num_samples} from {valid_set_size} samples on rank {self.global_rank}"
        )

        def val_pl_collate_fn(indices):
            inputs = [
                self._valid_dataset.get_record_batch(i)["text"].to_pylist()[0]
                for i in indices
            ]
            return self.collator(inputs)

        loader = DataLoader(
            valid_index_dataset,
            batch_size=self.batch_size,
            collate_fn=val_pl_collate_fn,
            num_workers=self.num_cpu_worker,
            pin_memory=True,
            drop_last=False,
        )
        self.logger.info("Finished loading validation data")
        return loader






























