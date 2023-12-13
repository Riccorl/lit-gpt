import argparse
import json
import logging
import os
from pathlib import Path

import huggingface_hub
import psutil
from datasets import Dataset, load_dataset, DatasetDict
from huggingface_hub import login
from tqdm import tqdm

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

HF_CACHE = os.getenv("HF_HOME", os.path.expanduser("~/.cache/huggingface"))


def save_to_jsonl(dataset, path_to_save, split_size):
    # now we can save the dataset in jsonl format, divided in multiple files
    # we would like to parallelize this step, we can use the multiprocessing library
    dataset_path = Path(path_to_save)
    dataset_path.mkdir(parents=True, exist_ok=True)

    # create a new file for each split, parallelize this step
    dataset_subsets = []
    for ds_split in dataset:
        print(f"Saving `{ds_split}` split")
        logger.info(f"Saving `{ds_split}` split")
        # if the dataset is too big, we can split it in multiple files
        # but if it is small enough, we can save it in a single file
        shards = dataset[ds_split].num_rows // split_size + 1
        print(f"Splitting in {shards} shards")
        for i in range(shards):
            dataset_subsets.append(
                dataset[ds_split][i * split_size : (i + 1) * split_size]
            )

        split_path = dataset_path / f"{ds_split}"
        split_path.mkdir(parents=True, exist_ok=True)

        for idx, subset in tqdm(enumerate(dataset_subsets)):
            print(f"Writing to {split_path / f'{idx}.jsonl'}")
            with open(split_path / f"{idx}.jsonl", "w") as f:
                # it is a dict from key to list of values for that key
                # e.g. "text" -> [tex1, ..., textn]
                # "id" -> [id1, ..., idn]
                # we instead want a list of dicts in the form of
                # [{"text": text1, "id": id1 }, ..., {"text": textn, "id": idn }]
                rows = {}
                for key, values in subset.items():
                    for i, value in enumerate(values):
                        if i not in rows:
                            rows[i] = {}
                        rows[i][key] = value
                rows = rows.values()
                f.writelines(json.dumps(r) + "\n" for r in rows)

        dataset_subsets = []


def download_streaming_dataset(dataset, path_to_save, split_size, row_to_download):
    dataset_samples = []
    hf_dataset = None
    print(dataset)
    for ds_split in dataset:
        print(f"Saving `{ds_split}` split")
        logger.info(f"Saving `{ds_split}` split")
        # if the dataset is too big, we can split it in multiple files
        # but if it is small enough, we can save it in a single file
        dataset_split = dataset[ds_split]
        pbar = tqdm(
            dataset_split, total=row_to_download, desc=f"Downloading {ds_split}"
        )
        for i, sample in enumerate(pbar, 1):
            dataset_samples.append(sample)

            if len(dataset_samples) >= row_to_download:
                # hf_dataset = Dataset.from_list(dataset_samples, split=ds_split)
                hf_dataset = DatasetDict({ds_split: Dataset.from_list(dataset_samples)})
                save_to_jsonl(hf_dataset, path_to_save, split_size)
                dataset_samples = []
                break

        if len(dataset_samples) > 0:
            hf_dataset = DatasetDict({ds_split: Dataset.from_list(dataset_samples)})
            save_to_jsonl(hf_dataset, path_to_save, split_size)
            dataset_samples = []


def main(args):
    token = huggingface_hub.HfFolder.get_token()
    if token is None:
        logger.info("HuggingFace Login")
        login()

    if args.streaming:
        logger.info("==== Starting download CulturaX in streaming mode ====")
        dataset = load_dataset(
            path=args.dataset_name,
            language=args.language,
            # split="train",  # TODO change this
            streaming=True,
            cache_dir=HF_CACHE,
            token=True,
        )
        # download_streaming_dataset(dataset, args)
        # TODO improve this
        download_streaming_dataset(
            dataset, args.path_to_save, args.split_size, args.max_row
        )
    else:
        logger.info("==== Starting download CulturaX ====")
        dataset = load_dataset(
            path=args.dataset_name,
            language=args.language,
            # split="train",
            cache_dir=HF_CACHE,
            token=True,
            num_proc=psutil.cpu_count(logical=False),
        )
        save_to_jsonl(dataset, args.path_to_save, args.split_size)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download HF Dataset")
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="uonlp/CulturaX",
        help="Dataset to download.",
    )
    parser.add_argument(
        "--language",
        type=str,
        default="it",
        help="Language of the dataset to download.",
    )
    parser.add_argument(
        "--path_to_save",
        type=str,
        default="./",
        help="Path to save the dataset.",
    )
    parser.add_argument(
        "--max_row",
        type=int,
        default=100_00,
        help="Max number of row to download",
    )
    parser.add_argument(
        "--step",
        type=int,
        default=10,
        help="Number of document to download to wait every memory check",
    )
    parser.add_argument(
        "--streaming",
        action="store_true",
        help="Download in streaming mode",
    )
    parser.add_argument(
        "--split_size",
        type=int,
        default=100_000,
        help="Number of document per file during jsonl saving",
    )
    args = parser.parse_args()

    main(args)
