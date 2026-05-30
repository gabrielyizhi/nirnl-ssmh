"""Build the NUS-WIDE h5py file expected by the local data loader.

This script uses the official NUS-WIDE train/test split, the 1k tag
features as text features, and concatenated normalized low-level image
features as image features.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import h5py
import numpy as np


FEATURE_FILES = {
    "CH": ("Train_Normalized_CH.dat", "Test_Normalized_CH.dat"),
    "CM55": ("Train_Normalized_CM55.dat", "Test_Normalized_CM55.dat"),
    "CORR": ("Train_Normalized_CORR.dat", "Test_Normalized_CORR.dat"),
    "EDH": ("Train_Normalized_EDH.dat", "Test_Normalized_EDH.dat"),
    "WT": ("Train_Normalized_WT.dat", "Test_Normalized_WT.dat"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nuswide-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--train-size", default=10500, type=int)
    parser.add_argument("--valid-size", default=2100, type=int)
    parser.add_argument("--test-size", default=2100, type=int)
    parser.add_argument("--num-labels", default=21, type=int)
    parser.add_argument("--seed", default=1, type=int)
    parser.add_argument(
        "--features",
        default="CH,CM55,CORR,EDH,WT",
        help="Comma-separated low-level feature groups to concatenate.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def label_name(path: Path, split: str) -> str:
    prefix = "Labels_"
    suffix = f"_{split}.txt"
    name = path.name
    if not name.startswith(prefix) or not name.endswith(suffix):
        raise ValueError(f"Unexpected label file name: {path}")
    return name[len(prefix) : -len(suffix)]


def load_split_labels(label_dir: Path, split: str) -> tuple[list[str], np.ndarray]:
    paths = sorted(label_dir.glob(f"Labels_*_{split}.txt"), key=lambda p: label_name(p, split))
    if not paths:
        raise FileNotFoundError(f"No label files found for split {split} in {label_dir}")

    names = [label_name(path, split) for path in paths]
    columns = []
    expected_rows = None
    for path in paths:
        values = np.loadtxt(path, dtype=np.float32).astype(np.int16)
        values = np.asarray(values).reshape(-1)
        if expected_rows is None:
            expected_rows = values.shape[0]
        elif values.shape[0] != expected_rows:
            raise ValueError(f"Row count mismatch in {path}: {values.shape[0]} != {expected_rows}")
        columns.append(values)
    return names, np.stack(columns, axis=1).astype(np.int16)


def choose_top_labels(
    train_names: list[str],
    train_labels: np.ndarray,
    test_names: list[str],
    test_labels: np.ndarray,
    num_labels: int,
) -> tuple[list[str], np.ndarray, np.ndarray]:
    if train_names != test_names:
        raise ValueError("Train and test label files do not have the same concept order.")
    frequencies = train_labels.sum(axis=0) + test_labels.sum(axis=0)
    ranked = sorted(range(len(train_names)), key=lambda i: (-int(frequencies[i]), train_names[i]))
    selected = np.asarray(ranked[:num_labels], dtype=np.int64)
    selected_names = [train_names[i] for i in selected]
    return selected_names, train_labels[:, selected], test_labels[:, selected]


def infer_dim(path: Path) -> int:
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            values = np.fromstring(line, sep=" ", dtype=np.float32)
            if values.size:
                return int(values.size)
    raise ValueError(f"Could not infer feature dimension from empty file: {path}")


def read_selected_matrix(path: Path, selected_indices: np.ndarray) -> np.ndarray:
    selected_indices = np.asarray(selected_indices, dtype=np.int64)
    dim = infer_dim(path)
    if selected_indices.size == 0:
        return np.empty((0, dim), dtype=np.float32)

    order = np.argsort(selected_indices)
    sorted_indices = selected_indices[order]
    sorted_output = np.empty((selected_indices.size, dim), dtype=np.float32)

    target_pos = 0
    target_row = int(sorted_indices[target_pos])
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for row_number, line in enumerate(handle):
            if row_number != target_row:
                continue
            values = np.fromstring(line, sep=" ", dtype=np.float32)
            if values.size != dim:
                raise ValueError(
                    f"Feature dimension mismatch in {path} row {row_number}: "
                    f"{values.size} != {dim}"
                )
            sorted_output[target_pos] = values
            target_pos += 1
            if target_pos == sorted_indices.size:
                break
            target_row = int(sorted_indices[target_pos])

    if target_pos != sorted_indices.size:
        raise ValueError(
            f"Only found {target_pos}/{sorted_indices.size} requested rows in {path}; "
            f"last requested row was {int(sorted_indices[-1])}."
        )

    output = np.empty_like(sorted_output)
    output[order] = sorted_output
    return output


def read_image_features(
    low_level_dir: Path,
    split: str,
    selected_indices: np.ndarray,
    feature_names: list[str],
) -> np.ndarray:
    matrices = []
    split_col = 0 if split == "Train" else 1
    for feature_name in feature_names:
        if feature_name not in FEATURE_FILES:
            raise ValueError(f"Unknown feature group: {feature_name}")
        path = low_level_dir / FEATURE_FILES[feature_name][split_col]
        print(f"Reading {split} image feature {feature_name}: {path}", flush=True)
        matrices.append(read_selected_matrix(path, selected_indices))
    return np.concatenate(matrices, axis=1).astype(np.float32)


def read_text_features(tags_dir: Path, split: str, selected_indices: np.ndarray) -> np.ndarray:
    path = tags_dir / f"{split}_Tags1k.dat"
    print(f"Reading {split} text tags: {path}", flush=True)
    return read_selected_matrix(path, selected_indices).astype(np.float32)


def sample_indices(labels: np.ndarray, size: int, rng: np.random.Generator) -> np.ndarray:
    pool = np.flatnonzero(labels.sum(axis=1) > 0)
    if pool.size == 0:
        raise ValueError("No samples have at least one selected label.")
    rng.shuffle(pool)
    return np.sort(pool[: min(size, pool.size)])


def write_dataset(handle: h5py.File, key: str, data: np.ndarray) -> None:
    handle.create_dataset(
        key,
        data=data,
        compression="gzip",
        compression_opts=4,
        shuffle=True,
    )


def main() -> None:
    args = parse_args()
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"{args.output} exists. Pass --overwrite to replace it.")

    root = args.nuswide_root
    label_dir = root / "Groundtruth" / "TrainTestLabels"
    tags_dir = root / "Tags"
    low_level_dir = root / "LowLevel" / "Low_Level_Features"
    feature_names = [name.strip() for name in args.features.split(",") if name.strip()]

    train_names, train_all_labels = load_split_labels(label_dir, "Train")
    test_names, test_all_labels = load_split_labels(label_dir, "Test")
    selected_names, train_labels, test_labels = choose_top_labels(
        train_names,
        train_all_labels,
        test_names,
        test_all_labels,
        args.num_labels,
    )

    rng = np.random.default_rng(args.seed)
    train_idx = sample_indices(train_labels, args.train_size, rng)
    test_pool = np.flatnonzero(test_labels.sum(axis=1) > 0)
    rng.shuffle(test_pool)
    valid_count = min(args.valid_size, test_pool.size)
    remaining = max(0, test_pool.size - valid_count)
    test_count = min(args.test_size, remaining)
    valid_idx = np.sort(test_pool[:valid_count])
    test_idx = np.sort(test_pool[valid_count : valid_count + test_count])
    if test_idx.size == 0:
        raise ValueError("No held-out test samples remain after validation sampling.")

    print("Selected labels:", ", ".join(selected_names), flush=True)
    print(
        f"Split sizes: train={train_idx.size}, valid={valid_idx.size}, test={test_idx.size}",
        flush=True,
    )

    train_imgs = read_image_features(low_level_dir, "Train", train_idx, feature_names)
    valid_imgs = read_image_features(low_level_dir, "Test", valid_idx, feature_names)
    test_imgs = read_image_features(low_level_dir, "Test", test_idx, feature_names)

    train_texts = read_text_features(tags_dir, "Train", train_idx)
    valid_texts = read_text_features(tags_dir, "Test", valid_idx)
    test_texts = read_text_features(tags_dir, "Test", test_idx)

    train_split_labels = train_labels[train_idx].astype(np.int16)
    valid_split_labels = test_labels[valid_idx].astype(np.int16)
    test_split_labels = test_labels[test_idx].astype(np.int16)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    tmp_output = args.output.with_suffix(args.output.suffix + ".tmp")
    if tmp_output.exists():
        tmp_output.unlink()

    with h5py.File(tmp_output, "w") as handle:
        write_dataset(handle, "train_imgs_deep", train_imgs)
        write_dataset(handle, "train_texts", train_texts)
        write_dataset(handle, "train_imgs_labels", train_split_labels)
        write_dataset(handle, "valid_imgs_deep", valid_imgs)
        write_dataset(handle, "valid_texts", valid_texts)
        write_dataset(handle, "valid_imgs_labels", valid_split_labels)
        write_dataset(handle, "test_imgs_deep", test_imgs)
        write_dataset(handle, "test_texts", test_texts)
        write_dataset(handle, "test_imgs_labels", test_split_labels)

        string_dtype = h5py.string_dtype(encoding="utf-8")
        handle.create_dataset("label_names", data=np.asarray(selected_names, dtype=object), dtype=string_dtype)
        handle.attrs["source"] = "NUS-WIDE official split, Tags1k text, normalized low-level image features"
        handle.attrs["image_features"] = ",".join(feature_names)
        handle.attrs["seed"] = args.seed

    os.replace(tmp_output, args.output)
    print(f"Wrote {args.output}", flush=True)
    print(f"Image dim: {train_imgs.shape[1]}, text dim: {train_texts.shape[1]}", flush=True)


if __name__ == "__main__":
    main()
