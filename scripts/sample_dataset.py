"""
Sample images from the processed_detection YOLO dataset for quick testing.

Usage:
    python scripts/sample_dataset.py                  # default 50 images
    python scripts/sample_dataset.py --count 25       # 25 images
    python scripts/sample_dataset.py --percent 5      # 5% of dataset
    python scripts/sample_dataset.py --seed 42        # reproducible
"""

import argparse
import random
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
SOURCE_DATASET = ROOT / "processed_detection"
OUTPUT_DIR = ROOT / "repo_model_AI" / "models" / "yolo_sample_test"


def sample_split(images_dir: Path, labels_dir: Path, out_images: Path, out_labels: Path, count: int, seed: int):
    """Sample a fixed number of images from one split and copy with labels."""
    all_images = sorted([f for f in images_dir.iterdir() if f.suffix.lower() in (".jpg", ".jpeg", ".png")])

    rng = random.Random(seed)
    sampled = rng.sample(all_images, min(count, len(all_images)))

    out_images.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)

    copied = 0
    for img_path in sampled:
        label_path = labels_dir / (img_path.stem + ".txt")
        shutil.copy2(img_path, out_images / img_path.name)
        if label_path.exists():
            shutil.copy2(label_path, out_labels / label_path.name)
        copied += 1

    return copied


def main():
    parser = argparse.ArgumentParser(description="Sample a YOLO dataset for quick testing")
    parser.add_argument("--count", type=int, default=50, help="Total number of images to sample (default: 50)")
    parser.add_argument("--percent", type=float, default=None, help="Percentage to sample (overrides --count)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--source", type=str, default=str(SOURCE_DATASET), help="Source dataset path")
    parser.add_argument("--output", type=str, default=str(OUTPUT_DIR), help="Output directory")
    args = parser.parse_args()

    source = Path(args.source)
    output = Path(args.output)

    if not (source / "images").exists():
        print(f"Error: source dataset not found at {source}")
        return

    # Clean previous sample
    if output.exists():
        shutil.rmtree(output)

    # Calculate per-split counts
    splits_info = {}
    for split in ("train", "val"):
        img_dir = source / "images" / split
        if img_dir.exists():
            n_total = len([f for f in img_dir.iterdir() if f.suffix.lower() in (".jpg", ".jpeg", ".png")])
            splits_info[split] = n_total

    total_source = sum(splits_info.values())

    if args.percent is not None:
        total_sample = max(2, round(total_source * args.percent / 100))
    else:
        total_sample = min(args.count, total_source)

    # Distribute proportionally across splits
    split_counts = {}
    allocated = 0
    sorted_splits = sorted(splits_info.keys())
    for i, split in enumerate(sorted_splits):
        if i == len(sorted_splits) - 1:
            split_counts[split] = total_sample - allocated
        else:
            n = max(1, round(total_sample * splits_info[split] / total_source))
            split_counts[split] = n
            allocated += n

    total = 0
    for split_name, count in split_counts.items():
        img_dir = source / "images" / split_name
        lbl_dir = source / "labels" / split_name
        n = sample_split(
            img_dir, lbl_dir,
            output / "images" / split_name,
            output / "labels" / split_name,
            count, args.seed,
        )
        print(f"  {split_name}: sampled {n}/{splits_info[split_name]} images")
        total += n

    # Write data.yaml (ultralytics-compatible)
    data_yaml = f"""\
path: {output.as_posix()}
train: images/train
val: images/val
nc: 1
names:
  0: microorganism
"""
    (output / "data.yaml").write_text(data_yaml)
    print(f"\nSampled {total} total images -> {output}")
    print(f"data.yaml written to {output / 'data.yaml'}")


if __name__ == "__main__":
    main()
