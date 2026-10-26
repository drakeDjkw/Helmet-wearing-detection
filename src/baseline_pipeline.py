"""
Helmet Wearing Detection - Baseline Pipeline (YOLOv8s)
======================================================

Reproduces the baseline from Helmet_wear_detection.ipynb as one script:

    1. Check the full dataset (image/label pairing, class distribution)
    2. Draw a random sample (default 1000 / 200 / 200, seed 42)
    3. Copy the sample into HWD_SAMPLE/{train,val,test}/{images,labels}
    4. Write data.yaml with the dataset's real class order
    5. Train YOLOv8s
    6. Evaluate the best weights on the test split
    7. Save metrics (precision, recall, mAP50, mAP50-95, inference time, FPS, size)
    8. Save a grid of predictions on random test images

Usage
-----
    python baseline_pipeline.py                       # full pipeline, notebook defaults
    python baseline_pipeline.py --epochs 50 --device cpu
    python baseline_pipeline.py --skip-sample         # reuse existing HWD_SAMPLE
    python baseline_pipeline.py --skip-train          # evaluate existing best.pt only
"""

import argparse
import json
import os
import random
import shutil
from collections import Counter
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")  # save figures to disk; no display needed
import matplotlib.pyplot as plt
import pandas as pd
import yaml

# ==========================================
# Configuration
# ==========================================
PROJECT_ROOT = Path("/Users/sandrakhyikwa/AI_Tech/Helmet-wearing-detection")

DEFAULTS = {
    "dataset": PROJECT_ROOT / "data/WearHelmetDetection.v1i.yolov8",
    "sample_dir": PROJECT_ROOT / "data/HWD_SAMPLE",
    "data_yaml": PROJECT_ROOT / "notebook/data.yaml",
    "runs_dir": PROJECT_ROOT / "reports/runs/detect",
}

# Order MUST match the class IDs in the label .txt files
# (verified: ID 2 = heads wearing helmets, the majority class).
CLASS_NAMES = ["Head_without_helmet", "Helmet_only", "Helmet_with_head"]

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Source split folder -> sample split folder
SPLITS = {"train": "train", "valid": "val", "test": "test"}


# ==========================================
# Dataset helpers
# ==========================================
def index_split(split_dir: Path) -> pd.DataFrame:
    """Return one row per image: image file, label file, class IDs."""
    images_dir, labels_dir = split_dir / "images", split_dir / "labels"

    images = {p.stem: p.name for p in images_dir.iterdir()
              if p.suffix.lower() in IMAGE_EXTS}
    labels = {p.stem: p.name for p in labels_dir.glob("*.txt")}

    missing_labels = images.keys() - labels.keys()
    missing_images = labels.keys() - images.keys()
    if missing_labels or missing_images:
        print(f"  [warn] {split_dir.name}: {len(missing_labels)} images without labels, "
              f"{len(missing_images)} labels without images (skipped)")

    records = []
    for stem in sorted(images.keys() & labels.keys()):
        with open(labels_dir / labels[stem]) as f:
            classes = [int(float(line.split()[0])) for line in f if line.strip()]
        records.append({"image": images[stem], "label": labels[stem],
                        "num_objects": len(classes), "classes": classes})
    return pd.DataFrame(records)


def class_distribution(df: pd.DataFrame) -> pd.DataFrame:
    counter = Counter(c for classes in df["classes"] for c in classes)
    total = sum(counter.values()) or 1
    rows = [{"class_id": i,
             "class": CLASS_NAMES[i],
             "objects": counter.get(i, 0),
             "percentage": round(counter.get(i, 0) / total * 100, 2)}
            for i in range(len(CLASS_NAMES))]
    return pd.DataFrame(rows)


def check_class_ids(df: pd.DataFrame, split: str) -> None:
    ids = {c for classes in df["classes"] for c in classes}
    bad = sorted(i for i in ids if i < 0 or i >= len(CLASS_NAMES))
    if bad:
        raise ValueError(f"{split}: class IDs {bad} outside 0..{len(CLASS_NAMES) - 1}. "
                         f"Check CLASS_NAMES against the dataset's data.yaml.")


def check_against_dataset_yaml(dataset_dir: Path) -> None:
    """Warn if the Roboflow data.yaml has a different class count."""
    yaml_path = dataset_dir / "data.yaml"
    if not yaml_path.exists():
        return
    with open(yaml_path) as f:
        names = yaml.safe_load(f).get("names", [])
    names = list(names.values()) if isinstance(names, dict) else list(names)
    print(f"  Dataset data.yaml names : {names}")
    print(f"  Pipeline CLASS_NAMES    : {CLASS_NAMES}")
    if len(names) != len(CLASS_NAMES):
        raise ValueError("Class count differs from the dataset's data.yaml.")


# ==========================================
# Step 1-3: sample and copy
# ==========================================
def build_sample(dataset_dir: Path, sample_dir: Path, sizes: dict, seed: int) -> dict:
    print("\n[1/6] Indexing dataset and drawing sample")
    check_against_dataset_yaml(dataset_dir)

    # Start clean so files from an earlier run with another seed don't stay behind
    if sample_dir.exists():
        shutil.rmtree(sample_dir)

    summary = {}
    for src_split, dst_split in SPLITS.items():
        full_df = index_split(dataset_dir / src_split)
        check_class_ids(full_df, src_split)

        n = min(sizes[dst_split], len(full_df))
        sample_df = full_df.sample(n=n, random_state=seed)

        for sub in ("images", "labels"):
            (sample_dir / dst_split / sub).mkdir(parents=True, exist_ok=True)
        for _, row in sample_df.iterrows():
            shutil.copy2(dataset_dir / src_split / "images" / row["image"],
                         sample_dir / dst_split / "images" / row["image"])
            shutil.copy2(dataset_dir / src_split / "labels" / row["label"],
                         sample_dir / dst_split / "labels" / row["label"])

        full_dist = class_distribution(full_df)
        samp_dist = class_distribution(sample_df)
        comp = full_dist[["class", "percentage"]].rename(columns={"percentage": "full_%"})
        comp["sample_%"] = samp_dist["percentage"]
        comp["diff_%"] = (comp["sample_%"] - comp["full_%"]).round(2)

        print(f"\n  {dst_split.upper()}: {n} / {len(full_df)} images, "
              f"{samp_dist['objects'].sum()} objects")
        print(comp.to_string(index=False))

        summary[dst_split] = {"images": n,
                              "objects": int(samp_dist["objects"].sum()),
                              "distribution": samp_dist.to_dict(orient="records")}
    return summary


# ==========================================
# Step 4: data.yaml
# ==========================================
def write_data_yaml(sample_dir: Path, yaml_path: Path) -> None:
    print("\n[2/6] Writing data.yaml")
    data = {
        "path": str(sample_dir.resolve()),
        "train": "train/images",
        "val": "val/images",
        "test": "test/images",
        "nc": len(CLASS_NAMES),
        "names": {i: n for i, n in enumerate(CLASS_NAMES)},
    }
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    with open(yaml_path, "w") as f:
        yaml.safe_dump(data, f, sort_keys=False)
    print(f"  Saved: {yaml_path}")


# ==========================================
# Step 5: train
# ==========================================
def train(args) -> Path:
    from ultralytics import YOLO

    print(f"\n[3/6] Training {args.model} for {args.epochs} epochs on {args.device}")
    model = YOLO(args.model)
    model.train(
        data=str(args.data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        patience=args.patience,
        seed=args.seed,
        deterministic=True,
        project=str(args.runs_dir),
        name=args.run_name,
        device=args.device,
        exist_ok=True,
    )
    return args.runs_dir / args.run_name / "weights" / "best.pt"


# ==========================================
# Step 6-7: evaluate and save metrics
# ==========================================
def evaluate(args, weights: Path) -> dict:
    from ultralytics import YOLO

    print(f"\n[4/6] Evaluating {weights.name} on the test split")
    model = YOLO(str(weights))
    m = model.val(
        data=str(args.data_yaml),
        split="test",
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=str(args.runs_dir),
        name=f"{args.run_name}_test",
        exist_ok=True,
    )

    speed = m.speed  # ms per image: preprocess / inference / loss / postprocess
    infer_ms = speed.get("inference", 0.0)
    total_ms = speed.get("preprocess", 0.0) + infer_ms + speed.get("postprocess", 0.0)

    results = {
        "Model": args.run_name,
        "Precision": round(float(m.box.mp), 4),
        "Recall": round(float(m.box.mr), 4),
        "mAP50": round(float(m.box.map50), 4),
        "mAP50-95": round(float(m.box.map), 4),
        "Inference Time (ms)": round(infer_ms, 2),
        "FPS": round(1000 / total_ms, 2) if total_ms > 0 else None,
        "Model Size (MB)": round(weights.stat().st_size / 1024 ** 2, 2),
        "Device": args.device,
        "Epochs": args.epochs,
        "Image Size": args.imgsz,
    }

    # Per-class AP (Helmet_only is the rare class; watch this number)
    per_class = []
    for i, cls_idx in enumerate(m.box.ap_class_index):
        p, r, ap50, ap = m.box.class_result(i)
        per_class.append({"class": CLASS_NAMES[int(cls_idx)],
                          "precision": round(float(p), 4),
                          "recall": round(float(r), 4),
                          "mAP50": round(float(ap50), 4),
                          "mAP50-95": round(float(ap), 4)})
    results["per_class"] = per_class
    return results


def save_metrics(results: dict, sample_summary: dict, out_dir: Path) -> None:
    print("\n[5/6] Saving metrics")
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "baseline_metrics.json", "w") as f:
        json.dump({"test_metrics": results, "sample": sample_summary}, f, indent=2)

    flat = {k: v for k, v in results.items() if k != "per_class"}
    pd.DataFrame([flat]).to_csv(out_dir / "baseline_metrics.csv", index=False)
    pd.DataFrame(results["per_class"]).to_csv(out_dir / "baseline_per_class.csv", index=False)

    print(pd.DataFrame([flat]).T.to_string(header=False))
    print()
    print(pd.DataFrame(results["per_class"]).to_string(index=False))
    print(f"\n  Saved to: {out_dir}")


# ==========================================
# Step 8: prediction grid
# ==========================================
def save_prediction_grid(weights: Path, test_images: Path, out_path: Path,
                         n: int, conf: float, seed: int, device: str) -> None:
    from ultralytics import YOLO

    print(f"\n[6/6] Predicting on {n} random test images")
    images = sorted(p for p in test_images.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    random.seed(seed)
    picks = random.sample(images, min(n, len(images)))

    model = YOLO(str(weights))
    preds = model.predict(source=[str(p) for p in picks], conf=conf,
                          device=device, verbose=False)

    cols = 3
    rows = (len(preds) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
    axes = axes.flatten() if hasattr(axes, "flatten") else [axes]

    for ax, pred, path in zip(axes, preds, picks):
        ax.imshow(cv2.cvtColor(pred.plot(), cv2.COLOR_BGR2RGB))  # plot() returns BGR
        ax.set_title(f"{path.name[:40]}  ({len(pred.boxes)} det)", fontsize=8)
        ax.axis("off")
    for ax in axes[len(preds):]:
        ax.axis("off")

    plt.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ==========================================
# Main
# ==========================================
def parse_args():
    p = argparse.ArgumentParser(description="Helmet wearing detection - YOLOv8s baseline")
    p.add_argument("--dataset", type=Path, default=DEFAULTS["dataset"])
    p.add_argument("--sample-dir", type=Path, default=DEFAULTS["sample_dir"])
    p.add_argument("--data-yaml", type=Path, default=DEFAULTS["data_yaml"])
    p.add_argument("--runs-dir", type=Path, default=DEFAULTS["runs_dir"])

    p.add_argument("--n-train", type=int, default=1000)
    p.add_argument("--n-val", type=int, default=200)
    p.add_argument("--n-test", type=int, default=200)

    p.add_argument("--model", default="yolov8s.pt")
    p.add_argument("--run-name", default="YOLOv8s")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--device", default="mps", help="mps / cpu / 0 (CUDA)")
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--n-predict", type=int, default=6)

    p.add_argument("--skip-sample", action="store_true", help="reuse existing sample dir")
    p.add_argument("--skip-train", action="store_true", help="evaluate existing best.pt")
    return p.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)

    if args.skip_sample:
        print("\n[1/6] Skipping sampling (using existing sample dir)")
        sample_summary = {}
    else:
        sample_summary = build_sample(
            args.dataset, args.sample_dir,
            {"train": args.n_train, "val": args.n_val, "test": args.n_test},
            args.seed,
        )
    write_data_yaml(args.sample_dir, args.data_yaml)

    weights = args.runs_dir / args.run_name / "weights" / "best.pt"
    if args.skip_train:
        print(f"\n[3/6] Skipping training (using {weights})")
    else:
        weights = train(args)
    if not weights.exists():
        raise FileNotFoundError(f"Weights not found: {weights}")

    results = evaluate(args, weights)
    out_dir = args.runs_dir / args.run_name
    save_metrics(results, sample_summary, out_dir)
    save_prediction_grid(weights, args.sample_dir / "test" / "images",
                         out_dir / "test_predictions.png",
                         args.n_predict, args.conf, args.seed, args.device)

    print("\nBaseline pipeline finished.")


if __name__ == "__main__":
    main()