# Helmet Wearing Detection

Object detection of helmet use on construction sites with YOLOv8. The model finds heads and helmets in an image and sorts each one into one of three classes, so that workers without a helmet can be flagged automatically.

## Problem

Head injuries are a major hazard on construction sites, and checking helmet use by hand does not scale across cameras and shifts. This project trains a detector that answers, for each person in a frame, whether a helmet is being worn, missing, or present but not worn.

## Classes

| ID | Class | Meaning |
|---|---|---|
| 0 | `Head_without_helmet` | A visible head with no helmet (violation) |
| 1 | `Helmet_only` | A helmet that is not on a head |
| 2 | `Helmet_with_head` | A head wearing a helmet (compliant) |

The order of this table must match the class IDs in the label files. It is also used by `data.yaml` and by every drawing or evaluation step. Earlier versions of the notebook had the names in the wrong order, so boxes were labelled incorrectly.

## Dataset

Roboflow export `WearHelmetDetection.v1i.yolov8`, in YOLO format (one `.txt` label file per image, with normalised `class x_center y_center width height`).

| Split | Images | Objects | Head_without_helmet | Helmet_only | Helmet_with_head |
|---|---|---|---|---|---|
| Train | 2,564 | 12,233 | 3,050 (24.9%) | 317 (2.6%) | 8,866 (72.5%) |
| Valid | 986 | 4,766 | 1,310 (27.5%) | 93 (2.0%) | 3,363 (70.6%) |
| Test | 394 | 1,677 | 326 (19.4%) | 34 (2.0%) | 1,317 (78.5%) |

Findings from the exploratory analysis:

- **Class imbalance.** `Helmet_with_head` makes up over 70% of objects, while `Helmet_only` is only about 2%. All three splits follow a similar distribution.
- **Objects per image.** The mean is 4.77, the median is 3 and the maximum is 42 (train split). Most images have one to four objects, but a small set of crowded scenes contains many small, overlapping heads.
- **Image size.** All sampled images are about 416 × 416 pixels (415–416 px on each side).
- **Label integrity.** Every image has a label file and every label file has an image.

## Baseline

To keep training fast on a laptop, the baseline uses a random subset of the data, drawn with seed 42:

| Split | Sample size |
|---|---|
| Train | 1,000 |
| Val | 200 |
| Test | 200 |

Each sample keeps the class distribution of its full split to within about 2 percentage points per class:

| Split | Head_without_helmet | Helmet_only | Helmet_with_head |
|---|---|---|---|
| Train sample | 23.5% (full 24.9%) | 2.7% (full 2.6%) | 73.8% (full 72.5%) |
| Val sample | 25.5% (full 27.5%) | 2.3% (full 2.0%) | 72.2% (full 70.6%) |
| Test sample | 18.6% (full 19.4%) | 1.7% (full 2.0%) | 79.6% (full 78.5%) |

Model: **YOLOv8s** (pretrained `yolov8s.pt`), 10 epochs, image size 640, batch 16, Apple `mps` device.

### Test results (200 images, 918 objects)

| Class | Precision | Recall | mAP50 | mAP50-95 |
|---|---|---|---|---|
| **All** | 0.767 | 0.750 | 0.774 | 0.601 |
| Head_without_helmet | 0.870 | 0.819 | 0.863 | 0.651 |
| Helmet_only | 0.514 | 0.562 | 0.512 | 0.428 |
| Helmet_with_head | 0.919 | 0.868 | 0.948 | 0.724 |

Model size: 21.5 MB (`best.pt`).

The two common classes are detected well. `Helmet_only` is the weak point: it has only 16 test instances and too few training examples for the model to learn it reliably.

## Project structure

```
Helmet-wearing-detection/
├── data/
│   ├── WearHelmetDetection.v1i.yolov8/   # full Roboflow export
│   │   ├── train/{images,labels}
│   │   ├── valid/{images,labels}
│   │   └── test/{images,labels}
│   └── HWD_SAMPLE/                       # sampled subset (created by the pipeline)
├── notebook/
│   ├── Helmet_wear_detection.ipynb       # EDA, sampling, training, evaluation
│   └── data.yaml                         # training config (created by the pipeline)
├── reports/
│   └── runs/detect/YOLOv8s/              # weights, curves, metrics, predictions
├── baseline_pipeline.py                  # end-to-end baseline as one script
└── README.md
```

## Getting started

### Requirements

- Python 3.10 or later
- `ultralytics`, `opencv-python`, `pandas`, `matplotlib`, `seaborn`, `pyyaml`, `tqdm`, `pillow`

```bash
python -m venv .venv
source .venv/bin/activate
pip install ultralytics opencv-python pandas matplotlib seaborn pyyaml tqdm pillow
```

### Run the baseline

```bash
python baseline_pipeline.py
```

This runs every step: sampling, writing `data.yaml`, training, test evaluation, saving metrics and drawing a prediction grid. Useful options:

| Option | Default | Purpose |
|---|---|---|
| `--epochs` | 10 | Number of training epochs |
| `--imgsz` | 640 | Training and evaluation image size |
| `--batch` | 16 | Batch size |
| `--device` | `mps` | `mps`, `cpu`, or a CUDA index such as `0` |
| `--model` | `yolov8s.pt` | Starting weights (for example `yolov8n.pt`, `yolov8m.pt`) |
| `--run-name` | `YOLOv8s` | Output folder name under `reports/runs/detect/` |
| `--n-train / --n-val / --n-test` | 1000 / 200 / 200 | Sample sizes |
| `--skip-sample` | off | Reuse the existing `HWD_SAMPLE` folder |
| `--skip-train` | off | Evaluate an existing `best.pt` without training |

Paths default to the author's machine. Pass `--dataset`, `--sample-dir`, `--data-yaml` and `--runs-dir` to run it somewhere else.

### Outputs

Written to `reports/runs/detect/<run-name>/`:

- `weights/best.pt`, `weights/last.pt`: trained weights
- `results.csv`, `results.png`: training curves
- `baseline_metrics.json` and `baseline_metrics.csv`: overall test metrics, inference time and FPS
- `baseline_per_class.csv`: test metrics for each class
- `test_predictions.png`: predictions on random test images

## Next steps

- Train on the full dataset instead of the 1,000-image sample, and for more than 10 epochs.
- Improve `Helmet_only`: oversample images that contain it, apply class-weighted loss, or collect more examples.
- Compare model sizes (YOLOv8n / s / m) on accuracy, inference time and model size.
- Check whether a higher input resolution helps with the small, distant heads in crowded scenes.

## Author

Yikwa Sandrakh (drakejkw), Group AIb-23-1, Baikal School of BRICS, Irkutsk National Research Technical University.