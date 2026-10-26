# Helmet Wearing Detection

Object detection of helmet use on construction sites with YOLOv8. The model finds heads and helmets in an image and sorts each one into one of three classes, so that workers without a helmet can be flagged for a safety officer to review. A Streamlit app runs the model on images, recorded video and a webcam.

## Problem

Checking helmet use by walking the site or watching camera feeds by eye does not scale across work areas and shifts. This project trains a detector that answers, for each person in a frame, whether a helmet is being worn, so that a safety officer only needs to review the flagged frames.

## Classes

| ID | Class | Meaning | Used for alerts |
|---|---|---|---|
| 0 | `Head_without_helmet` | A visible head with no helmet | Yes: raises an alert |
| 1 | `Helmet_only` | A helmet that is not on a head | No: shown only (see [decisions](docs/project_decisions.md)) |
| 2 | `Helmet_with_head` | A head wearing a helmet | Counted as compliant |

The order of this table must match the class IDs in the label files. It is also used by `data.yaml` and by every drawing or evaluation step.

## Dataset

Roboflow export `WearHelmetDetection.v1i.yolov8`, in YOLO format (one `.txt` label file per image, with normalised `class x_center y_center width height`). Original source: **[SOURCE TO CONFIRM — see `data/WearHelmetDetection.v1i.yolov8/README.roboflow.txt`]**.

The dataset is not stored in this repository. Download it and place it at `data/WearHelmetDetection.v1i.yolov8/`.

| Split | Images | Objects | Head_without_helmet | Helmet_only | Helmet_with_head |
|---|---|---|---|---|---|
| Train | 2,564 | 12,233 | 3,050 (24.9%) | 317 (2.6%) | 8,866 (72.5%) |
| Valid | 986 | 4,766 | 1,310 (27.5%) | 93 (2.0%) | 3,363 (70.6%) |
| Test | 394 | 1,677 | 326 (19.4%) | 34 (2.0%) | 1,317 (78.5%) |

Findings from the exploratory analysis (`notebook/Helmet_wear_detection.ipynb`):

- **Class imbalance.** `Helmet_with_head` makes up over 70% of objects, while `Helmet_only` is only about 2%.
- **Objects per image.** The mean is 4.77, the median is 3 and the maximum is 42 (train split). Many scenes contain several small, overlapping heads.
- **Image size.** All sampled images are about 416 × 416 px (415–416 px on each side).
- **No empty images.** Every training image contains at least one object, so the model has never seen a scene without people.
- **Label integrity.** Every image has a label file and every label file has an image.

## Baseline

The baseline uses a fixed-seed random sample (seed 42) of each split. Details are in [`docs/split_strategy.md`](docs/split_strategy.md).

| Split | Sample | Objects | Head_without_helmet | Helmet_only | Helmet_with_head |
|---|---|---|---|---|---|
| Train | 1,000 | 4,849 | 25.5% | 2.1% | 72.5% |
| Val | 200 | 991 | 24.5% | 2.8% | 72.7% |
| Test | 200 | 882 | 20.9% | 2.2% | 77.0% |

**Model:** YOLOv8s (pretrained `yolov8s.pt`), 25 epochs, image size 640, batch 16, seed 42. It was trained on an Apple M4 (MPS) in about 44 minutes.

### Test results (200 images, 882 objects)

| Class | Instances | Precision | Recall | mAP50 | mAP50-95 |
|---|---|---|---|---|---|
| **All** | 882 | 0.671 | 0.634 | 0.660 | 0.506 |
| Head_without_helmet | 184 | 0.847 | 0.783 | 0.865 | 0.653 |
| Helmet_only | 19 | 0.260 | 0.211 | 0.157 | 0.129 |
| Helmet_with_head | 679 | 0.907 | 0.909 | 0.957 | 0.734 |

| Model size | Inference time | Throughput |
|---|---|---|
| 21.5 MB | 2.67 ms per image (MPS, batched evaluation, model inference only) | 54.5 FPS (including pre- and post-processing) |

Source: [`reports/baseline_metrics.json`](reports/baseline_metrics.json).

What the results show:

- **The two classes that matter for safety are detected well.** `Helmet_with_head` reaches 0.957 mAP50 and `Head_without_helmet` reaches 0.865.
- **Violations are still missed.** Of the 184 test heads without a helmet, 32 were missed and 7 were labelled `Helmet_with_head`. That is a false-negative rate of 21.2%. This is the failure the [acceptance check](docs/project_decisions.md) targets.
- **`Helmet_only` is not usable yet.** It has 0.157 mAP50 on 19 test boxes, and it pulls the overall mAP50 down. The first version of the app does not use it for alerts.

## Project structure

```
Helmet-wearing-detection/
├── app/                                  # Streamlit app
│   ├── Home.py                           # entry page
│   ├── helmet_utils.py                   # model loading, status rules, drawing
│   ├── pages/
│   │   ├── 1_Image_Detection.py
│   │   ├── 2_Video_Detection.py
│   │   └── 3_Realtime_Detection.py
│   ├── sample_utils/download.py          # optional weights download
│   └── requirements-app.txt
├── data/                                 # not in Git
│   ├── WearHelmetDetection.v1i.yolov8/   # full Roboflow export
│   └── HWD_SAMPLE/                       # sampled subset (created by the pipeline)
├── docs/
│   ├── project_brief.md                  # user, task, data and first results
│   ├── split_strategy.md                 # how data is split and evaluated
│   └── project_decisions.md              # Lab 03: required behaviour, acceptance check
├── models/                               # not in Git: put helmet_yolov8s.pt here
├── notebook/
│   └── Helmet_wear_detection.ipynb       # EDA, sampling, training, evaluation
├── reports/
│   ├── baseline_metrics.json             # baseline test metrics
│   └── runs/detect/                      # curves, confusion matrices, predictions
├── src/
│   └── baseline_pipeline.py              # end-to-end baseline as one script
├── requirements.txt
└── README.md
```

## Getting started

### Install

Python 3.10–3.12 is recommended. The webcam page needs PyAV (`av`), which may not yet have ready-made packages for Python 3.14.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install -r app/requirements-app.txt
```

### Train and evaluate the baseline

```bash
python src/baseline_pipeline.py --epochs 25 --run-name YOLOv8s_e25
```

This runs every step: sampling, writing `data.yaml`, training, test evaluation, saving metrics and drawing a prediction grid. Give each run its own `--run-name`, so its results are not overwritten.

| Option | Default | Purpose |
|---|---|---|
| `--epochs` | 10 | Number of training epochs |
| `--imgsz` | 640 | Training and evaluation image size |
| `--batch` | 16 | Batch size |
| `--device` | `mps` | `mps`, `cpu`, or a CUDA index such as `0` |
| `--model` | `yolov8s.pt` | Starting weights (for example `yolov8n.pt`) |
| `--run-name` | `YOLOv8s` | Output folder under `reports/runs/detect/` |
| `--n-train / --n-val / --n-test` | 1000 / 200 / 200 | Sample sizes |
| `--skip-sample` | off | Reuse the existing `HWD_SAMPLE` folder |
| `--skip-train` | off | Evaluate an existing `best.pt` without training |
| `--dataset`, `--sample-dir`, `--data-yaml`, `--runs-dir` | author's paths | Set these when running on another machine |

Outputs in `reports/runs/detect/<run-name>/`:

- `weights/best.pt`: trained weights (not in Git)
- `results.csv`, `results.png`: training curves
- `baseline_metrics.json`, `baseline_metrics.csv`, `baseline_per_class.csv`: test metrics
- `test_predictions.png`: predictions on random test images

### Run the app

The model weights are not stored in Git. Copy your trained weights to `models/` first:

```bash
mkdir -p models
cp reports/runs/detect/YOLOv8s/weights/best.pt models/helmet_yolov8s.pt
streamlit run app/Home.py
```

To use weights stored somewhere else, set `HELMET_MODEL_PATH=/path/to/best.pt`. To download them from a GitHub Release, set `HELMET_MODEL_URL` instead.

| Page | Input | Output |
|---|---|---|
| Image Detection | JPG / PNG photo | Annotated image, frame status, detection table |
| Video Detection | MP4 / MOV / AVI | Annotated video and a list of moments to review |
| Realtime Detection | Webcam | Live annotated stream with frame status |

Each frame gets one of four statuses, following [`docs/project_decisions.md`](docs/project_decisions.md):

| Status | When |
|---|---|
| **Violation** | A `Head_without_helmet` box is at or above the alert cutoff |
| **Uncertain: check by eye** | A `Head_without_helmet` box is between the review and alert cutoffs, or a head received both labels |
| **No people detected** | No heads were found. This is never reported as compliant |
| **Compliant** | All detected heads wear helmets |

Inputs that cannot be read (empty files, non-image files, images smaller than 32 px) are rejected with a reason. Both cutoffs are provisional and are set with sliders in the app.

## Next steps

- Check for leakage between splits: whether copies of the same source photo (`hard_hat_workers<N>`) appear in both training and test.
- Tune the alert and review cutoffs on the validation sample, aiming for the [acceptance threshold](docs/project_decisions.md) on missed violations.
- Train on the full training set (2,564 images), and compare YOLOv8n with YOLOv8s and 416 px with 640 px input.
- Measure false alarms on frames without people, and test on images from real site cameras.

## Acknowledgements

## Author

Yikwa Sandrakh (drakejkw), Group AIb-23-1, Baikal School of BRICS, Irkutsk National Research Technical University.