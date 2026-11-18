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

Roboflow export `WearHelmetDetection.v1i.yolov8`, in YOLO format (one `.txt` label file per image, with normalised `class x_center y_center width height`). The class names are read from the export's `data.yaml`.

The file names show two sources:

- **Public hard-hat images** (`hard_hat_workers<N>_png.rf.<hash>.jpg`). Original source and licence: **[TO CONFIRM — see `data/WearHelmetDetection.v1i.yolov8/README.roboflow.txt`]**.
- **Web images collected by the author** (`helmet_with_head_<N>`, `head_without_helmet_<N>`, `helmet_only_<N>`). A Colab script used `icrawler` to send 48 search phrases (English, Russian, Chinese) to Bing and Baidu. It downloaded 9,348 images, and perceptual-hash de-duplication (Hamming distance ≤ 5, across all classes) kept 5,927. The folder name was only a hint: every image was annotated with all three classes in Roboflow (Auto Label, then manual review). These images are for research use only and must not be redistributed.

The dataset is not stored in this repository. Download it and place it at `data/WearHelmetDetection.v1i.yolov8/`.

**Removed source.** Files whose names are only digits (`^\d+_jpg\.rf\.`) carried wrong labels. The notebook moves them out of `valid` (1,417 images) and `test` (710 images) into `data/WearHelmetDetection.v1i.yolov8_removed_source/` before any analysis. The training split is **not** filtered by this rule yet.

After that step:

| Split | Images | Objects | Head_without_helmet | Helmet_only | Helmet_with_head |
|---|---|---|---|---|---|
| Train | 7,262 | 19,715 | 4,821 (24.5%) | 3,476 (17.6%) | 11,418 (57.9%) |
| Valid | 1,666 | 5,923 | 1,581 (26.7%) | 504 (8.5%) | 3,838 (64.8%) |
| Test | 1,026 | 2,788 | 581 (20.8%) | 546 (19.6%) | 1,661 (59.6%) |
| **Total** | **9,954** | **28,426** | 6,983 (24.6%) | 4,526 (15.9%) | 16,917 (59.5%) |

Findings from the exploratory analysis (`notebook/Helmet_wear_detection.ipynb`):

- **Class imbalance.** `Helmet_with_head` is the largest class in every split (58–65% of objects). The share of `Helmet_only` varies between splits: 17.6% in train, 8.5% in validation, 19.6% in test.
- **Objects per image.** In the training split the mean is 2.71, the median is 1 and the maximum is 97. The largest group of images (38.0%) contains exactly one object.
- **Empty images.** 1,362 training images (18.8%) have an empty label file, so the model also sees scenes without any of the three classes. Validation and test contain 151 and 160 such images.
- **Image size.** In a random sample of 3,000 training images, widths range from 320 to 7,360 px and heights from 320 to 6,000 px, with 829 distinct sizes. The most common are 416 × 415 and 416 × 416 (the hard-hat images), then 1,500 × 1,101 and 800 × 800. YOLO letterboxes every image to 640 px.
- **Label integrity.** Every image has a label file, no label line is malformed, and every class ID is valid. (The training image folder holds one extra non-image file, which is ignored.)

## Baseline

The baseline uses a fixed-seed random sample (`random_state=42`, drawn from a sorted file list) of each split. The original split boundaries are kept.

| Split | Sample | Objects | Head_without_helmet | Helmet_only | Helmet_with_head |
|---|---|---|---|---|---|
| Train | 3,000 | 8,074 | 25.8% (full 24.5%) | 16.6% (full 17.6%) | 57.5% (full 57.9%) |
| Val | 500 | 1,910 | 26.2% (full 26.7%) | 9.9% (full 8.5%) | 63.9% (full 64.8%) |
| Test | 300 | 914 | 24.2% (full 20.8%) | 27.4% (full 19.6%) | 48.5% (full 59.6%) |

The training and validation samples stay within 1.5 percentage points of their splits. The test sample departs further (−11.1 points for `Helmet_with_head`), because whole images are drawn and a few crowded images shift a 300-image sample. Ultralytics' scan registered 2,999 of the 3,000 training images (570 without objects), and 48 of the 300 test images have no objects.

**Model:** YOLOv8s (pretrained `yolov8s.pt`, COCO head replaced by a 3-class head), 25 epochs, image size 640, batch 16, seed 42, deterministic, mixed precision. `optimizer=auto` chose AdamW (lr 0.001429, momentum 0.9), with a 3-epoch warm-up and linear decay. Default Ultralytics augmentation (mosaic, switched off for the last 10 epochs; translate 0.1; scale 0.5; horizontal flip 0.5; HSV 0.015 / 0.7 / 0.4). Trained on an Apple M4 (MPS) in 2.79 hours. The best checkpoint by fitness (0.1 × mAP50 + 0.9 × mAP50-95) is **epoch 24**: validation mAP50 0.788, mAP50-95 0.595.

### Test results (300 images, 914 objects)

Scored once with `best.pt` on the test sample (`split="test"`).

| Class | Images | Instances | Precision | Recall | mAP50 | mAP50-95 |
|---|---|---|---|---|---|---|
| **All** | 300 | 914 | 0.795 | 0.800 | 0.817 | 0.604 |
| Head_without_helmet | 58 | 221 | 0.754 | 0.735 | 0.748 | 0.495 |
| Helmet_only | 72 | 250 | 0.758 | 0.780 | 0.795 | 0.597 |
| Helmet_with_head | 164 | 443 | 0.873 | 0.885 | 0.909 | 0.721 |

| Model size | Inference time | Throughput |
|---|---|---|
| 21.5 MB | 213.2 ms per image (CPU of the Apple M4, batched evaluation, model inference only) | about 4.7 FPS (213.7 ms including pre- and post-processing) |

The test evaluation did not set a device, so it ran on the CPU. During validation on MPS, inference took 9.8 ms and post-processing 20.9 ms per image (about 32 FPS).

Sources: notebook output, `reports/runs/detect/YOLOv8s/results.csv` (training) and `reports/runs/detect/val/` (test curves and confusion matrix).

What the results show:

- **Heads with a helmet are detected best** (0.909 mAP50). `Helmet_only`, which was unusable with the earlier data (0.157 mAP50 on 19 boxes), now reaches 0.795 on 250 test boxes.
- **The violation class is now the weakest** (0.748 mAP50, 0.495 mAP50-95). From the test confusion matrix: of 221 heads without a helmet, 164 were detected correctly, 45 were missed as background and 12 were labelled `Helmet_with_head`. That is a **false-negative rate of 57 / 221 = 25.8%**, against the proposed acceptance threshold of 10% ([decisions](docs/project_decisions.md)). The model does not meet it yet.
- **`Helmet_only` stays out of alerts by design.** A helmet that nobody wears says nothing about whether a worker is protected, so the app shows these boxes but does not count them.

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
├── config/
│   └── input_contract.json               # Lab 05 input contract
├── docs/
│   ├── project_brief.md                  # user, task, data and first results
│   ├── split_strategy.md                 # how data is split and evaluated
│   ├── project_decisions.md              # required behaviour, acceptance check
│   └── inference_architecture.drawio     # inference path diagram
├── models/                               # not in Git: put helmet_yolov8s.pt here
├── notebook/
│   └── Helmet_wear_detection.ipynb       # EDA, sampling, training, evaluation
├── reports/
│   ├── baseline_metrics.json             # earlier 1,000-image run (to be regenerated)
│   ├── inference_demo.jsonl              # records written by the Lab 04 demo
│   ├── validation_demo.jsonl             # Lab 05 validation trace
│   └── runs/detect/
│       ├── YOLOv8s/                      # training curves, results.csv, args.yaml
│       └── val/                          # test-sample evaluation, confusion matrix
├── src/
│   ├── baseline_pipeline.py              # end-to-end baseline as one script (see note below)
│   ├── inference_demo.py                 # Lab 04 demo + Lab 05 validated demo
│   └── input_validation.py               # Lab 05 request validator
├── tests/
│   └── test_input_validation.py          # Lab 05 tests (standard library only)
├── requirements.txt
└── README.md
```

## Getting started

### Install

The baseline was trained with Python 3.14, Ultralytics 8.4.92 and PyTorch 2.14.1. For the app, Python 3.10–3.12 is recommended, because the webcam page needs PyAV (`av`), which may not yet have ready-made packages for Python 3.14.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install -r app/requirements-app.txt
```

### Train and evaluate the baseline

The reported baseline was produced by **`notebook/Helmet_wear_detection.ipynb`**. Run it top to bottom: it removes the mislabelled source, runs the EDA, draws the 3,000 / 500 / 300 sample into `data/HWD_SAMPLE/`, writes `notebook/data.yaml`, trains, and evaluates on the test sample.

`src/baseline_pipeline.py` runs the same steps as one script, but it does **not** yet remove the mislabelled source, and its defaults (1,000 / 200 / 200 images, 10 epochs, evaluation on MPS) differ from the notebook. To get close to the notebook settings:

```bash
python src/baseline_pipeline.py --epochs 25 --n-train 3000 --n-val 500 --n-test 300 --run-name YOLOv8s_e25
```

Give each run its own `--run-name`, so its results are not overwritten.

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

Detections below the review cutoff are discarded. Inputs that cannot be read (empty files, non-image files, images or frames smaller than 32 px) are rejected with a reason. Both cutoffs (defaults 0.50 and 0.25) are provisional and are set with sliders in the app. The video page processes every 5th frame by default and merges flagged frames with the same status that are less than 1 s apart into one event.

## inference demo

The demo sends one site image through the YOLOv8s baseline and the same
status rules the app uses (`app/helmet_utils.py`). It records two attempts: a
normal one and a simulated "inference unavailable" one.

It needs the trained weights at `models/helmet_yolov8s.pt` (see
[Run the app](#run-the-app)) or `reports/runs/detect/YOLOv8s/weights/best.pt`.
Use `--weights` to point to another file. Install both requirement files
first, because the demo imports `app/helmet_utils.py`, which uses Streamlit.

From the repository root:

```bash
python src/inference_demo.py --input-ref data/WearHelmetDetection.v1i.yolov8/test/images/hard_hat_workers1116_png.rf.7cc65945da90d7396b636cd982363c8a.jpg
```

This creates **`reports/inference_demo.jsonl`** with two lines:

| Line | status | simulated | action |
|---|---|---|---|
| 1 | `ok` (VIOLATION, COMPLIANT or NO_PEOPLE) | `false` | show frame status to the safety officer; alert if VIOLATION |
| 1 | `review` (UNCERTAIN, or input rejected) | `false` | put frame in the safety officer's review queue (check by eye) |
| 2 | `unavailable` | `true` | show 'detection offline' and check the area by eye |

**Decision rule (`decision-v1`, provisional):**

- A `Head_without_helmet` box at 0.50 or more makes the frame a VIOLATION.
- A box between 0.25 and 0.50, or one head with both labels, makes it
  UNCERTAIN, which goes to review.
- `Helmet_only` is not used for alerts.
- An input that fails the check (empty, not an image, under 32 px) is
  rejected with a reason and goes to review.

The path is drawn in
[`docs/inference_architecture.drawio`](docs/inference_architecture.drawio);
open it at <https://app.diagrams.net>.

## Input validation

Before prediction, each request is checked against
[`config/input_contract.json`](config/input_contract.json) by
[`src/input_validation.py`](src/input_validation.py). The checks read only the
values in the request (`input_ref`, `format`, `width`, `height`, `source`) and
the runtime preprocessing identity; they do not open or decode the file.
Decoding is still checked inside `project_predict()`.

| Rule | Kind | Fails as | Why |
|---|---|---|---|
| `input_reference_present` | structural | error | prediction needs a reference |
| `format_supported` | structural | error | normal input is JPG or PNG (Lab 03) |
| `dimensions_minimum` | structural | error | the app rejects images under 32 px |
| `source_registered` | project-specific | error | an unknown source cannot be traced |
| `source_evaluated` | review | warning | frames from unevaluated sources go to the officer (Lab 03 failure cases) |
| `preprocessing_identity` | structural | error | runtime `letterbox640-v1` must match the contract |

`pass` calls `project_predict()`, `warning` returns `route_for_review` without
predicting, and `error` returns `reject_input`. The most severe result wins:
error > warning > pass.

```bash
python src/inference_demo.py                        # writes reports/validation_demo.jsonl
python -m unittest tests/test_input_validation.py
```

Each line of the trace keeps one `request_id` with the checks
(`rule`, `observed`, `status`, `action`), `validation_status`, `action`,
`prediction_called` and the prediction outcome.

## Next steps

- Apply the mislabelled-source filter to the training split too, and add the filter to `src/baseline_pipeline.py`.
- Check for leakage between splits: whether copies of the same source photo (`hard_hat_workers<N>`, or one web photo exported several times) appear in both training and test.
- Tune the alert and review cutoffs on the validation sample, aiming for the [acceptance threshold](docs/project_decisions.md) of under 10% missed violations (now 25.8%).
- Train on the full training set (7,262 images), and compare YOLOv8n with YOLOv8s and 416 px with 640 px input.
- Measure end-to-end single-frame time on the target device, false alarms on frames without people, and results on images from real site cameras.
- Update `docs/split_strategy.md`, `docs/project_decisions.md` and `reports/baseline_metrics.json`, which still describe the earlier 1,000-image run.

## Acknowledgements

## Author

Yikwa Sandrakh (drakejkw), Group AIb-23-1, Baikal School of BRICS, Irkutsk National Research Technical University.
