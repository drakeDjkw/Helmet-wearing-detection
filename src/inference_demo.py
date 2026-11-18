"""Lab 04 + Lab 05: validate a request, then connect it to a traceable response.

Helmet Wearing Detection. Lab 05 (default): three requests (pass, error,
warning) go through src/input_validation.py first. Only a passing request
reaches project_predict(), which runs the YOLOv8s baseline and the same status
rules the Streamlit app uses (app/helmet_utils.py). The combined trace is
saved to reports/validation_demo.jsonl.

Run from the repository root:
    python src/inference_demo.py

Lab 04 (one image, plus a simulated "unavailable" attempt, written to
reports/inference_demo.jsonl):
    python src/inference_demo.py --input-ref data/WearHelmetDetection.v1i.yolov8/test/images/hard_hat_workers1116_png.rf.7cc65945da90d7396b636cd982363c8a.jpg
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))

import helmet_utils as hu  # noqa: E402  same input checks and status rules as the app

sys.path.insert(0, str(ROOT / "src"))
from input_validation import load_contract, route_request  # noqa: E402

MODEL_VERSION = "baseline-v1"   # YOLOv8s, 25 epochs (test mAP50 0.660)
RULE_VERSION = "decision-v1"    # status rules in app/helmet_utils.py, cutoffs below

# Runtime identity of the deterministic preprocessing used for prediction.
# letterbox640-v1 = cv2.imdecode to 3-channel BGR (helmet_utils.decode_image_bytes),
# then Ultralytics letterbox resize to IMGSZ=640 and /255 scaling inside
# model.predict(). Training augmentation (mosaic, flips, HSV) never runs here.
# Defined beside the inference code and checked against config/input_contract.json;
# it is never read from the incoming request.
PREPROCESSING_ID = "letterbox640-v1"

# Actions decided in Lab 03 (docs/project_decisions.md).
NORMAL_ACTION = "show frame status to the safety officer; alert if VIOLATION"
REVIEW_ACTION = "put frame in the safety officer's review queue (check by eye)"
UNAVAILABLE_ACTION = "show 'detection offline' and check the area by eye"

# Lab 02 weights: copied to models/ as the README describes, else the run folder.
WEIGHTS_CANDIDATES = [
    ROOT / "models" / "helmet_yolov8s.pt",
    ROOT / "reports" / "runs" / "detect" / "YOLOv8s" / "weights" / "best.pt",
]
IMGSZ = 640

# decision-v1 (provisional: Lab 03 says both cutoffs are not set yet).
ALERT_CONF = 0.50    # Head_without_helmet at or above this -> VIOLATION
REVIEW_CONF = 0.25   # between REVIEW_CONF and ALERT_CONF -> UNCERTAIN (review)

WEIGHTS: Path | None = None   # set in main()
_model = None


def load_model():
    """Load the baseline weights once."""
    global _model
    if _model is None:
        if WEIGHTS is None or not WEIGHTS.is_file():
            raise FileNotFoundError(
                "baseline weights not found; expected one of: "
                + ", ".join(str(p.relative_to(ROOT)) for p in WEIGHTS_CANDIDATES))
        from ultralytics import YOLO
        _model = YOLO(str(WEIGHTS))
    return _model


def project_predict(input_ref: str) -> dict[str, Any]:
    """Return one prediction by reusing the model and the app's rules.

    Shape: {"result": {...}, "needs_review": bool}
    """
    path = Path(input_ref)
    if not path.is_absolute():
        path = ROOT / path

    # First input check (Lab 03): empty, undecodable or < 32 px -> rejected.
    try:
        image_bgr = hu.decode_image_bytes(path.read_bytes() if path.is_file() else b"")
    except hu.InvalidInput as reason:
        return {"result": {"frame_status": "REJECTED", "reason": str(reason)},
                "needs_review": True}

    model = load_model()
    # Preprocessing (letterbox resize to 640 px, normalisation) happens inside
    # model.predict(), the same step the baseline uses for its prediction grid.
    fr = hu.detect(model, image_bgr, alert_conf=ALERT_CONF,
                   review_conf=REVIEW_CONF, imgsz=IMGSZ)

    result = {
        "frame_status": fr.status,            # VIOLATION / UNCERTAIN / NO_PEOPLE / COMPLIANT
        "n_without_helmet": fr.n_without,
        "n_with_helmet": fr.n_with,
        "n_check_by_eye": fr.n_uncertain,
        "n_helmet_only": fr.n_helmet_only,    # shown, not used for alerts
        "detections": [
            {"class": d.label, "confidence": round(d.score, 3),
             "box_xyxy": d.box, "check_by_eye": d.uncertain}
            for d in fr.detections
        ],
    }
    # Lab 03: an uncertain frame goes to review, never counted as compliant.
    return {"result": result, "needs_review": fr.status == hu.UNCERTAIN}


def run_attempt(input_ref: str, *, simulate_unavailable: bool) -> dict[str, Any]:
    started = time.perf_counter()

    if simulate_unavailable:
        status = "unavailable"
        result = None
        action = UNAVAILABLE_ACTION
    else:
        try:
            prediction = project_predict(input_ref)
        except Exception as error:   # real failure, e.g. weights missing
            print(f"prediction failed: {error}")
            status, result, action = "unavailable", None, UNAVAILABLE_ACTION
        else:
            result = prediction["result"]
            needs_review = bool(prediction.get("needs_review", False))
            status = "review" if needs_review else "ok"
            action = REVIEW_ACTION if needs_review else NORMAL_ACTION

    return {
        "request_id": str(uuid4()),
        "input_ref": input_ref,
        "status": status,
        "result": result,
        "model_version": MODEL_VERSION,
        "rule_version": RULE_VERSION,
        "latency_ms": round((time.perf_counter() - started) * 1000, 1),
        "action": action,
        "simulated": simulate_unavailable,
    }


# ------------------------------------------------------------------
# Lab 05: validation before prediction
# ------------------------------------------------------------------
LAB04_IMAGE = ("data/WearHelmetDetection.v1i.yolov8/test/images/"
               "hard_hat_workers1116_png.rf.7cc65945da90d7396b636cd982363c8a.jpg")

# One request per status. Values are what a caller would send; the validator
# does not open the files.
DEMO_CASES: list[tuple[str, dict[str, Any]]] = [
    ("normal test image", {
        "input_ref": LAB04_IMAGE, "format": "jpg", "width": 416, "height": 416,
        "source": "dataset-test-sample",
    }),
    ("invalid request", {
        # Missing reference and an unsupported format (errors), from an
        # unevaluated source (warning): error must win.
        "input_ref": "", "format": "gif", "width": 416, "height": 416,
        "source": "app-image-upload",
    }),
    ("readable image from an unevaluated source", {
        "input_ref": "uploads/site_a_gate_0815.jpg", "format": "jpg",
        "width": 1280, "height": 720, "source": "app-image-upload",
    }),
]


def predict_from_request(request: dict[str, Any]) -> dict[str, Any]:
    """Adapter: route_request() passes the whole request, Lab 04 takes input_ref."""
    return project_predict(request["input_ref"])


def run_validated(case: str, request: dict[str, Any], contract: dict[str, Any]) -> dict[str, Any]:
    request = {"request_id": str(uuid4()), **request}
    record = route_request(request, predict_from_request, contract, PREPROCESSING_ID)

    # What happens after prediction uses the Lab 04 actions.
    if record["prediction_status"] == "returned":
        needs_review = bool(record["prediction"].get("needs_review", False))
        after = REVIEW_ACTION if needs_review else NORMAL_ACTION
    elif record["prediction_status"] == "failed":
        after = UNAVAILABLE_ACTION
    else:
        after = None

    return {"case": case, **record,
            "after_prediction_action": after,
            "model_version": MODEL_VERSION,
            "rule_version": RULE_VERSION}


def run_lab05(output: Path) -> None:
    contract = load_contract()
    # Load the model before timing, so latency_ms measures one prediction.
    try:
        load_model()
    except Exception as error:
        print(f"model not loaded: {error}")

    records = [run_validated(case, request, contract) for case, request in DEMO_CASES]
    write_evidence(records, output)

    for r in records:
        failed = [c["rule"] for c in r["checks"] if c["status"] != "pass"]
        print(f"{r['validation_status']:<8} {r['action']:<17} "
              f"prediction_called={str(r['prediction_called']):<5} "
              f"checks_not_passed={failed}  request_id={r['request_id']}")
    print(f"Saved {len(records)} records to {output}")


def write_evidence(records: list[dict[str, Any]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )


def main() -> None:
    global WEIGHTS
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-ref", default=None,
                        help="Lab 04 mode: path to one site image, relative to the repository root")
    parser.add_argument("--weights", type=Path, default=None,
                        help="Trained weights (default: models/helmet_yolov8s.pt, "
                             "then reports/runs/detect/YOLOv8s/weights/best.pt)")
    parser.add_argument("--output", default=None,
                        help="Default: reports/validation_demo.jsonl "
                             "(Lab 04 mode: reports/inference_demo.jsonl)")
    args = parser.parse_args()

    WEIGHTS = args.weights or next((p for p in WEIGHTS_CANDIDATES if p.is_file()), None)

    default_out = "reports/inference_demo.jsonl" if args.input_ref else "reports/validation_demo.jsonl"
    output = Path(args.output or default_out)
    if not output.is_absolute():
        output = ROOT / output

    if args.input_ref is None:
        run_lab05(output)
        return

    # Lab 04 mode, unchanged.
    try:
        load_model()
    except Exception as error:
        print(f"model not loaded: {error}")

    records = [
        run_attempt(args.input_ref, simulate_unavailable=False),
        run_attempt(args.input_ref, simulate_unavailable=True),
    ]
    write_evidence(records, output)

    for record in records:
        print(json.dumps(record, ensure_ascii=False))
    print(f"Saved {len(records)} records to {output}")


if __name__ == "__main__":
    main()
