"""Lab 04: connect one project input to a traceable response.

Helmet Wearing Detection. One construction-site image goes through the Lab 02
YOLOv8s baseline and the same status rules the Streamlit app uses
(app/helmet_utils.py). The response and a simulated "unavailable" attempt are
saved to reports/inference_demo.jsonl.

Run from the repository root:
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

MODEL_VERSION = "baseline-v1"   # YOLOv8s, 25 epochs (test mAP50 0.660)
RULE_VERSION = "decision-v1"    # status rules in app/helmet_utils.py, cutoffs below

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
    """Load the Lab 02 baseline weights once."""
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
    """Return one prediction by reusing the Lab 02 model and the app's rules.

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


def write_evidence(records: list[dict[str, Any]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )


def main() -> None:
    global WEIGHTS
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-ref", required=True,
                        help="Path to one site image, relative to the repository root")
    parser.add_argument("--weights", type=Path, default=None,
                        help="Trained weights (default: models/helmet_yolov8s.pt, "
                             "then reports/runs/detect/YOLOv8s/weights/best.pt)")
    parser.add_argument("--output", default="reports/inference_demo.jsonl")
    args = parser.parse_args()

    WEIGHTS = args.weights or next((p for p in WEIGHTS_CANDIDATES if p.is_file()), None)

    # Load the model before timing, so latency_ms measures one prediction
    # rather than the one-off model load.
    try:
        load_model()
    except Exception as error:
        print(f"model not loaded: {error}")

    records = [
        run_attempt(args.input_ref, simulate_unavailable=False),
        run_attempt(args.input_ref, simulate_unavailable=True),
    ]
    output = Path(args.output)
    if not output.is_absolute():
        output = ROOT / output
    write_evidence(records, output)

    for record in records:
        print(json.dumps(record, ensure_ascii=False))
    print(f"Saved {len(records)} records to {output}")


if __name__ == "__main__":
    main()
