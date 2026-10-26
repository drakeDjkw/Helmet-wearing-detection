"""
Shared detection logic for the Helmet Wearing Detection app.

All pages use this module, so the rules from docs/project_decisions.md
are applied the same way to images, videos and webcam frames:

    * VIOLATION      - at least one Head_without_helmet box >= alert cutoff
    * UNCERTAIN      - a Head_without_helmet box between review and alert cutoff,
                       or a head that received both a "with" and a "without" box
    * NO_PEOPLE      - no heads detected (never reported as compliant)
    * COMPLIANT      - heads detected, all with helmets
    * Helmet_only    - drawn, but not used for alerts or compliance counts
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
import streamlit as st

APP_DIR = Path(__file__).resolve().parent
ROOT = APP_DIR.parent

# Local weights. Override with the HELMET_MODEL_PATH environment variable.
MODEL_LOCAL_PATH = Path(os.environ.get("HELMET_MODEL_PATH",
                                       ROOT / "models" / "helmet_yolov8s.pt"))
# Optional: a GitHub Release URL for best.pt, used when the local file is missing.
MODEL_URL = os.environ.get("HELMET_MODEL_URL", "")

# Class names as trained (order comes from the model itself, these are checked)
HEAD_WITHOUT = "Head_without_helmet"
HELMET_ONLY = "Helmet_only"
HEAD_WITH = "Helmet_with_head"
EXPECTED_CLASSES = {HEAD_WITHOUT, HELMET_ONLY, HEAD_WITH}

# Colours in BGR (OpenCV drawing order)
CLASS_COLORS_BGR = {
    HEAD_WITHOUT: (0, 0, 230),      # red    - violation
    HELMET_ONLY: (0, 191, 255),     # amber  - not used for alerts
    HEAD_WITH: (0, 200, 0),         # green  - compliant
}
UNCERTAIN_COLOR_BGR = (0, 140, 255)  # orange - check by eye

# Frame status labels
VIOLATION = "VIOLATION"
UNCERTAIN = "UNCERTAIN"
NO_PEOPLE = "NO_PEOPLE"
COMPLIANT = "COMPLIANT"

STATUS_TEXT = {
    VIOLATION: "Violation: worker without helmet",
    UNCERTAIN: "Uncertain: check by eye",
    NO_PEOPLE: "No people detected",
    COMPLIANT: "All detected workers wear helmets",
}

DEFAULT_ALERT_CONF = 0.50
DEFAULT_REVIEW_CONF = 0.25
CONFLICT_IOU = 0.5   # a "with" and a "without" box this similar = same head


class InvalidInput(Exception):
    """Raised when an input cannot be used; the message is shown to the user."""


@dataclass
class Detection:
    label: str
    score: float
    box: List[int]                 # x1, y1, x2, y2 in original image pixels
    uncertain: bool = False


@dataclass
class FrameResult:
    status: str
    detections: List[Detection] = field(default_factory=list)
    n_without: int = 0             # confident heads without helmet
    n_with: int = 0                # heads with helmet
    n_uncertain: int = 0           # heads to check by eye
    n_helmet_only: int = 0         # shown, not counted

    @property
    def n_heads(self) -> int:
        return self.n_without + self.n_with + self.n_uncertain

    def as_rows(self) -> List[dict]:
        return [{"class": d.label,
                 "confidence": round(d.score, 3),
                 "check by eye": "yes" if d.uncertain else "",
                 "x1": d.box[0], "y1": d.box[1], "x2": d.box[2], "y2": d.box[3]}
                for d in self.detections]


# ==========================================
# Model
# ==========================================
@st.cache_resource(show_spinner="Loading model...")
def load_model(path: str = str(MODEL_LOCAL_PATH)):
    """Load the YOLO model once per server process."""
    from ultralytics import YOLO

    path = Path(path)
    if not path.exists() and MODEL_URL:
        from sample_utils.download import download_file
        download_file(MODEL_URL, path)
    if not path.exists():
        st.error(
            f"Model file not found: `{path}`.\n\n"
            "Copy your trained weights there "
            "(`reports/runs/detect/<run>/weights/best.pt` → `models/helmet_yolov8s.pt`), "
            "or set `HELMET_MODEL_PATH` / `HELMET_MODEL_URL`."
        )
        st.stop()

    model = YOLO(str(path))
    names = set(model.names.values())
    if not EXPECTED_CLASSES.issubset(names):
        st.warning(f"Model classes {sorted(names)} differ from the expected "
                   f"{sorted(EXPECTED_CLASSES)}. Alerts may not work.")
    return model


# ==========================================
# Input checks
# ==========================================
def decode_image_bytes(data: bytes) -> np.ndarray:
    """Decode uploaded bytes to a BGR image, or raise InvalidInput."""
    if not data:
        raise InvalidInput("The file is empty.")
    arr = np.frombuffer(data, dtype=np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)   # always 3-channel BGR (drops alpha)
    if image is None:
        raise InvalidInput("The file could not be read as an image.")
    check_frame(image)
    return image


def check_frame(image: Optional[np.ndarray]) -> None:
    if image is None or image.size == 0:
        raise InvalidInput("The frame is empty.")
    h, w = image.shape[:2]
    if h < 32 or w < 32:
        raise InvalidInput(f"The image is too small ({w}×{h} px).")


# ==========================================
# Detection and status rules
# ==========================================
def _iou(a: List[int], b: List[int]) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def detect(model, image_bgr: np.ndarray,
           alert_conf: float = DEFAULT_ALERT_CONF,
           review_conf: float = DEFAULT_REVIEW_CONF,
           imgsz: int = 640) -> FrameResult:
    """Run the model on one BGR image and apply the status rules."""
    check_frame(image_bgr)
    review_conf = min(review_conf, alert_conf)

    # Ultralytics letterboxes internally; pass the original BGR image unchanged.
    result = model.predict(image_bgr, conf=review_conf, imgsz=imgsz, verbose=False)[0]
    names = result.names

    detections: List[Detection] = []
    if result.boxes is not None and len(result.boxes):
        for xyxy, cls, conf in zip(result.boxes.xyxy.cpu().numpy(),
                                   result.boxes.cls.cpu().numpy(),
                                   result.boxes.conf.cpu().numpy()):
            detections.append(Detection(label=names[int(cls)], score=float(conf),
                                        box=[int(v) for v in xyxy]))

    without = [d for d in detections if d.label == HEAD_WITHOUT]
    with_h = [d for d in detections if d.label == HEAD_WITH]

    # Rule 1: Head_without_helmet below the alert cutoff -> check by eye
    for d in without:
        if d.score < alert_conf:
            d.uncertain = True
    # Rule 2: same head labelled both with and without helmet -> check by eye
    for d in without:
        for w in with_h:
            if _iou(d.box, w.box) >= CONFLICT_IOU:
                d.uncertain = True
                w.uncertain = True

    fr = FrameResult(status=COMPLIANT, detections=detections)
    conflicted_with = sum(1 for w in with_h if w.uncertain)
    fr.n_without = sum(1 for d in without if not d.uncertain)
    fr.n_uncertain = sum(1 for d in without if d.uncertain)
    fr.n_with = len(with_h) - conflicted_with
    fr.n_helmet_only = sum(1 for d in detections if d.label == HELMET_ONLY)

    if fr.n_without > 0:
        fr.status = VIOLATION
    elif fr.n_uncertain > 0:
        fr.status = UNCERTAIN
    elif fr.n_with == 0:
        fr.status = NO_PEOPLE      # empty frame is not evidence of compliance
    return fr


# ==========================================
# Drawing
# ==========================================
def draw(image_bgr: np.ndarray, fr: FrameResult, banner: bool = True) -> np.ndarray:
    """Return a copy of the image with boxes, labels and a status banner (BGR)."""
    out = image_bgr.copy()
    h, w = out.shape[:2]
    scale = max(0.4, min(w, h) / 900)
    thick = max(1, int(round(scale * 2)))

    for d in fr.detections:
        color = UNCERTAIN_COLOR_BGR if d.uncertain else CLASS_COLORS_BGR.get(d.label, (255, 255, 255))
        x1, y1, x2, y2 = d.box
        cv2.rectangle(out, (x1, y1), (x2, y2), color, thick)
        text = f"{d.label}{' ?' if d.uncertain else ''} {d.score:.2f}"
        (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
        ty = y1 - 4 if y1 - th - 8 > 0 else y1 + th + 6
        cv2.rectangle(out, (x1, ty - th - 4), (x1 + tw + 6, ty + base - 2), color, -1)
        cv2.putText(out, text, (x1 + 3, ty - 2), cv2.FONT_HERSHEY_SIMPLEX,
                    scale, (255, 255, 255), 1, cv2.LINE_AA)

    if banner:
        banner_color = {VIOLATION: CLASS_COLORS_BGR[HEAD_WITHOUT],
                        UNCERTAIN: UNCERTAIN_COLOR_BGR,
                        COMPLIANT: CLASS_COLORS_BGR[HEAD_WITH],
                        NO_PEOPLE: (110, 110, 110)}[fr.status]
        text = (f"{STATUS_TEXT[fr.status]}  |  without: {fr.n_without}  "
                f"with: {fr.n_with}  check: {fr.n_uncertain}")
        bs = scale * 0.9
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, bs, 1)
        cv2.rectangle(out, (0, 0), (min(w, tw + 16), th + 14), banner_color, -1)
        cv2.putText(out, text, (8, th + 6), cv2.FONT_HERSHEY_SIMPLEX,
                    bs, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def to_rgb(image_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


# ==========================================
# Streamlit widgets shared by all pages
# ==========================================
def threshold_controls(key_prefix: str, disabled: bool = False):
    """Two sliders: alert cutoff and the lower review cutoff."""
    alert = st.slider("Alert cutoff (Head_without_helmet)", 0.05, 0.95,
                      DEFAULT_ALERT_CONF, 0.05, key=f"{key_prefix}_alert",
                      disabled=disabled,
                      help="A head without helmet at or above this confidence raises an alert.")
    review = st.slider("Review cutoff", 0.05, 0.95, DEFAULT_REVIEW_CONF, 0.05,
                       key=f"{key_prefix}_review", disabled=disabled,
                       help="Detections between this value and the alert cutoff "
                            "are marked 'check by eye' instead of raising an alert.")
    if review > alert:
        st.caption("Review cutoff is above the alert cutoff, so it is set equal to it.")
        review = alert
    st.caption("Both cutoffs are provisional. They will be tuned on the validation "
               "sample (see docs/project_decisions.md).")
    return alert, review


def show_status(fr: FrameResult) -> None:
    msg = STATUS_TEXT[fr.status]
    if fr.status == VIOLATION:
        st.error(f"**{msg}** — {fr.n_without} head(s) without helmet")
    elif fr.status == UNCERTAIN:
        st.warning(f"**{msg}** — {fr.n_uncertain} head(s) need review")
    elif fr.status == NO_PEOPLE:
        st.info(f"**{msg}.** This is not a compliance result: the model was "
                "trained only on images that contain people.")
    else:
        st.success(f"**{msg}** — {fr.n_with} head(s)")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Without helmet", fr.n_without)
    c2.metric("With helmet", fr.n_with)
    c3.metric("Check by eye", fr.n_uncertain)
    c4.metric("Helmet only*", fr.n_helmet_only)
    st.caption("*Helmet_only is shown but not used for alerts, because its test "
               "accuracy is still too low (see docs/project_decisions.md).")
