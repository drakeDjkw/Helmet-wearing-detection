import sys
import tempfile
from pathlib import Path

import cv2
import pandas as pd
import streamlit as st

sys.path.append(str(Path(__file__).resolve().parents[1]))
import alerting  # noqa: E402
import helmet_utils as hu  # noqa: E402

st.set_page_config(page_title="Video Detection", page_icon="🎞️", layout="centered")

TEMP_DIR = Path(tempfile.gettempdir()) / "helmet_app"
TEMP_DIR.mkdir(exist_ok=True)

st.title("Helmet Detection: Video")
st.write("Upload a recorded video. The app marks every processed frame, raises an alert "
         "when a worker without a helmet is seen in several processed frames in a row, and "
         "lists the alerts and the moments to review, so you can jump straight to them.")

model = hu.load_model()

running = st.session_state.get("video_running", False)
video_file = st.file_uploader("Upload video", type=["mp4", "mov", "avi", "mkv"],
                              disabled=running)
alert_conf, review_conf = hu.threshold_controls("vid", disabled=running)
stride = st.slider("Process every N-th frame", 1, 30, 5, disabled=running,
                   help="Higher values are faster. Skipped frames are written "
                        "with the boxes of the last processed frame.")
window, min_hits, cooldown_s, _ = hu.alert_controls("vid", disabled=running, sound=False)


def video_time(s: float) -> str:
    return f"{int(s // 60)}:{s % 60:04.1f}"


def group_events(rows, gap_s=1.0):
    """Merge flagged frames that are less than gap_s apart into one event."""
    events = []
    for r in rows:
        if events and r["time_s"] - events[-1]["end_s"] <= gap_s and r["status"] == events[-1]["status"]:
            ev = events[-1]
            ev["end_s"] = r["time_s"]
            ev["max_without"] = max(ev["max_without"], r["without"])
            ev["max_check"] = max(ev["max_check"], r["check"])
        else:
            events.append({"status": r["status"], "start_s": r["time_s"], "end_s": r["time_s"],
                           "max_without": r["without"], "max_check": r["check"]})
    for ev in events:
        ev["start"] = video_time(ev["start_s"])
        ev["end"] = video_time(ev["end_s"])
    return events


def process_video(uploaded, alert_conf, review_conf, stride, alert_cfg):
    in_path = TEMP_DIR / f"input{Path(uploaded.name).suffix.lower()}"
    out_path = TEMP_DIR / "helmet_prediction.mp4"
    in_path.write_bytes(uploaded.getbuffer())

    cap = cv2.VideoCapture(str(in_path))
    if not cap.isOpened():
        st.error("**Input rejected:** the file could not be opened as a video.")
        return None

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0          # some files report 0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

    if width < 32 or height < 32:
        st.error(f"**Input rejected:** frame size {width}×{height} is too small.")
        cap.release()
        return None

    duration = total / fps if total else 0
    st.write(f"Duration {int(duration // 60)}:{int(duration % 60):02d} · "
             f"{width}×{height} px · {fps:.1f} FPS · {total} frames")

    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"),
                             fps, (width, height))
    progress = st.progress(0.0, text="Running detection...")
    preview = st.empty()

    flagged, counts = [], {s: 0 for s in hu.STATUS_TEXT}
    last_fr, idx, unreadable = None, 0, 0
    tracker = alerting.AlertTracker(*alert_cfg)      # alerts in video time (seconds)

    while True:
        ok, frame = cap.read()                        # BGR, as the model expects
        if not ok:
            break
        ts = idx / fps
        change = None
        if idx % stride == 0 or last_fr is None:
            try:
                last_fr = hu.detect(model, frame, alert_conf, review_conf)
            except hu.InvalidInput:
                unreadable += 1
                idx += 1
                continue
            counts[last_fr.status] += 1
            if last_fr.status in (hu.VIOLATION, hu.UNCERTAIN):
                flagged.append({"time_s": ts, "status": last_fr.status,
                                "without": last_fr.n_without, "check": last_fr.n_uncertain})
            confs = [d.score for d in last_fr.detections
                     if d.label == hu.HEAD_WITHOUT and not d.uncertain]
            change = tracker.update(ts, last_fr.status == hu.VIOLATION,
                                    n_without=last_fr.n_without, max_conf=max(confs, default=0.0))
        annotated = hu.draw(frame, last_fr)
        event = tracker.current
        if event is not None:
            hu.draw_alert(annotated, f"ALERT #{event.id}: worker without helmet  "
                                     f"({video_time(event.raised_ts)})")
            if change == alerting.RAISED:
                event.snapshot = annotated.copy()
                tracker.trim_snapshots()
        writer.write(annotated)

        if idx % (stride * 5) == 0:
            preview.image(hu.to_rgb(annotated), width="stretch")
        idx += 1
        if total:
            progress.progress(min(idx / total, 1.0), text=f"Running detection... frame {idx}/{total}")

    cap.release()
    writer.release()
    progress.empty()

    if idx == 0:
        st.error("**Input rejected:** no frames could be read from the video.")
        return None
    end_ts = idx / fps
    tracker.close(end_ts)                             # an alert still active at the end
    return {"out_path": out_path, "counts": counts, "events": group_events(flagged),
            "alerts": tracker.rows(end_ts, time_fmt=video_time),
            "snapshots": [(e.id, e.raised_ts, e.snapshot) for e in tracker.events
                          if e.snapshot is not None],
            "processed": sum(counts.values()), "unreadable": unreadable}


if video_file is not None:
    if st.button("Process video", type="primary", width="stretch", disabled=running):
        st.session_state.video_running = True
        try:
            st.session_state.video_result = process_video(
                video_file, alert_conf, review_conf, stride, (window, min_hits, cooldown_s))
        finally:
            st.session_state.video_running = False

res = st.session_state.get("video_result")
if res:
    st.success(f"Video processed: {res['processed']} frames checked.")
    c = res["counts"]
    alerts = res.get("alerts", [])
    c0, c1, c2, c3, c4 = st.columns(5)
    c0.metric("Alerts", len(alerts))
    c1.metric("Violation frames", c[hu.VIOLATION])
    c2.metric("Uncertain frames", c[hu.UNCERTAIN])
    c3.metric("Compliant frames", c[hu.COMPLIANT])
    c4.metric("No people", c[hu.NO_PEOPLE])
    if res["unreadable"]:
        st.warning(f"{res['unreadable']} frame(s) could not be read and were skipped.")

    if alerts:
        st.error(f"🚨 **{len(alerts)} alert(s): worker without a helmet.** "
                 "Check each moment below, then speak to the worker or crew supervisor.")
        st.write("#### Alerts")
        adf = pd.DataFrame(alerts)
        st.dataframe(adf.rename(columns={
            "event": "Alert", "raised": "From", "ended": "To", "duration_s": "Duration (s)",
            "alert_latency_s": "Alert latency (s)", "max_heads_without": "Max heads without helmet",
            "max_confidence": "Max confidence", "violating_frames": "Violating frames"}),
            width="stretch", hide_index=True)
        st.caption("Times are positions in the video. Alert latency: time from the first "
                   "violating frame to the alert. Durations and frame counts refer to "
                   "processed frames (every N-th frame).")
        st.download_button("Download alert log (CSV)", adf.to_csv(index=False).encode(),
                           file_name="helmet_video_alerts.csv", mime="text/csv")
        if res.get("snapshots"):
            cols = st.columns(3)
            for i, (eid, ts, snap) in enumerate(res["snapshots"][:9]):
                cols[i % 3].image(hu.to_rgb(snap), caption=f"Alert #{eid} at {video_time(ts)}",
                                  width="stretch")

    if res["events"]:
        st.write("#### Moments to review")
        df = pd.DataFrame(res["events"])[["status", "start", "end", "max_without", "max_check"]]
        df.columns = ["Status", "From", "To", "Max heads without helmet", "Max heads to check"]
        st.dataframe(df, width="stretch", hide_index=True)
    else:
        st.info("No violations or uncertain results were found in the processed frames.")

    with open(res["out_path"], "rb") as f:
        st.download_button("Download annotated video", data=f,
                           file_name="helmet_prediction.mp4", mime="video/mp4",
                           width="stretch")
    st.caption("The video uses the mp4v codec, which most browsers cannot play inline. "
               "Download it and open it in a video player such as VLC or QuickTime.")
