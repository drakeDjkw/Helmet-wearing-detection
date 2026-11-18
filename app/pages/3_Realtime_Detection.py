import sys
import threading
import time
from pathlib import Path

import av
import cv2
import pandas as pd
import streamlit as st
from streamlit_webrtc import WebRtcMode, webrtc_streamer

sys.path.append(str(Path(__file__).resolve().parents[1]))
import alerting  # noqa: E402
import helmet_utils as hu  # noqa: E402

st.set_page_config(page_title="Realtime Detection", page_icon="🎥", layout="centered")

# A fixed public STUN server is enough for local use. For deployment behind strict
# networks, add a TURN server here.
ICE_SERVERS = [{"urls": ["stun:stun.l.google.com:19302"]}]
POLL_S = 0.5            # how often the page refreshes the status while the camera runs

st.title("Helmet Detection: Realtime")
st.write("Detect helmet use from a webcam. Select the camera and press **Start**. "
         "An alert is raised only when a worker without a helmet is seen in several "
         "recent frames, so a single wrong frame does not trigger it.")

model = hu.load_model()
alert_conf, review_conf = hu.threshold_controls("rt")

with st.expander("Alert settings", expanded=False):
    window = st.slider("Frames to look at", 1, 15, 5,
                       help="The alert decision uses this many of the most recent processed frames.")
    if window > 1:
        min_hits = st.slider("Violating frames needed to raise an alert", 1, window, min(3, window),
                             help="Higher values give fewer false alarms but a later alert.")
    else:
        min_hits = 1
        st.caption("With one frame, every violating frame raises an alert.")
    cooldown_s = st.slider("Pause after an alert ends (s)", 0, 60, 10,
                           help="No new alert is raised for this long after the previous one cleared.")
    sound_on = st.toggle("Play a sound when an alert is raised", value=True,
                         help="Some browsers block sound until you have clicked on the page.")
    st.caption("An alert clears when none of the recent frames shows a violation. "
               "Uncertain detections never raise an alert; they are marked 'check by eye'.")


class SharedState:
    """Passes settings into, and results out of, the video thread safely."""

    def __init__(self):
        self.lock = threading.Lock()
        self.alert_conf = hu.DEFAULT_ALERT_CONF
        self.review_conf = hu.DEFAULT_REVIEW_CONF
        self.tracker = alerting.AlertTracker()
        self.last_result = None
        self.last_time = 0.0
        self.frames = 0
        self.proc_ms = 0.0          # moving average of detection + drawing time
        self.last_sound_id = 0      # last alert the page has already played a sound for


if "rt_state" not in st.session_state:
    st.session_state.rt_state = SharedState()
state: SharedState = st.session_state.rt_state

with state.lock:
    state.alert_conf, state.review_conf = alert_conf, review_conf
    state.tracker.configure(window, min_hits, cooldown_s)


def draw_alert(image, event: alerting.AlertEvent, now: float):
    """Red frame border and a bottom banner while an alert is active (BGR, in place)."""
    h, w = image.shape[:2]
    thick = max(6, min(w, h) // 60)
    cv2.rectangle(image, (0, 0), (w - 1, h - 1), (0, 0, 230), thick)
    text = f"ALERT #{event.id}: worker without helmet  ({event.duration_s(now):.0f} s)"
    scale = max(0.5, min(w, h) / 900)
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)
    cv2.rectangle(image, (0, h - th - 20), (min(w, tw + 20), h), (0, 0, 230), -1)
    cv2.putText(image, text, (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, scale,
                (255, 255, 255), 2, cv2.LINE_AA)
    return image


def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
    t0 = time.time()
    image = frame.to_ndarray(format="bgr24")         # BGR, as the model expects
    with state.lock:
        a, r = state.alert_conf, state.review_conf
    try:
        fr = hu.detect(model, image, a, r)
    except hu.InvalidInput:
        return frame                                  # pass bad frames through unchanged

    out = hu.draw(image, fr)
    confs = [d.score for d in fr.detections if d.label == hu.HEAD_WITHOUT and not d.uncertain]
    now = time.time()
    with state.lock:
        change = state.tracker.update(now, fr.status == hu.VIOLATION, n_without=fr.n_without,
                                      max_conf=max(confs, default=0.0))
        event = state.tracker.current
        if event is not None:
            draw_alert(out, event, now)
            if change == alerting.RAISED:
                event.snapshot = out.copy()           # the frame that raised the alert
                state.tracker.trim_snapshots()
        state.last_result, state.last_time = fr, now
        state.frames += 1
        ms = (time.time() - t0) * 1000
        state.proc_ms = ms if state.frames == 1 else 0.9 * state.proc_ms + 0.1 * ms
    return av.VideoFrame.from_ndarray(out, format="bgr24")


ctx = webrtc_streamer(
    key="helmet-detection",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration={"iceServers": ICE_SERVERS},
    video_frame_callback=video_frame_callback,
    media_stream_constraints={"video": {"width": {"ideal": 1280}}, "audio": False},
    async_processing=True,
)

st.divider()


def events_table(now: float) -> pd.DataFrame:
    with state.lock:
        rows = state.tracker.rows(now)
    return pd.DataFrame(rows)


if ctx.state.playing:
    show_details = st.checkbox("Show detections of the current frame", value=False)
    alert_box, metrics_box, sound_box = st.empty(), st.empty(), st.empty()
    snapshot_box, log_box, table_box = st.empty(), st.empty(), st.empty()
    shown_snapshot = 0

    while ctx.state.playing:
        now = time.time()
        with state.lock:
            fr, t = state.last_result, state.last_time
            event = state.tracker.current
            n_events = len(state.tracker.events)
            last = state.tracker.events[-1] if state.tracker.events else None
            frames, proc_ms = state.frames, state.proc_ms
            new_sound = last is not None and last.id > state.last_sound_id
            if new_sound:
                state.last_sound_id = last.id

        # 1. Alert or per-frame status
        with alert_box.container():
            if event is not None:
                st.error(f"🚨 **ALERT #{event.id}: worker without a helmet** for "
                         f"{event.duration_s(now):.0f} s, up to {event.max_without} head(s) "
                         f"(confidence up to {event.max_conf:.2f}). "
                         "Check the area and speak to the worker or crew supervisor.")
            elif fr is not None:
                hu.show_status(fr)
            age = now - t
            if fr is not None and age > 2:
                st.caption(f"Last result is {age:.0f} s old: the video may have stalled.")

        # 2. Session figures
        with metrics_box.container():
            c1, c2, c3 = st.columns(3)
            c1.metric("Alerts this session", n_events)
            c2.metric("Frames processed", frames)
            c3.metric("Processing per frame", f"{proc_ms:.0f} ms" if frames else "–")

        # 3. Sound, once per new alert
        if new_sound and sound_on:
            sound_box.audio(alerting.beep_wav(last.id), format="audio/wav", autoplay=True)
        elif not new_sound and event is None:
            sound_box.empty()

        # 4. Snapshot of the latest alert (only redrawn when a new alert arrives)
        if last is not None and last.id != shown_snapshot and last.snapshot is not None:
            snapshot_box.image(hu.to_rgb(last.snapshot),
                               caption=f"Alert #{last.id} at the moment it was raised",
                               width="stretch")
            shown_snapshot = last.id

        # 5. Alert log
        if n_events:
            log_box.dataframe(events_table(now).tail(10), width="stretch", hide_index=True)

        if show_details and fr is not None and fr.detections:
            table_box.dataframe(pd.DataFrame(fr.as_rows()), width="stretch")
        else:
            table_box.empty()
        time.sleep(POLL_S)

else:
    st.caption("Start the camera to see live results and alerts.")
    df = events_table(time.time())
    if not df.empty:
        st.subheader("Alerts from this session")
        st.dataframe(df, width="stretch", hide_index=True)
        st.caption("alert_latency_s: time from the first frame with a violation to the alert.")
        c1, c2 = st.columns(2)
        c1.download_button("Download alert log (CSV)", df.to_csv(index=False).encode(),
                           file_name=f"helmet_alerts_{time.strftime('%Y%m%d_%H%M%S')}.csv",
                           mime="text/csv")
        if c2.button("Clear alert log"):
            with state.lock:
                state.tracker.reset()
                state.last_sound_id = 0
            st.rerun()
        with state.lock:
            snaps = [(e.id, e.snapshot) for e in state.tracker.events if e.snapshot is not None]
        if snaps:
            st.write("Latest alert snapshots:")
            cols = st.columns(3)
            for i, (eid, snap) in enumerate(reversed(snaps[-6:])):
                cols[i % 3].image(hu.to_rgb(snap), caption=f"Alert #{eid}", width="stretch")
