import sys
import threading
import time
from pathlib import Path

import av
import pandas as pd
import streamlit as st
from streamlit_webrtc import WebRtcMode, webrtc_streamer

sys.path.append(str(Path(__file__).resolve().parents[1]))
import helmet_utils as hu  # noqa: E402

st.set_page_config(page_title="Realtime Detection", page_icon="🎥", layout="centered")

# A fixed public STUN server is enough for local use. For deployment behind strict
# networks, add a TURN server here.
ICE_SERVERS = [{"urls": ["stun:stun.l.google.com:19302"]}]

st.title("Helmet Detection: Realtime")
st.write("Detect helmet use from a webcam. Select the camera and press **Start**. "
         "The status bar on the video shows the result for the current frame.")

model = hu.load_model()
alert_conf, review_conf = hu.threshold_controls("rt")


class SharedState:
    """Passes settings into, and results out of, the video thread safely."""

    def __init__(self):
        self.lock = threading.Lock()
        self.alert_conf = hu.DEFAULT_ALERT_CONF
        self.review_conf = hu.DEFAULT_REVIEW_CONF
        self.last_result = None
        self.last_time = 0.0


if "rt_state" not in st.session_state:
    st.session_state.rt_state = SharedState()
state: SharedState = st.session_state.rt_state

with state.lock:
    state.alert_conf, state.review_conf = alert_conf, review_conf


def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
    image = frame.to_ndarray(format="bgr24")         # BGR, as the model expects
    with state.lock:
        a, r = state.alert_conf, state.review_conf
    try:
        fr = hu.detect(model, image, a, r)
    except hu.InvalidInput:
        return frame                                  # pass bad frames through unchanged
    with state.lock:
        state.last_result, state.last_time = fr, time.time()
    return av.VideoFrame.from_ndarray(hu.draw(image, fr), format="bgr24")


ctx = webrtc_streamer(
    key="helmet-detection",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration={"iceServers": ICE_SERVERS},
    video_frame_callback=video_frame_callback,
    media_stream_constraints={"video": {"width": {"ideal": 1280}}, "audio": False},
    async_processing=True,
)

st.divider()

if st.checkbox("Show live status and detections", value=False):
    status_box, table_box = st.empty(), st.empty()
    while ctx.state.playing:
        with state.lock:
            fr, t = state.last_result, state.last_time
        if fr is not None:
            age = time.time() - t
            with status_box.container():
                hu.show_status(fr)
                if age > 2:
                    st.caption(f"Last result is {age:.0f} s old: the video may have stalled.")
            if fr.detections:
                table_box.dataframe(pd.DataFrame(fr.as_rows()), width="stretch")
            else:
                table_box.empty()
        time.sleep(0.5)
    if not ctx.state.playing:
        st.caption("Start the camera to see live results.")
