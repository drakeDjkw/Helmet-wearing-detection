import sys
from io import BytesIO
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image

sys.path.append(str(Path(__file__).resolve().parents[1]))
import alerting  # noqa: E402
import helmet_utils as hu  # noqa: E402

st.set_page_config(page_title="Image Detection", page_icon="📷", layout="wide")

st.title("Helmet Detection: Image")
st.write("Upload a photo of a work area. Each detected head is marked as wearing "
         "or not wearing a helmet. A photo with a confident head without a helmet raises an alert.")

model = hu.load_model()

image_file = st.file_uploader("Upload image", type=["jpg", "jpeg", "png", "bmp", "webp"])
alert_conf, review_conf = hu.threshold_controls("img")
sound_on = st.toggle("Play a sound when an alert is raised", value=False, key="img_sound",
                     help="Some browsers block sound until you have clicked on the page.")

if image_file is not None:
    # Invalid input: reject with a reason, no detections
    try:
        image_bgr = hu.decode_image_bytes(image_file.getvalue())
    except hu.InvalidInput as e:
        st.error(f"**Input rejected:** {e} Upload a JPG or PNG photo.")
        st.stop()

    fr = hu.detect(model, image_bgr, alert_conf, review_conf)
    annotated_bgr = hu.draw(image_bgr, fr)
    if fr.status == hu.VIOLATION:
        hu.draw_alert(annotated_bgr, f"ALERT: {fr.n_without} worker(s) without helmet")
    annotated = hu.to_rgb(annotated_bgr)

    if fr.status == hu.VIOLATION:
        confs = [d.score for d in fr.detections if d.label == hu.HEAD_WITHOUT and not d.uncertain]
        st.error(f"🚨 **ALERT: {fr.n_without} worker(s) without a helmet** "
                 f"(confidence up to {max(confs):.2f}). "
                 "Confirm on the image, then speak to the worker or crew supervisor.")
        if sound_on:
            st.audio(alerting.beep_wav(), format="audio/wav", autoplay=True)
    hu.show_status(fr)

    col1, col2 = st.columns(2)
    with col1:
        st.write("#### Image")
        st.image(hu.to_rgb(image_bgr), width="stretch")
    with col2:
        st.write("#### Detections")
        st.image(annotated, width="stretch")

        buffer = BytesIO()
        Image.fromarray(annotated).save(buffer, format="PNG")
        st.download_button("Download annotated image", data=buffer.getvalue(),
                           file_name=f"helmet_{Path(image_file.name).stem}.png",
                           mime="image/png")

    if fr.detections:
        with st.expander(f"Detection table ({len(fr.detections)} boxes)"):
            st.dataframe(pd.DataFrame(fr.as_rows()), width="stretch")
