import streamlit as st

st.set_page_config(page_title="Helmet Wearing Detection", page_icon="⛑️")

st.title("Helmet Wearing Detection")

st.markdown(
    """
This app flags construction workers who are not wearing a safety helmet.
A YOLOv8s model finds each head in an image and gives it one of three classes:

| Class | Meaning | Used for alerts |
|---|---|---|
| `Head_without_helmet` | A head with no helmet | **Yes**: raises an alert |
| `Helmet_with_head` | A head wearing a helmet | Counted as compliant |
| `Helmet_only` | A helmet not on a head | No: shown only |

#### How results are reported

- **Violation**: at least one head without a helmet above the alert cutoff.
  Open the image, confirm the violation, then speak to the worker or crew supervisor.
- **Uncertain: check by eye**: a head without a helmet with lower confidence, or a head
  that received both labels. No alert is raised. Decide from the image.
- **No people detected**: the model found no heads. This is **not** reported as
  compliant, because workers may be present whose heads the model did not find.
- **Invalid input**: files that cannot be read as an image or video are rejected with a reason.

#### Alerts

- **Image**: a photo with a violation raises an alert (red border and message).
- **Video and webcam**: one frame is not enough. An alert is raised when at least 3 of the
  last 5 processed frames show a violation, stays active while it continues (one alert per
  event, not per frame) and clears when none of the last 5 frames does. After an alert ends,
  no new one is raised for 10 s. These values can be changed under *Alert settings*.
- Each alert records when it started and ended, its **alert latency** (first violating frame
  to alert), the most heads without a helmet and the highest confidence. The log can be
  downloaded as CSV.

#### Pages

Choose a page in the sidebar:

1. **Image Detection**: upload a photo. This is the closest match to how the model was evaluated.
2. **Video Detection**: upload a recorded video and get an annotated video, the alerts with
   snapshots, and a list of moments to review.
3. **Realtime Detection**: use a webcam, with live alerts (red border, optional sound) and an alert log.

#### Limitations

The model is YOLOv8s, trained on a 3,000-image sample of the Roboflow dataset
`WearHelmetDetection.v1i.yolov8` (hard-hat photographs and web images) and evaluated on a
300-image test sample of still photographs, not site CCTV footage. On that sample it misses
about one in four heads without a helmet (57 of 221, or 25.8%), against a target of under 10%.
It can also mistake a hood or cap for a helmet and report such a worker as compliant.
Treat every result as a prompt for human review, not as a final decision.
    """
)

st.divider()
st.caption(
    "Project, Baikal School of BRICS, Irkutsk National Research Technical University. "
    "Built with Ultralytics YOLOv8s and Streamlit."
)