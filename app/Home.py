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
  compliant, because the model was trained only on images that contain people.
- **Invalid input**: files that cannot be read as an image or video are rejected with a reason.

#### Pages

Choose a page in the sidebar:

1. **Image Detection**: upload a photo. This is the closest match to how the model was evaluated.
2. **Video Detection**: upload a recorded `.mp4` and get an annotated video and a list of flagged moments.
3. **Realtime Detection**: use a webcam.

#### Limitations

The model was trained on a 1,000-image sample of a Roboflow helmet dataset
(`WearHelmetDetection.v1i.yolov8`) and evaluated on still photographs, not site CCTV footage.
On the test sample it still misses about one in five heads without a helmet.
Treat every result as a prompt for human review, not as a final decision.
    """
)

st.divider()
st.caption(
    "Project, Baikal School of BRICS, Irkutsk National Research Technical University. "
    "Built with Ultralytics YOLOv8s and Streamlit."
)
