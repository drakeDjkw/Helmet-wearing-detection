# Project decisions

## Lab 03 — Acceptance

All measured values below come from `reports/baseline_metrics.json` and the test confusion matrix in `reports/runs/detect/YOLOv8s_test/`. The run used YOLOv8s, 25 epochs, 640 px input, an Apple M4 (MPS), and was scored on the 200-image test sample (882 objects). Proposed thresholds are marked as **proposed** and have not been met or tested.

### Decision after the baseline

**Decision:** Simplify

**Reason:** Based on the measured test mAP50 of 0.865 for `Head_without_helmet` and 0.957 for `Helmet_with_head`, against 0.157 for `Helmet_only` (recall 0.211, only 19 test boxes), and on the limitation that `Helmet_only` makes up about 2% of the labels, we decided to keep the project direction but simplify what the first version reports. The first version answers one question per detected head: is a helmet being worn or not? `Helmet_only` detections stay in the model output but do not trigger alerts and are not counted in compliance figures until that class has enough data to be evaluated.

The user (site safety officer), the input (images of construction scenes) and the next action (review the flagged frame, then speak to the worker) stay as described in `project_brief.md`.

### Required behavior

- **Normal input:** A readable image or video frame (JPG or PNG) of a work area. The system returns one box per detected head, with its class and confidence. A frame with at least one `Head_without_helmet` box at or above the alert cutoff is flagged for the safety officer, with the box drawn on the image and a count of compliant and non-compliant heads. The officer opens the frame, confirms the violation and speaks to the worker or crew supervisor.
- **Invalid input:** A file that cannot be decoded as an image, is empty, or is not an image format. The system returns no detections and rejects the input with a reason ("file could not be read as an image"), and logs it. A frame in which no heads are detected is reported as **"no people detected"**, never as "compliant". The model was trained only on images that contain people (no training image has an empty label file), so an empty result carries no information about compliance.
- **Valid input with an uncertain result:** A `Head_without_helmet` box with confidence below the alert cutoff but above a lower review cutoff, or a head that receives both a `Head_without_helmet` and a `Helmet_with_head` box. The system does not raise an alert and does not count the head as compliant. It places the frame in a review queue marked "uncertain: check by eye". The officer decides from the image. Both cutoff values are **not set yet**. They will be chosen on the validation sample, not on the test sample.

### One quality that matters

- **Response time.** An alert is only useful while the worker is still in the area. The figure measured so far is 2.67 ms of model inference per image on the M4 (MPS), from the batched test evaluation. That figure leaves out image loading, pre-processing (except Ultralytics' own) and drawing, so it does not represent the time a user would wait. **Target: not set yet.** What to measure first: end-to-end `predict()` time for a single frame, from file to returned boxes, on the device that would run the system. We also need to know how often the site cameras deliver frames, and how quickly the safety officer needs an alert.

### One acceptance check

- **Important failure:** A worker without a helmet is not flagged, either because the head is missed entirely or because it is labelled `Helmet_with_head`. In both cases the officer is never shown the violation.
- **Metric that would show this failure:** False-negative rate for `Head_without_helmet`: the share of true `Head_without_helmet` boxes that the system does not report as `Head_without_helmet` (missed as background, or given another class) at the chosen alert cutoff.
  - *Measured baseline (test sample, confusion matrix):* 184 true `Head_without_helmet` boxes. Of these, 145 were detected correctly, 32 were missed as background and 7 were labelled `Helmet_with_head`. False-negative rate = 39 / 184 = **21.2%**. The reported recall for the class is 0.783.
- **Preliminary acceptance threshold:** **Proposed:** false-negative rate for `Head_without_helmet` below **10%** in a later evaluation, while the false-alarm count stays low enough for one officer to review. The false-alarm limit is **not set yet**.
- **Where will we check it?** On the held-out test sample, scored once after the alert cutoff has been fixed on the validation sample. If the leakage check (below) finds shared source images, we will check it on a regrouped test split instead. Later, we will also check it on labelled frames from a real site camera, if we can obtain them.
- **Why this number for now?** The baseline misses about one violation in five. Because every alert is reviewed by a person, a false alarm costs the officer a few seconds, while a missed violation leaves a worker unprotected. We therefore accept more false alarms in exchange for fewer misses. Below 10% means the officer would see at least nine of every ten violations. The value is a starting point for discussion, not a safety standard.
- **Evidence still missing:**
  - A leakage check: whether copies of the same source photo (`hard_hat_workers<N>`) appear in both training and test.
  - The uncertainty of the estimate: 184 boxes is a small sample, and the full test split (394 images) has not been used.
  - The false-negative rate on images that resemble real site cameras (high angles, night, low resolution). The current images appear to be photographs, not CCTV frames.
  - A false-alarm rate on frames without people, which the current test set does not contain.
  - A breakdown of the 39 misses by head size and occlusion, to find out which cases cause them.

### Failure cases

| Area | What could go wrong and why it matters | How would we notice it? | Planned response |
| --- | --- | --- | --- |
| Data or input | Site camera frames differ from the training photos (high CCTV angle, night, rain, low resolution, empty scenes). Misses and false alarms rise without warning. | False-negative and false-alarm rates on a small labelled set from the target camera. A jump in alerts on frames that the officer marks "no one there". | Before use, score a labelled sample from each camera. Add frames from that camera, including empty ones, to training. Mark frames from unchecked cameras "not evaluated". |
| Model behavior | A worker without a helmet is missed or labelled `Helmet_with_head`: a small, distant or partly hidden head, or one with a hood or cap (**the acceptance-check failure**). | `Head_without_helmet` false-negative rate on the test sample (now 21.2%). The confusion-matrix row for that class. Officer spot checks of unflagged frames. | Lower the alert cutoff for this class, chosen on validation. Send borderline boxes to the review queue. Retrain on the full training set with more examples of the missed cases. |
| Integration or workflow | The same worker triggers an alert on every frame, or false alarms pile up, so the officer starts ignoring alerts. A frame with no detections is shown as "compliant". | A growing review queue. A high share of alerts dismissed by the officer. Frames with zero detections counted in compliance figures. | Group alerts by camera and time window (one alert per person-event, not per frame). Show "no people detected" separately from "compliant". Track the dismissal rate as a workflow metric. |