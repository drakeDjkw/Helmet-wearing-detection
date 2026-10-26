# Split strategy

**Branch:** Our team trains the model. We fine-tune a pretrained YOLOv8s.

**Training and evaluation data:** We keep the three splits of the provided Roboflow export `WearHelmetDetection.v1i.yolov8` (2,564 / 986 / 394 images) and draw a fixed-seed sample (seed 42) from each one.

* **Training sample:** 1,000 images from `train`. The model is fitted on these.
* **Validation sample:** 200 images from `valid`. It is used only during training, to choose the best epoch. Early stopping (patience 20) was set but could not trigger in a 10-epoch run.
* **Test sample:** 200 images from `test`. It is held out. It is not used to fit the model, choose the model, or adjust any setting, and it is scored once after training. Its mAP50 (0.774) is the baseline result.

**Split method:** Random, at the image level, as supplied with the Roboflow export. The split is not grouped by source photo, site or camera. In the sample, the class proportions stay within about 2 percentage points of the full splits.

| Split | Full split | Sample | Objects in sample | Head_without_helmet | Helmet_only | Helmet_with_head |
|---|---|---|---|---|---|---|
| Train | 2,564 | 1,000 | 4,747 | 23.5% (full 24.9%) | 2.7% (full 2.6%) | 73.8% (full 72.5%) |
| Validation | 986 | 200 | 972 | 25.5% (full 27.5%) | 2.3% (full 2.0%) | 72.2% (full 70.6%) |
| Test | 394 | 200 | 918 | 18.6% (full 19.4%) | 1.7% (full 2.0%) | 79.6% (full 78.5%) |

**Does the evaluation resemble future input?** Partly. The test images show the same kinds of scenes as the training images: construction workers photographed at various distances, often several per image. However, they are not footage from fixed site cameras. Many look like photographs collected from the web (some carry website watermarks, and some are posed portraits). All of them were resized to about 416 × 416 px. Every test image also contains at least one labelled object, whereas a real site camera often shows no people. This is a limitation for deployment on a real site.

**Limitation of this first evaluation:** The file names keep the original image name (`hard_hat_workers<N>_png.rf.<hash>`). Roboflow exports can contain several augmented copies of one source image, and a random image-level split could place copies of the same photo in both training and test, which would make the score optimistic. This has not yet been checked. In addition, `Helmet_only` makes up only about 1.7% of the test objects (16 boxes), so its score has very wide uncertainty. Only 200 of the 394 available test images were used.