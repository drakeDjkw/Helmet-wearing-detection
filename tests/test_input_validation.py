"""Lab 05 tests for src/input_validation.py.

Run from the repository root:
    python3 -m unittest tests/test_input_validation.py     (macOS / Linux)
    py -m unittest tests/test_input_validation.py          (Windows)

The prediction function is a recording fake: these tests check whether
prediction is called, not what the model predicts. They need only the
standard library.
"""

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from input_validation import load_contract, route_request, validate_request  # noqa: E402

PREPROCESSING_ID = "letterbox640-v1"   # same value as src/inference_demo.py


def make_request(**changes):
    request = {
        "request_id": "test-request",
        "input_ref": "data/WearHelmetDetection.v1i.yolov8/test/images/example.jpg",
        "format": "jpg",
        "width": 416,
        "height": 416,
        "source": "dataset-test-sample",
    }
    request.update(changes)
    return request


class RecordingPredict:
    def __init__(self):
        self.calls = []

    def __call__(self, request):
        self.calls.append(request)
        return {"result": {"frame_status": "COMPLIANT"}, "needs_review": False}


class InputValidationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.contract = load_contract()

    def route(self, request, preprocessing_id=PREPROCESSING_ID):
        predict = RecordingPredict()
        record = route_request(request, predict, self.contract, preprocessing_id)
        return record, predict

    # --- the three behaviours the lab asks for -------------------------

    def test_error_overrides_warning(self):
        # Unsupported format (error) + unevaluated source (warning).
        record, predict = self.route(make_request(format="gif", source="app-image-upload"))
        statuses = {c["rule"]: c["status"] for c in record["checks"]}

        self.assertEqual(statuses["format_supported"], "error")
        self.assertEqual(statuses["source_evaluated"], "warning")
        self.assertEqual(record["validation_status"], "error")
        self.assertEqual(record["action"], "reject_input")
        self.assertFalse(record["prediction_called"])
        self.assertEqual(predict.calls, [])

    def test_warning_does_not_call_prediction(self):
        record, predict = self.route(make_request(source="app-image-upload",
                                                  width=1280, height=720))

        self.assertEqual(record["validation_status"], "warning")
        self.assertEqual(record["action"], "route_for_review")
        self.assertFalse(record["prediction_called"])
        self.assertEqual(record["prediction_status"], "not_called")
        self.assertEqual(predict.calls, [])

    def test_pass_calls_prediction_once(self):
        request = make_request()
        record, predict = self.route(request)

        self.assertEqual(record["validation_status"], "pass")
        self.assertEqual(record["action"], "predict")
        self.assertTrue(record["prediction_called"])
        self.assertEqual(predict.calls, [request])
        self.assertEqual(record["request_id"], "test-request")

    # --- rule-level checks ---------------------------------------------

    def test_every_check_has_rule_observed_status_action(self):
        result = validate_request(make_request(), self.contract, PREPROCESSING_ID)
        for check in result["checks"]:
            self.assertEqual(set(check), {"rule", "observed", "status", "action"})

    def test_small_or_non_integer_dimensions_are_errors(self):
        for width, height in [(31, 416), (416, 0), ("416", 416), (True, 416)]:
            with self.subTest(width=width, height=height):
                result = validate_request(make_request(width=width, height=height),
                                          self.contract, PREPROCESSING_ID)
                self.assertEqual(result["validation_status"], "error")

    def test_unknown_source_is_error_and_skips_evaluation_rule(self):
        result = validate_request(make_request(source="camera-x"), self.contract,
                                  PREPROCESSING_ID)
        self.assertEqual(result["validation_status"], "error")
        self.assertEqual([s["rule"] for s in result["rules_skipped"]], ["source_evaluated"])

    def test_preprocessing_mismatch_is_error_and_request_cannot_override_it(self):
        record, predict = self.route(make_request(preprocessing_id="letterbox640-v1"),
                                     preprocessing_id="letterbox416-v1")
        self.assertEqual(record["validation_status"], "error")
        self.assertEqual(predict.calls, [])


if __name__ == "__main__":
    unittest.main()
