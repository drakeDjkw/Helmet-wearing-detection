"""Lab 05: request-level input validation for Helmet Wearing Detection.

The validator decides what happens before prediction:

    pass    -> call the prediction function            (action "predict")
    warning -> do not predict automatically            (action "route_for_review")
    error   -> do not predict                          (action "reject_input")

When several checks fail, the most severe result wins: error > warning > pass.

Scope: these checks read the values supplied in the request (input_ref,
format, width, height, source) and compare the runtime preprocessing identity
with config/input_contract.json. They do not open the referenced file, prove
that it exists or decode it. "input_reference_present" means the field is
non-empty, not that the file was found. Decoding is checked later, inside
project_predict(), by app/helmet_utils.decode_image_bytes().

This module uses only the standard library, so the tests run without
OpenCV, Streamlit or Ultralytics.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONTRACT_PATH = ROOT / "config" / "input_contract.json"

SEVERITY = {"pass": 0, "warning": 1, "error": 2}


def load_contract(path: Path | str = DEFAULT_CONTRACT_PATH) -> dict[str, Any]:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


# ------------------------------------------------------------------
# Single checks
# ------------------------------------------------------------------
def _check(rule: str, observed: Any, status: str, contract: dict[str, Any]) -> dict[str, Any]:
    """One check result in the shape the trace uses."""
    return {
        "rule": rule,
        "observed": observed,
        "status": status,
        "action": contract["decision"]["check_action"][status],
    }


def _fail_status(contract: dict[str, Any], rule: str) -> str:
    return contract["rules"][rule]["severity_on_fail"]


def _is_int(value: Any) -> bool:
    # bool is a subclass of int; True must not count as a width of 1.
    return isinstance(value, int) and not isinstance(value, bool)


def check_input_reference(request: dict[str, Any], contract: dict[str, Any]) -> dict[str, Any]:
    rule = "input_reference_present"
    value = request.get("input_ref")
    ok = isinstance(value, str) and value.strip() != ""
    return _check(rule, value, "pass" if ok else _fail_status(contract, rule), contract)


def check_format(request: dict[str, Any], contract: dict[str, Any]) -> dict[str, Any]:
    rule = "format_supported"
    value = request.get("format")
    normalised = value.strip().lower().lstrip(".") if isinstance(value, str) else value
    ok = normalised in contract["rules"][rule]["allowed"]
    return _check(rule, normalised, "pass" if ok else _fail_status(contract, rule), contract)


def check_dimensions(request: dict[str, Any], contract: dict[str, Any]) -> dict[str, Any]:
    rule = "dimensions_minimum"
    spec = contract["rules"][rule]
    width, height = request.get("width"), request.get("height")
    ok = (_is_int(width) and _is_int(height)
          and width >= spec["min_width"] and height >= spec["min_height"])
    observed = {"width": width, "height": height}
    return _check(rule, observed, "pass" if ok else _fail_status(contract, rule), contract)


def check_source_registered(request: dict[str, Any], contract: dict[str, Any]) -> dict[str, Any]:
    rule = "source_registered"
    value = request.get("source")
    ok = isinstance(value, str) and value in contract["sources"]
    return _check(rule, value, "pass" if ok else _fail_status(contract, rule), contract)


def check_source_evaluated(request: dict[str, Any], contract: dict[str, Any]) -> dict[str, Any]:
    """Only meaningful for a registered source; the caller skips it otherwise."""
    rule = "source_evaluated"
    source = request["source"]
    evaluated = bool(contract["sources"][source]["evaluated"])
    observed = {"source": source, "evaluated": evaluated}
    return _check(rule, observed, "pass" if evaluated else _fail_status(contract, rule), contract)


def check_preprocessing(runtime_preprocessing_id: str, contract: dict[str, Any]) -> dict[str, Any]:
    """Compare the id defined beside the inference code, never a request field."""
    rule = "preprocessing_identity"
    expected = contract["preprocessing"]["expected_id"]
    observed = {"runtime": runtime_preprocessing_id, "expected": expected}
    ok = runtime_preprocessing_id == expected
    return _check(rule, observed, "pass" if ok else _fail_status(contract, rule), contract)


# ------------------------------------------------------------------
# Combined decision
# ------------------------------------------------------------------
def combine(checks: list[dict[str, Any]], contract: dict[str, Any]) -> tuple[str, str]:
    """Most severe status wins (error > warning > pass); returns (status, action)."""
    status = max((c["status"] for c in checks), key=SEVERITY.__getitem__, default="pass")
    return status, contract["decision"]["overall_action"][status]


def validate_request(request: dict[str, Any], contract: dict[str, Any],
                     runtime_preprocessing_id: str) -> dict[str, Any]:
    """Run every applicable rule and combine the results.

    A "preprocessing_id" field in the request is ignored on purpose: the
    transform identity belongs to the deployed code, not to the caller.
    """
    checks = [
        check_input_reference(request, contract),
        check_format(request, contract),
        check_dimensions(request, contract),
    ]
    skipped: list[dict[str, str]] = []

    registered = check_source_registered(request, contract)
    checks.append(registered)
    if registered["status"] == "pass":
        checks.append(check_source_evaluated(request, contract))
    else:
        skipped.append({"rule": "source_evaluated",
                        "reason": "source_registered did not pass"})

    checks.append(check_preprocessing(runtime_preprocessing_id, contract))

    status, action = combine(checks, contract)
    return {"checks": checks, "rules_skipped": skipped,
            "validation_status": status, "action": action}


# ------------------------------------------------------------------
# Route one request
# ------------------------------------------------------------------
def route_request(request: dict[str, Any],
                  predict: Callable[[dict[str, Any]], Any],
                  contract: dict[str, Any],
                  runtime_preprocessing_id: str) -> dict[str, Any]:
    """Validate, then call predict(request) only when validation passed.

    Returns one trace record. The same request_id ties the validation
    decision to the prediction outcome of this attempt.
    """
    started = time.perf_counter()
    request_id = request.get("request_id") or str(uuid4())
    decision = validate_request(request, contract, runtime_preprocessing_id)

    prediction_called = False
    prediction: Any = None
    prediction_error: str | None = None

    if decision["validation_status"] == "pass":
        prediction_called = True
        try:
            prediction = predict(request)
        except Exception as error:  # e.g. weights missing; keep the attempt traceable
            prediction_error = f"{type(error).__name__}: {error}"

    if not prediction_called:
        prediction_status = "not_called"
    elif prediction_error is not None:
        prediction_status = "failed"
    else:
        prediction_status = "returned"

    return {
        "request_id": request_id,
        "input_ref": request.get("input_ref"),
        "source": request.get("source"),
        "contract_version": contract["contract_version"],
        "preprocessing_id": runtime_preprocessing_id,
        "checks": decision["checks"],
        "rules_skipped": decision["rules_skipped"],
        "validation_status": decision["validation_status"],
        "action": decision["action"],
        "prediction_called": prediction_called,
        "prediction_status": prediction_status,
        "prediction": prediction,
        "prediction_error": prediction_error,
        "latency_ms": round((time.perf_counter() - started) * 1000, 1),
    }
