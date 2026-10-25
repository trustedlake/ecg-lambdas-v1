"""Contract of the ECG reading and interpretation lambdas.

Module shared by the mock lambdas (and reusable by the real ones):
error/warning catalog, supported layouts and the response envelope.
The readable specification lives in LAMBDA_CONTRACT.md, at the repo root.
"""

import time
import uuid

CONTRACT_VERSION = "2.0"
INTERPRET_LEGACY_VERSION = "1.0"

STATUS_OK = "OK"
STATUS_OK_WITH_WARNINGS = "OK_WITH_WARNINGS"
STATUS_REFUSED = "REFUSED"
STATUS_ERROR = "ERROR"

# Standard order of the 12 leads.
STANDARD_LEADS = ("I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6")

# layout -> (grid of leads per row, 10 s rhythm leads).
# Each grid row spans 10 s of paper, split equally between the columns.
LAYOUTS = {
    "3x4": ((("I", "aVR", "V1", "V4"), ("II", "aVL", "V2", "V5"), ("III", "aVF", "V3", "V6")), ()),
    "3x4+1": ((("I", "aVR", "V1", "V4"), ("II", "aVL", "V2", "V5"), ("III", "aVF", "V3", "V6")), ("II",)),
    "3x4+3": ((("I", "aVR", "V1", "V4"), ("II", "aVL", "V2", "V5"), ("III", "aVF", "V3", "V6")), ("V1", "II", "V5")),
    "6x2": (tuple(zip(("I", "II", "III", "aVR", "aVL", "aVF"), ("V1", "V2", "V3", "V4", "V5", "V6"))), ()),
    "6x2+1": (tuple(zip(("I", "II", "III", "aVR", "aVL", "aVF"), ("V1", "V2", "V3", "V4", "V5", "V6"))), ("II",)),
    "12x1": (tuple((lead,) for lead in STANDARD_LEADS), ()),
}

PAPER_SECONDS = 10.0
SPEEDS_MM_S = (12.5, 25.0, 50.0)
GAINS_MM_MV = (5.0, 10.0, 20.0)

SCENARIO_SUCCESS = "success"
SCENARIO_WARNINGS = "success_with_warnings"
SCENARIO_FAILURE = "failure"
SCENARIO_REFUSED = "refused"

# code -> (category, retryable, default message in English)
ERRORS = {
    # Comuns
    "INVALID_PAYLOAD": ("validation", False, "The payload must be a JSON object."),
    "UNSUPPORTED_CONTRACT_VERSION": ("validation", False, "Contract version not supported for this operation."),
    "REQUEST_ID_INVALID": ("validation", False, "Missing or invalid request_id (use the attempt UUID)."),
    "MOCK_SCENARIO_INVALID": ("validation", False, "Invalid test (mock) scenario."),
    "STORAGE_ERROR": ("internal", True, "Failed to read or write artifacts in S3."),
    "INTERNAL_ERROR": ("internal", True, "Unexpected internal error in the lambda."),
    # Reading — input validation
    "IMAGE_SOURCE_FORBIDDEN": ("validation", False, "Source reserved for tests cannot be read by the adapter."),
    "IMAGE_MISSING": ("validation", False, "Provide the ECG image (image.bucket and image.key)."),
    "IMAGE_NOT_FOUND": ("validation", False, "The given image does not exist in S3."),
    "IMAGE_FORMAT_UNSUPPORTED": ("validation", False, "Unsupported image format; send JPG, PNG or WebP."),
    "IMAGE_TOO_LARGE": ("validation", False, "The image exceeds the maximum size of 15 MB."),
    "IMAGE_TOO_SMALL": ("validation", False, "Insufficient resolution; the image must be at least 800x500 px."),
    "IMAGE_UNREADABLE": ("validation", False, "Could not decode the uploaded image."),
    "LAYOUT_MISSING": ("validation", False, "Provide the lead layout (e.g. 3x4+1)."),
    "LAYOUT_UNSUPPORTED": ("validation", False, "Unsupported lead layout."),
    "READ_REFUSED_UNREADABLE_TRACE": ("refused", False, "No method could recover a reliable signal; send another image for review."),
    "READ_SELECTOR_UNAVAILABLE": ("processing", False, "Approved selector model unavailable; processing not run."),
    "CALIBRATION_INVALID": ("validation", False, "Invalid calibration; speed must be 12.5/25/50 mm/s and gain 5/10/20 mm/mV."),
    # Reading — refusal (visible obstacle; the user must retake the photo)
    "READ_REFUSED_BLURRY": ("refused", False, "The image is blurry or out of focus; retake the photo with the paper in sharp focus."),
    "READ_REFUSED_CROPPED": ("refused", False, "The trace is cut off at the edges; frame the whole paper in the photo."),
    "READ_REFUSED_OCCLUDED": ("refused", False, "Something overlaps the trace (finger, glare, object or annotation); remove it and retake the photo."),
    "READ_REFUSED_LOW_CONTRAST": ("refused", False, "Insufficient contrast between the trace and the grid; improve the lighting."),
    "READ_REFUSED_NOT_ECG": ("refused", False, "The image does not look like a paper ECG."),
    "READ_REFUSED_LAYOUT_MISMATCH": ("refused", False, "The detected layout does not match the given layout."),
    # Reading — processing failure
    "READ_GRID_NOT_FOUND": ("processing", False, "Could not detect the millimeter grid of the paper."),
    "READ_TRACE_EXTRACTION_FAILED": ("processing", False, "Could not extract the trace of the leads."),
    # Interpretation — input validation
    "SEX_MISSING": ("validation", False, "Informe o sexo do paciente (patient.sex)."),
    "SEX_INVALID": ("validation", False, "Invalid patient sex; use \"F\" or \"M\"."),
    "SIGNAL_MISSING": ("validation", False, "Provide the ECG signal (signal)."),
    "SIGNAL_INVALID": ("validation", False, "ECG signal in an invalid format."),
    "SIGNAL_SAMPLING_RATE_INVALID": ("validation", False, "Invalid sampling rate; use between 100 and 2000 Hz."),
    "SIGNAL_LEAD_UNKNOWN": ("validation", False, "Unknown lead in the signal."),
    "SIGNAL_TOO_SHORT": ("validation", False, "Signal too short; each lead needs at least 1 s."),
    # Interpretation — processing failure
    "INTERP_NO_QRS_DETECTED": ("processing", False, "No QRS complex detected in the signal."),
    "INTERP_SIGNAL_TOO_NOISY": ("processing", False, "Signal too noisy for a reliable interpretation."),
}

WARNINGS = {
    # Reading
    "AMBIGUOUS_TRACE": "The trace has ambiguous regions that need human review.",
    "GRID_CALIBRATION_UNCERTAIN": "Grid calibration not confirmed.",
    "GEOMETRY_FALLBACK": "Geometry estimated by a fallback method.",
    "ORIENTATION_UNCERTAIN": "Image orientation could not be confirmed.",
    "METHODS_PARTIAL_RESULT": "Some methods produced no signals; check the individual statuses.",
    "LEAD_PARTIAL_TRACE": "Trace only partially extracted; unread stretches are returned as null.",
    "LEAD_SATURATED": "Trace touches the neighboring lead; amplitudes may be clipped.",
    "LOW_TRACE_GRID_CONTRAST": "Trace and grid have little colour separation; check the reading by eye.",
    "IMAGE_ROTATION_CORRECTED": "The image was tilted and was aligned automatically.",
    "CALIBRATION_PULSE_NOT_FOUND": "Calibration pulse not found; the provided calibration was used.",
    # Interpretation
    "LEAD_MISSING": "Lead missing from the signal; criteria that depend on it were skipped.",
    "LEAD_LOW_QUALITY": "Low-quality lead; measurements on it are less reliable.",
    "MEASUREMENT_UNRELIABLE": "Measurement computed with low confidence.",
}

READ_FAILURE_CODES = ("READ_GRID_NOT_FOUND", "READ_TRACE_EXTRACTION_FAILED", "STORAGE_ERROR", "INTERNAL_ERROR")
READ_REFUSAL_CODES = tuple(code for code, (category, _, _) in ERRORS.items() if category == "refused")
INTERP_FAILURE_CODES = ("INTERP_NO_QRS_DETECTED", "INTERP_SIGNAL_TOO_NOISY", "INTERNAL_ERROR")


class ContractError(Exception):
    """Catalog error: becomes an ERROR or REFUSED envelope, never a FunctionError."""

    def __init__(self, code, message=None, details=None):
        if code not in ERRORS:
            raise ValueError(f"error code not in the catalog: {code}")
        self.code = code
        self.message = message or ERRORS[code][2]
        self.details = details or {}
        super().__init__(f"{code}: {self.message}")

    def as_dict(self):
        category, retryable, _ = ERRORS[self.code]
        return {"code": self.code, "category": category, "retryable": retryable,
                "message": self.message, "details": self.details}


def warning(code, lead=None, message=None):
    if code not in WARNINGS:
        raise ValueError(f"warning code not in the catalog: {code}")
    item = {"code": code, "message": message or WARNINGS[code]}
    if lead:
        item["lead"] = lead
    return item


class Envelope:
    """Builds the standard response; measures the duration since creation."""

    def __init__(self, operation, engine, engine_version):
        self.operation = operation
        self.contract_version = INTERPRET_LEGACY_VERSION if operation in ("interpret", "legacy_read") else CONTRACT_VERSION
        self.engine = engine
        self.engine_version = engine_version
        self.request_id = None
        self.scenario = None
        self._started = time.monotonic()

    def _build(self, status, result=None, warnings=None, error=None):
        meta = {"engine": self.engine, "engine_version": self.engine_version,
                "duration_ms": int((time.monotonic() - self._started) * 1000)}
        if self.scenario:
            meta["mock_scenario"] = self.scenario
        return {"contract_version": self.contract_version, "operation": self.operation,
                "request_id": self.request_id, "status": status,
                "warnings": warnings or [], "error": error, "result": result, "meta": meta}

    def success(self, result, warnings=None):
        status = STATUS_OK_WITH_WARNINGS if warnings else STATUS_OK
        return self._build(status, result=result, warnings=warnings)

    def failure(self, exc):
        status = STATUS_REFUSED if ERRORS[exc.code][0] == "refused" else STATUS_ERROR
        return self._build(status, error=exc.as_dict())


def validate_common(event, operation="interpret"):
    """Validates what is common to both lambdas and returns (request_id, mock)."""
    if not isinstance(event, dict):
        raise ContractError("INVALID_PAYLOAD")
    version = event.get("contract_version", INTERPRET_LEGACY_VERSION if operation in ("interpret", "legacy_read") else CONTRACT_VERSION)
    allowed = ("1.0", "2.0") if operation == "interpret" else (("1.0",) if operation == "legacy_read" else ("2.0",))
    if version not in allowed:
        raise ContractError("UNSUPPORTED_CONTRACT_VERSION", details={"received": version})
    request_id = event.get("request_id")
    try:
        request_id = str(uuid.UUID(str(request_id)))
    except (ValueError, TypeError):
        raise ContractError("REQUEST_ID_INVALID", details={"received": request_id}) from None
    return request_id


def resolve_mock(event, default_scenario, allowed_scenarios, allowed_codes):
    """Picks the mock scenario: payload `mock` > env MOCK_SCENARIO > success.

    Exclusive to the mock lambdas; the real ones must ignore the `mock` field.
    """
    mock = event.get("mock") if isinstance(event, dict) else None
    if mock is None:
        mock = {}
    if not isinstance(mock, dict):
        raise ContractError("MOCK_SCENARIO_INVALID", details={"received": mock})
    scenario = mock.get("scenario") or default_scenario or SCENARIO_SUCCESS
    if scenario not in allowed_scenarios:
        raise ContractError("MOCK_SCENARIO_INVALID",
                            details={"received": scenario, "allowed": list(allowed_scenarios)})
    code = mock.get("code")
    if code is not None:
        allowed = allowed_codes.get(scenario, ())
        if code not in allowed:
            raise ContractError("MOCK_SCENARIO_INVALID",
                                details={"received_code": code, "allowed_codes": list(allowed)})
    return scenario, code
