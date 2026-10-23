# Contract of the ECG reading and interpretation lambdas — v1.0

Source of truth for the contract between the Django backend and the two lambdas:

| Lambda | Mock (this repo, `mock_lambdas/`) | Real (Chris, OpenCV 5) |
|---|---|---|
| **Reading** (`operation: "read"`) | `ecg-reading` | replaces the mock while keeping this contract |
| **Interpretation** (`operation: "interpret"`) | `ecg-interpretation` | replaces the mock while keeping this contract |

The backend picks the function by name (env `AWS_LAMBDA_ECG_READING` / `AWS_LAMBDA_ECG_INTERPRETATION`);
swapping the mock for the real one is just a name change. The code catalog lives in code at
[`mock_lambdas/ecg_contract/__init__.py`](mock_lambdas/ecg_contract/__init__.py), and the real lambda can
reuse that module. A test fails if any catalog code is missing from this document.

## 1. General rules

- **Synchronous** invocation (`InvocationType=RequestResponse`), JSON payload and response. The AWS limit is 6 MB
  in each direction, so the image goes by S3 reference and the signal goes inline (12 leads at 500 Hz ≈ 150–450 KB).
- **Expected errors never become `FunctionError`.** The lambda always returns the envelope below. `FunctionError`
  (unhandled exception, timeout, OOM) means an infrastructure failure and the backend treats it as `INTERNAL_ERROR`.
- `request_id` is required and must be a UUID: use the attempt id (`ExamAttempt.id`). It goes into the
  logs and into the artifact prefix in S3.
- `contract_version` is optional and defaults to `"1.0"`. Any other value returns `UNSUPPORTED_CONTRACT_VERSION`.
- Unknown fields are ignored. This allows passing the reading output straight into the interpretation.
- Input validation runs **before** any processing. An invalid input returns a validation error
  even when a mock scenario was requested.
- Messages in English, ready to show to the user. Use `code` to decide the flow and `message` to display.

### 1.1 Response envelope (both lambdas)

```json
{
  "contract_version": "1.0",
  "operation": "read | interpret",
  "request_id": "received uuid (null if request_id itself is invalid)",
  "status": "OK | OK_WITH_WARNINGS | REFUSED | ERROR",
  "warnings": [{"code": "LEAD_PARTIAL_TRACE", "message": "…", "lead": "V2"}],
  "error": null,
  "result": { "…": "operation-specific; null when status is REFUSED/ERROR" },
  "meta": {"engine": "ecg-reading", "engine_version": "git sha", "duration_ms": 235,
           "mock_scenario": "success (mocks only)"}
}
```

| `status` | Meaning | `result` | `error` | What the app does |
|---|---|---|---|---|
| `OK` | Success with no caveats | filled | `null` | continues the flow |
| `OK_WITH_WARNINGS` | Success with warnings (`warnings` not empty) | filled | `null` | continues, showing the warnings |
| `REFUSED` | **Reading only.** Visible obstacle in the photo | `null` | filled (`category: "refused"`) | asks for a new photo and shows the reason and the preview |
| `ERROR` | Invalid input or processing failure | `null` | filled | shows the message; `retryable` says whether trying again is worthwhile |

Format of `error`:

```json
{"code": "READ_GRID_NOT_FOUND", "category": "validation | refused | processing | internal",
 "retryable": false, "message": "Readable message in English.", "details": {"…": "optional context"}}
```

## 2. Reading lambda (`read`)

### 2.1 Request

```json
{
  "contract_version": "1.0",
  "request_id": "3f6c1f9e-8f0a-4c1e-9a52-0d2b7e0b6a11",
  "image": {"bucket": "ecg-lambdas-images-785236377332-sa-east-1", "key": "<user>/<ts>/<exam>/<attempt>/original.jpg"},
  "layout": "3x4+1",
  "calibration": {"speed_mm_s": 25, "gain_mm_mv": 10},
  "mock": {"scenario": "refused", "code": "READ_REFUSED_BLURRY"}
}
```

| Field | Required | Rule | Error |
|---|---|---|---|
| `image.bucket`, `image.key` | yes | non-empty strings | `IMAGE_MISSING` |
| object in S3 | yes | must exist | `IMAGE_NOT_FOUND` |
| size | — | ≤ 15 MB | `IMAGE_TOO_LARGE` |
| format (by content, not by extension) | — | JPEG, PNG or WebP | `IMAGE_FORMAT_UNSUPPORTED` / `IMAGE_UNREADABLE` |
| resolution (after EXIF) | — | longer side ≥ 800 px and shorter side ≥ 500 px | `IMAGE_TOO_SMALL` |
| `layout` | yes | one of the layouts in table 2.2 | `LAYOUT_MISSING` / `LAYOUT_UNSUPPORTED` |
| `calibration` | no (default 25 mm/s, 10 mm/mV) | `speed_mm_s` ∈ {12.5, 25, 50}; `gain_mm_mv` ∈ {5, 10, 20} | `CALIBRATION_INVALID` |
| `mock` | no | mocks only; the real one ignores it (see section 4) | `MOCK_SCENARIO_INVALID` |

### 2.2 Supported layouts

Each grid row spans 10 s of paper, split equally between the columns. Rhythm leads span 10 s each.

| `layout` | Grid (rows) | Rhythm (10 s) |
|---|---|---|
| `3x4` | I aVR V1 V4 · II aVL V2 V5 · III aVF V3 V6 (2.5 s each) | — |
| `3x4+1` | same | II |
| `3x4+3` | same | V1, II, V5 |
| `6x2` | I V1 · II V2 · III V3 · aVR V4 · aVL V5 · aVF V6 (5 s each) | — |
| `6x2+1` | same | II |
| `12x1` | one lead per row (10 s each) | — |

### 2.3 Response `result` (status `OK` / `OK_WITH_WARNINGS`)

```json
{
  "layout": "3x4+1",
  "calibration": {"speed_mm_s": 25.0, "gain_mm_mv": 10.0},
  "sampling_rate_hz": 500,
  "units": "mV",
  "grid": {"small_box_px": 5.888, "large_box_px": 29.44, "origin_px": [64.0, 60.0],
           "rotation_deg": 1.8, "paper_bbox_px": [64.0, 60.0, 1536.0, 960.0]},
  "leads": [
    {"name": "I", "kind": "panel", "row": 0, "col": 0, "start_s": 0.0, "duration_s": 2.5,
     "quality": "good", "coverage": 1.0, "bbox_px": [64.0, 60.0, 432.0, 285.0], "baseline_y_px": 172.5,
     "samples": [0.0, 0.0012, 0.0031, "… 1250 values (mV) or null"]},
    {"name": "II", "kind": "rhythm", "row": 3, "col": 0, "start_s": 0.0, "duration_s": 10.0, "…": "…"}
  ],
  "artifacts": {
    "overlay": {"bucket": "ecg-lambdas-results-…", "key": "reading/<request_id>/<run_id>/overlay.png", "content_type": "image/png"},
    "signal":  {"bucket": "ecg-lambdas-results-…", "key": "reading/<request_id>/<run_id>/signal.json", "content_type": "application/json"}
  }
}
```

- `leads[].samples`: values in **mV** at the `sampling_rate_hz` rate, with length `duration_s × sampling_rate_hz`.
  Use **`null`** for stretches with no reading (never an invented 0). `start_s` is the instant on the paper (0–10 s), and
  leads in the same column are simultaneous.
- `kind`: `panel` (grid cell) or `rhythm` (long strip). The same lead can appear in both.
- `quality`: `good` | `partial` | `saturated` | `low`. `coverage`: fraction of non-null samples (0–1).
- `grid`, `bbox_px` and `baseline_y_px` are coordinates in pixels **of the original image after EXIF rotation**,
  so the app can draw or highlight regions if it wants to.
- `artifacts.overlay`: **PNG of the original image with the detected gridline and the extracted trace drawn on top.**
  It is the image the user sees to accept or reject the reading. The backend generates the presigned URL; the lambda
  never returns a URL.
- `artifacts.signal`: the same `result` (without `artifacts`), persisted for audit and reprocessing.
- Artifacts always go in the results bucket (lambda env `RESULTS_BUCKET`) and under
  `reading/<request_id>/<run_id>/`. Each execution generates a new `run_id`, so nothing is overwritten.

### 2.4 Refusal (`status: "REFUSED"`)

The reading is refused when there is a **visible obstacle** that the user resolves by taking another photo. The `error` carries the reason and,
when possible, a preview with the problem region highlighted:

```json
{"code": "READ_REFUSED_OCCLUDED", "category": "refused", "retryable": false,
 "message": "Something overlaps the trace (finger, glare, object or annotation); remove it and retake the photo.",
 "details": {"region_px": [560, 300, 960, 700],
             "preview": {"bucket": "…results…", "key": "reading/<request_id>/<run_id>/refusal_preview.png", "content_type": "image/png"}}}
```

Difference between `REFUSED` and a processing `ERROR`: in a refusal **the photo** has a problem and the fix is another
photo. In an `ERROR` with `category: "processing"` **the algorithm** could not read an apparently good photo.

## 3. Interpretation lambda (`interpret`)

### 3.1 Request

```json
{
  "contract_version": "1.0",
  "request_id": "3f6c1f9e-8f0a-4c1e-9a52-0d2b7e0b6a11",
  "patient": {"sex": "F"},
  "signal": {"sampling_rate_hz": 500, "units": "mV",
             "leads": [{"name": "II", "samples": [0.0, 0.01, "…"]}, "…"]},
  "mock": {"scenario": "success_with_warnings"}
}
```

`signal` has the same format as the reading `result`: the backend can pass the whole reading `result` through, because extra fields are ignored.

| Field | Rule | Error |
|---|---|---|
| `patient.sex` | required, `"F"` or `"M"` (the backend maps `Patient.gender`) | `SEX_MISSING` / `SEX_INVALID` |
| `signal` | required object | `SIGNAL_MISSING` / `SIGNAL_INVALID` |
| `signal.sampling_rate_hz` | number between 100 and 2000 | `SIGNAL_SAMPLING_RATE_INVALID` |
| `signal.units` | optional; if present, `"mV"` | `SIGNAL_INVALID` |
| `signal.leads` | non-empty list | `SIGNAL_MISSING` |
| `leads[].name` | one of the 12 standard leads (I, II, III, aVR, aVL, aVF, V1–V6) | `SIGNAL_LEAD_UNKNOWN` |
| `leads[].samples` | list of numbers or `null`, at most 60 s | `SIGNAL_INVALID` |
| valid samples | ≥ 1 s (`sampling_rate_hz` non-null samples) per lead | `SIGNAL_TOO_SHORT` |

Missing leads are **not** an error: they generate the `LEAD_MISSING` warning, and the criteria that depend on them come out as `not_evaluated`.

### 3.2 Response `result`

```json
{
  "patient": {"sex": "F"},
  "summary": {
    "classification": "normal | borderline | abnormal",
    "headline": "ECG within normal limits",
    "rhythm": "Sinus rhythm",
    "heart_rate_bpm": 72,
    "key_findings": ["Sinus rhythm", "Normal electrical axis"],
    "text": "Sinus rhythm, HR 72 bpm, QTc 416 ms. ECG within normal limits."
  },
  "details": {
    "measurements": {
      "heart_rate_bpm": {"value": 72, "unit": "bpm", "confidence": "high", "lead": "II"},
      "rr_ms": {"value": 833, "unit": "ms", "confidence": "high"},
      "pr_ms": {"value": 160, "unit": "ms", "confidence": "medium"},
      "qrs_ms": {"value": 92, "unit": "ms", "confidence": "medium"},
      "qt_ms": {"value": 380, "unit": "ms", "confidence": "medium"},
      "qtc_ms": {"value": 416, "unit": "ms", "confidence": "medium", "formula": "Bazett"},
      "qrs_axis_deg": {"value": 45, "unit": "deg", "confidence": "medium"}
    },
    "findings": [{"code": "SINUS_RHYTHM", "label": "Sinus rhythm", "severity": "info | warning | critical",
                  "evidence": "HR 72 bpm, regular RR"}],
    "criteria": [{"name": "QTc (Bazett)", "value": 416, "unit": "ms", "reference": "≤ 460 ms (female)",
                  "sex_specific": true, "status": "normal | abnormal | not_evaluated"}],
    "per_lead": [{"name": "I", "r_amplitude_mv": 0.711, "s_amplitude_mv": 0.321, "quality": "good"}]
  },
  "disclaimer": "Result generated automatically from the extracted signal. It does not replace a physician's assessment and report."
}
```

- `summary` is the **general summary** (main screen). `details` is the **detailed result**.
- `confidence`: `high` | `medium` | `low`. `measurements.*.value` can be `null` when not measurable.
- Sex enters the criteria with a sex-dependent threshold (`sex_specific: true`). In the mock: QTc ≤ 460 ms (F) or
  ≤ 450 ms (M), and Cornell ≤ 2.0 mV (F) or ≤ 2.8 mV (M). Sokolow-Lyon (≤ 3.5 mV) does not depend on sex.
- `findings[].code` is a stable identifier in `UPPER_SNAKE_CASE`. The mock emits `SINUS_RHYTHM`,
  `SINUS_BRADYCARDIA`, `SINUS_TACHYCARDIA`, `NORMAL_AXIS`, `AXIS_DEVIATION`, `QTC_PROLONGED` and `LVH_VOLTAGE`.
  The real lambda may add new codes, as long as they are documented here.
- The mock **measures** HR and amplitudes from the received signal (R peaks). PR, QRS, QT and the axis are fixed.

## 4. Test scenarios (mock lambdas only)

The scenario is chosen by `mock.scenario` in the payload. Without that field, the function's `MOCK_SCENARIO` env applies and, failing that, `success`.
`mock.code` picks the specific code; without it, the default in the table applies.

| `mock.scenario` | Reading | Interpretation | Accepted `mock.code` (default in **bold**) |
|---|---|---|---|
| `success` | `OK`, synthetic 72 bpm signal and overlay | `OK`, normal ECG | — |
| `success_with_warnings` | `OK_WITH_WARNINGS`: V2 partial (`null`), aVL saturated and rotation corrected | `OK_WITH_WARNINGS`: low-quality V2 and unreliable QT | — |
| `failure` | `ERROR` | `ERROR` | reading: **`READ_TRACE_EXTRACTION_FAILED`**, `READ_GRID_NOT_FOUND`, `STORAGE_ERROR`, `INTERNAL_ERROR` · interpretation: **`INTERP_NO_QRS_DETECTED`**, `INTERP_SIGNAL_TOO_NOISY`, `INTERNAL_ERROR` |
| `refused` | `REFUSED`, with preview | — (invalid: `MOCK_SCENARIO_INVALID`) | **`READ_REFUSED_BLURRY`**, `READ_REFUSED_CROPPED`, `READ_REFUSED_OCCLUDED`, `READ_REFUSED_LOW_CONTRAST`, `READ_REFUSED_NOT_ECG`, `READ_REFUSED_LAYOUT_MISMATCH` |

Validation errors (sections 2.1 and 3.1) are reproduced with genuinely invalid payloads, no scenario needed.
The mock overlay carries the mark "MOCK — synthetic signal", so it is not confused with a real reading.

## 5. Error catalog

`retryable: true` only for `STORAGE_ERROR` and `INTERNAL_ERROR` (transient failures). All the others have `retryable: false`.

### Common
| Code | Category | Message |
|---|---|---|
| `INVALID_PAYLOAD` | validation | The payload must be a JSON object. |
| `UNSUPPORTED_CONTRACT_VERSION` | validation | Contract version not supported; use 1.0. |
| `REQUEST_ID_INVALID` | validation | Missing or invalid request_id (use the attempt UUID). |
| `MOCK_SCENARIO_INVALID` | validation | Invalid test (mock) scenario. *(mocks only)* |
| `STORAGE_ERROR` | internal | Failed to read or write artifacts in S3. |
| `INTERNAL_ERROR` | internal | Unexpected internal error in the lambda. |

### Reading
| Code | Category | Message |
|---|---|---|
| `IMAGE_MISSING` | validation | Provide the ECG image (image.bucket and image.key). |
| `IMAGE_NOT_FOUND` | validation | The given image does not exist in S3. |
| `IMAGE_FORMAT_UNSUPPORTED` | validation | Unsupported image format; send JPG, PNG or WebP. |
| `IMAGE_TOO_LARGE` | validation | The image exceeds the maximum size of 15 MB. |
| `IMAGE_TOO_SMALL` | validation | Insufficient resolution; the image must be at least 800x500 px. |
| `IMAGE_UNREADABLE` | validation | Could not decode the uploaded image. |
| `LAYOUT_MISSING` | validation | Provide the lead layout (e.g. 3x4+1). |
| `LAYOUT_UNSUPPORTED` | validation | Unsupported lead layout. |
| `CALIBRATION_INVALID` | validation | Invalid calibration; speed must be 12.5/25/50 mm/s and gain 5/10/20 mm/mV. |
| `READ_REFUSED_BLURRY` | refused | The image is blurry or out of focus; retake the photo with the paper in sharp focus. |
| `READ_REFUSED_CROPPED` | refused | The trace is cut off at the edges; frame the whole paper in the photo. |
| `READ_REFUSED_OCCLUDED` | refused | Something overlaps the trace (finger, glare, object or annotation); remove it and retake the photo. |
| `READ_REFUSED_LOW_CONTRAST` | refused | Insufficient contrast between the trace and the grid; improve the lighting. |
| `READ_REFUSED_NOT_ECG` | refused | The image does not look like a paper ECG. |
| `READ_REFUSED_LAYOUT_MISMATCH` | refused | The detected layout does not match the given layout. |
| `READ_GRID_NOT_FOUND` | processing | Could not detect the millimeter grid of the paper. |
| `READ_TRACE_EXTRACTION_FAILED` | processing | Could not extract the trace of the leads. |

### Interpretation
| Code | Category | Message |
|---|---|---|
| `SEX_MISSING` | validation | Provide the patient sex (patient.sex). |
| `SEX_INVALID` | validation | Invalid patient sex; use "F" or "M". |
| `SIGNAL_MISSING` | validation | Provide the ECG signal (signal). |
| `SIGNAL_INVALID` | validation | ECG signal in an invalid format. |
| `SIGNAL_SAMPLING_RATE_INVALID` | validation | Invalid sampling rate; use between 100 and 2000 Hz. |
| `SIGNAL_LEAD_UNKNOWN` | validation | Unknown lead in the signal. |
| `SIGNAL_TOO_SHORT` | validation | Signal too short; each lead needs at least 1 s. |
| `INTERP_NO_QRS_DETECTED` | processing | No QRS complex detected in the signal. |
| `INTERP_SIGNAL_TOO_NOISY` | processing | Signal too noisy for a reliable interpretation. |

### Warnings (`warnings[].code`)
| Code | Lambda | Message |
|---|---|---|
| `LEAD_PARTIAL_TRACE` | reading | Trace only partially extracted; unread stretches are returned as null. |
| `LEAD_SATURATED` | reading | Trace touches the neighboring lead; amplitudes may be clipped. |
| `IMAGE_ROTATION_CORRECTED` | reading | The image was tilted and was aligned automatically. |
| `CALIBRATION_PULSE_NOT_FOUND` | reading | Calibration pulse not found; the provided calibration was used. |
| `LEAD_MISSING` | interpretation | Lead missing from the signal; criteria that depend on it were skipped. |
| `LEAD_LOW_QUALITY` | interpretation | Low-quality lead; measurements on it are less reliable. |
| `MEASUREMENT_UNRELIABLE` | interpretation | Measurement computed with low confidence. |

## 6. Requirements for the real lambdas

1. Same envelope, same field names, same codes. A new code goes first into this document and into the
   catalog (`ecg_contract`), with a **minor** version (1.1) if compatible and a **major** one (2.0) if it breaks something.
2. Ignore the `mock` field. `meta.engine` now identifies the real algorithm (e.g. `ecg-reading-opencv5`).
3. Reading: write `overlay.png` (original image + gridline + trace) and `signal.json` under
   `reading/<request_id>/<run_id>/` in the results bucket. On refusal, `refusal_preview.png` is recommended, not mandatory.
4. Do not raise an exception for an expected error. Wrap the handler in a `try/except` that returns `INTERNAL_ERROR`.
5. Run the `mock_lambdas/tests` suite swapping the handler for the real one: the validation and catalog tests apply
   to both implementations. The scenario tests are exclusive to the mock.
6. Timeout: the backend waits up to `AWS_LAMBDA_READ_TIMEOUT` (270 s), and the lambda must finish in ≤ 240 s.
