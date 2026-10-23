# ECG lambdas contract — reading 2.0, mock interpretation 1.0

**v1 is the base**; the real reading uses `ecg-analyser/extractor/lambda_v1.py` without changing the six algorithms or the chromatic gate. Interpretation is still a mock, without the diagnoser. **No deploy, memory/timeout change, ECR image or cost was authorized.**

The contract code is in `mock_lambdas/ecg_contract/__init__.py`. The current ZIP build **still packages `reading/mock_handler.py`**, not the real adapter. The real one requires later packaging, including dependencies and the approved model (not distributed in the ZIP).

## 1. Envelope

Every reading response has `contract_version: "2.0"`, `operation: "read"`, `request_id`, `status`, `warnings`, `error`, `result`, `meta`.

- `OK`: at least one method produced a signal and there are no warnings.
- `OK_WITH_WARNINGS`: a signal exists, but there are quality warnings and/or methods that did not finish.
- `REFUSED`: no method produced a signal and **all** ended `BLOCKED`/`ABSTAINED`; `error.details.methods` contains the six results.
- `ERROR`: validation, infrastructure, selector unavailable or execution error with no viable output; `error.details.methods` contains the six results, if they got to run.
- `meta`: `engine: "ecg-analyser"`, version and measured `duration_ms`. No medical value is synthesized.

Expected errors are returned in the envelope, never as a `FunctionError`. `request_id` is a UUID; the `run_id` is created per execution to prevent overwriting. Images and signals go to S3; the synchronous response contains references, **not** all the data.

## 2. Real reading request

```json
{
  "contract_version": "2.0",
  "request_id": "3f6c1f9e-8f0a-4c1e-9a52-0d2b7e0b6a11",
  "image": {"bucket": "ecg-images", "key": "exam/attempt/original.png"},
  "layout": "3x4+1",
  "calibration": {"speed_mm_s": 25.0, "gain_mm_mv": 10.0}
}
```

- `contract_version`: only `2.0` in the real read. Without the field, 2.0 is assumed only for envelope compatibility. Wrong declared version: `UNSUPPORTED_CONTRACT_VERSION`.
- `image`: required S3 object. Only JPG/PNG/WebP, up to 15 MB, 24 megapixels, minimum dimension 800 × 500 in any orientation; the limits are validation limits, not clinical quality. `IMAGE_MISSING`, `IMAGE_NOT_FOUND`, `IMAGE_FORMAT_UNSUPPORTED`, `IMAGE_UNREADABLE`, `IMAGE_TOO_LARGE` or `IMAGE_TOO_SMALL`.
- Reading from the `ECG Images/1. Testes` source is forbidden, including as an S3 key (`IMAGE_SOURCE_FORBIDDEN`). Test sources in production/training are not selected automatically.
- `layout`: **required**, no detection. Only `3x4+1` (generic configuration `report_normal.json`) and `12x1` (`data_12x1.json`) have compatible generic configurations **in this package**. The other layouts of the old contract (`3x4`, `3x4+3`, `6x2`, `6x2+1`) return `LAYOUT_UNSUPPORTED` until they get verified explicit configs. Missing field: `LAYOUT_MISSING`.
- `calibration`: **required object, with both fields required**, no inference and no hidden default. Accepted speeds: 12.5, 25, 50 mm/s. Accepted gains: 5, 10, 20 mm/mV. Omitted/incorrect value: `CALIBRATION_INVALID`.
- `mock`: ignored by the real handler. Synthetic scenarios are exclusive to the old `mock_handler.py`.

## 3. Real reading response

```json
{
  "contract_version": "2.0",
  "operation": "read",
  "request_id": "3f6c1f9e-8f0a-4c1e-9a52-0d2b7e0b6a11",
  "status": "OK_WITH_WARNINGS",
  "warnings": [{"code": "METHODS_PARTIAL_RESULT", "message": "..."}],
  "error": null,
  "result": {
    "layout": "3x4+1",
    "calibration": {"speed_mm_s": 25.0, "gain_mm_mv": 10.0},
    "units": "mV",
    "methods": [
      {
        "method": "baseline_peaks",
        "version": "peaks34-323b4e6",
        "status": "OK",
        "reason": null,
        "selector_score": 87.5,
        "sampling_rate_hz": 200.0,
        "files": {
          "json": {"bucket": "results", "key": "reading/<request_id>/<run_id>/baseline_peaks/result.json", "content_type": "application/json"},
          "image": {"bucket": "results", "key": "reading/<request_id>/<run_id>/baseline_peaks/preview.png", "content_type": "image/png"},
          "leads": {"bucket": "results", "key": "reading/<request_id>/<run_id>/baseline_peaks/leads.json", "content_type": "application/json"}
        }
      }
    ]
  },
  "meta": {"engine": "ecg-analyser", "engine_version": "dev", "duration_ms": 0}
}
```

**Illustrative sample, not an executed result.** `methods` always has six methods in registry order: `baseline_peaks`, `chromatic_ink`, `faint_ink`, `uneven_photo`, `grayscale_contrast`, `same_appearance`.

| Individual status | Behavior |
|---|---|
| `OK` | Measured extraction and chromatic gate allowed; `result.json`, `preview.png` and `leads.json` in S3. Real `selector_score`. |
| `BLOCKED` | True chromatic gate; `result.json` and `rejection_preview.png` in S3, **no signal**. |
| `ABSTAINED` | `MethodAbstention`, invalid grid or no lead measured; JSON and refusal image, no signal. |
| `ERROR` | Unexpected error in the method; JSON persisted, `selector_score=null`, no signal. |

`selector_score` uses exclusively the approved selector (`selector/score.py`: `load_leads`, `score_leads`, `record_score`) on the arrays measured by the method. Immutable `model_selector.pkl` model, SHA-256 starting with `cb0f395367ec`. **A missing file leads to `READ_SELECTOR_UNAVAILABLE`, not to a fictitious score.** Selecting by the highest score is the backend's responsibility. The six methods have research status; `OK` is not clinical validation.

Native rate in Hz = `grid.x.pixels_per_mm × speed_mm_s`; do not resample the export to 500 Hz. In `leads.json` each lead carries `name`, `kind`, `row`, `col`, `start_s`, `duration_s`, `quality`, `coverage`, `bbox_px`, `baseline_y_px`, `samples` (mV), `signal_source` (`analysis`, or `traced_unvalidated` when under 1 s of the lead passed the extractor's validity filter and the traced line is sent instead, with `quality: "uncertain"`; interpretation then forces review). `II_longo` is presented as `II`, `kind: rhythm`. NaN values and invalid samples are exported as `null`. No missing stretch is interpolated for delivery; the internal resampling to 150 Hz is **only for the approved classifier**, preserving the gaps.

The methods can emit `LEAD_PARTIAL_TRACE`, `AMBIGUOUS_TRACE`, `GRID_CALIBRATION_UNCERTAIN`, `GEOMETRY_FALLBACK`, `ORIENTATION_UNCERTAIN` and `METHODS_PARTIAL_RESULT`. `INCOMPLETE_TRACE` was mapped to `LEAD_PARTIAL_TRACE`. General warnings omit `lead` when they have no specific lead (`lead: null` is neither needed nor emitted).

## 4. The interpretation contract remains a mock

The `operation: "interpret"` endpoint still accepts `contract_version: "1.0"` and the legacy `signal` format, and also explicitly accepts `"2.0"`. The backend **must not pass** the 2.0 reading `result` directly to interpretation: it must select the best method with `status: "OK"`, download its `files.leads` and send that object as `signal`, together with `patient.sex` and `request_id`. No automatic interpretation call was added to the adapter.

Legacy signal compatibility rule: `sampling_rate_hz` between 100–2000, unit `mV`, `leads[].name` among I, II, III, aVR, aVL, aVF, V1–V6, `samples` numbers or `null`, at least 1 valid second per lead; warnings for missing leads. The interpretation is **synthetic** and is not a medical diagnosis. Full specification of the previous mock: `LEGACY_V1_CONTRACT.md`.

## 5. Error / warning catalog

All codes below are defined in `ecg_contract`; all the old codes are in `LEGACY_V1_CONTRACT.md`. The new codes are:

- `IMAGE_SOURCE_FORBIDDEN` — validation: source reserved for tests.
- `READ_REFUSED_UNREADABLE_TRACE` — refusal when all abstained and no chromatic gate blocked.
- `READ_SELECTOR_UNAVAILABLE` — processing unavailable because the approved model is missing.
- `AMBIGUOUS_TRACE` — samples with uncertainty/ambiguity.
- `GRID_CALIBRATION_UNCERTAIN` — grid without consistent calibration.
- `GEOMETRY_FALLBACK` — geometry by fallback.
- `ORIENTATION_UNCERTAIN` — uncertain orientation.
- `METHODS_PARTIAL_RESULT` — at least one method delivered no signal.

## 6. Limitations before production

The provided ZIP does not include `selector/model_selector.pkl`, does not include images from the `ECG Images/0. Treino` folder and does not contain images from the reserved set. Unit tests can verify structure, rules, gate and fake S3, **not clinical performance/accuracy** nor the time and peak memory on real images. The container/ECR, size/memory/timeout and the backend integration require separate decisions and cost assessment.

### Full index of the shared catalog codes

| Code | Type |
|---|---|
| `INVALID_PAYLOAD` | `validation` error |
| `UNSUPPORTED_CONTRACT_VERSION` | `validation` error |
| `REQUEST_ID_INVALID` | `validation` error |
| `MOCK_SCENARIO_INVALID` | `validation` error |
| `STORAGE_ERROR` | `internal` error |
| `INTERNAL_ERROR` | `internal` error |
| `IMAGE_SOURCE_FORBIDDEN` | `validation` error |
| `IMAGE_MISSING` | `validation` error |
| `IMAGE_NOT_FOUND` | `validation` error |
| `IMAGE_FORMAT_UNSUPPORTED` | `validation` error |
| `IMAGE_TOO_LARGE` | `validation` error |
| `IMAGE_TOO_SMALL` | `validation` error |
| `IMAGE_UNREADABLE` | `validation` error |
| `LAYOUT_MISSING` | `validation` error |
| `LAYOUT_UNSUPPORTED` | `validation` error |
| `READ_REFUSED_UNREADABLE_TRACE` | `refused` error |
| `READ_SELECTOR_UNAVAILABLE` | `processing` error |
| `CALIBRATION_INVALID` | `validation` error |
| `READ_REFUSED_BLURRY` | `refused` error |
| `READ_REFUSED_CROPPED` | `refused` error |
| `READ_REFUSED_OCCLUDED` | `refused` error |
| `READ_REFUSED_LOW_CONTRAST` | `refused` error |
| `READ_REFUSED_NOT_ECG` | `refused` error |
| `READ_REFUSED_LAYOUT_MISMATCH` | `refused` error |
| `READ_GRID_NOT_FOUND` | `processing` error |
| `READ_TRACE_EXTRACTION_FAILED` | `processing` error |
| `SEX_MISSING` | `validation` error |
| `SEX_INVALID` | `validation` error |
| `SIGNAL_MISSING` | `validation` error |
| `SIGNAL_INVALID` | `validation` error |
| `SIGNAL_SAMPLING_RATE_INVALID` | `validation` error |
| `SIGNAL_LEAD_UNKNOWN` | `validation` error |
| `SIGNAL_TOO_SHORT` | `validation` error |
| `INTERP_NO_QRS_DETECTED` | `processing` error |
| `INTERP_SIGNAL_TOO_NOISY` | `processing` error |
| `AMBIGUOUS_TRACE` | warning |
| `GRID_CALIBRATION_UNCERTAIN` | warning |
| `GEOMETRY_FALLBACK` | warning |
| `ORIENTATION_UNCERTAIN` | warning |
| `METHODS_PARTIAL_RESULT` | warning |
| `LEAD_PARTIAL_TRACE` | warning |
| `LEAD_SATURATED` | warning |
| `IMAGE_ROTATION_CORRECTED` | warning |
| `CALIBRATION_PULSE_NOT_FOUND` | warning |
| `LEAD_MISSING` | warning |
| `LEAD_LOW_QUALITY` | warning |
| `MEASUREMENT_UNRELIABLE` | warning |
