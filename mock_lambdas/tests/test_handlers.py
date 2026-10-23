"""Tests of the v1 contract of the mock lambdas (no AWS; in-memory fake S3).

    pip install pillow boto3
    python -m unittest discover -s mock_lambdas/tests -v
"""

import importlib.util
import io
import json
import os
import sys
import unittest
import uuid

from botocore.exceptions import ClientError
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "interpretation"))  # diagnoser.py sits next to the handler
os.environ.setdefault("RESULTS_BUCKET", "results-test")
os.environ.pop("MOCK_SCENARIO", None)

from ecg_contract import ERRORS, LAYOUTS, WARNINGS  # noqa: E402


def _load(name, filename="handler.py"):
    spec = importlib.util.spec_from_file_location(f"{name}_handler", os.path.join(ROOT, name, filename))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


interpretation = _load("interpretation")


class FakeS3:
    def __init__(self):
        self.objects = {}

    def head_object(self, Bucket, Key):
        if (Bucket, Key) not in self.objects:
            raise ClientError({"Error": {"Code": "404"}}, "HeadObject")
        return {"ContentLength": len(self.objects[(Bucket, Key)])}

    def get_object(self, Bucket, Key):
        return {"Body": io.BytesIO(self.objects[(Bucket, Key)])}

    def put_object(self, Bucket, Key, Body, ContentType):
        self.objects[(Bucket, Key)] = Body


def _image_bytes(fmt="JPEG", size=(1600, 1000)):
    buf = io.BytesIO()
    Image.new("RGB", size, (255, 235, 235)).save(buf, format=fmt)
    return buf.getvalue()


class InterpretationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import math
        samples = [round(math.exp(-(((i / 500) % (60 / 72) - .2) / .012) ** 2), 4) for i in range(2500)]
        cls.signal = {"sampling_rate_hz": 500, "units": "mV",
                      "leads": [{"name": n, "samples": samples} for n in ("I", "II", "III", "aVR", "aVL", "aVF",
                                                                           "V1", "V2", "V3", "V4", "V5", "V6")]}

    def call(self, **overrides):
        event = {"contract_version": "1.0", "request_id": str(uuid.uuid4()),
                 "patient": {"sex": "F"}, "signal": self.signal}
        event.update(overrides)
        return interpretation.handler(event, None)

    def assertError(self, response, code):
        self.assertEqual(response["status"], "ERROR", response["error"])
        self.assertEqual(response["error"]["code"], code)
        self.assertIsNone(response["result"])

    def test_success_summary_and_details(self):
        response = self.call(mock={"scenario": "success"})
        self.assertEqual(response["status"], "OK", response["warnings"])
        summary, details = response["result"]["summary"], response["result"]["details"]
        self.assertEqual(summary["classification"], "normal")
        self.assertEqual(summary["heart_rate_bpm"], 72)
        self.assertIn("Sinus rhythm", summary["key_findings"])
        self.assertEqual(len(details["per_lead"]), 12)
        self.assertIn("NORMAL_AXIS", {f["code"] for f in details["findings"]})
        self.assertTrue(response["result"]["disclaimer"])

    def test_sex_changes_thresholds(self):
        female = self.call(patient={"sex": "F"}, mock={"scenario": "success"})["result"]["details"]["criteria"]
        male = self.call(patient={"sex": "M"}, mock={"scenario": "success"})["result"]["details"]["criteria"]
        self.assertIn("female", female[0]["reference"])
        self.assertIn("450", male[0]["reference"])
        self.assertNotEqual(female[1]["reference"], male[1]["reference"])

    def test_real_diagnoser_path(self):
        response = self.call()
        self.assertIn(response["status"], ("OK", "OK_WITH_WARNINGS"), response)
        result = response["result"]
        self.assertTrue(70 <= result["summary"]["heart_rate_bpm"] <= 74)
        codes = {f["code"] for f in result["details"]["findings"]}
        self.assertFalse(codes & {"ST_ELEVATION", "ST_DEPRESSION", "PATHOLOGICAL_Q", "RBBB_PATTERN", "LBBB_PATTERN"})
        self.assertIsNone(result["details"]["measurements"]["pr_ms"]["value"])
        self.assertTrue(result["disclaimer"])
        self.assertIn(result["summary"]["alert_level"], ("ok", "review"))
        self.assertTrue(all(f["alert_level"] in ("ok", "review", "urgent") for f in result["details"]["findings"]))

    def test_marked_tachycardia_is_urgent(self):
        fast = {"sampling_rate_hz": 500, "leads": [{"name": "II", "samples": ([0.0] * 87 + [1.0] + [0.0] * 87) * 30}]}
        result = self.call(signal=fast)["result"]
        self.assertGreaterEqual(result["summary"]["heart_rate_bpm"], 150)
        self.assertEqual(result["summary"]["alert_level"], "urgent")

    def test_cornell_evidence_marks_both_leads(self):
        def lead(name, r, s):
            beat = [0.0] * 400
            beat[190:198] = [r * 0.5, r, r * 0.5, -s * 0.5, -s, -s * 0.5, 0, 0]
            return {"name": name, "samples": beat * 8}
        names = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
        signal = {"sampling_rate_hz": 500, "leads": [lead(n, 1.2 if n == "aVL" else 0.6, 1.4 if n == "V3" else 0.2) for n in names]}
        result = self.call(signal=signal, patient={"sex": "F"})["result"]
        finding = next(f for f in result["details"]["findings"] if f["code"] == "LVH_VOLTAGE")
        self.assertEqual(len(finding["marks"]), 2)
        self.assertTrue(any("aVL" in m["label"] for m in finding["marks"]) and any("V3" in m["label"] for m in finding["marks"]))

    def test_success_with_warnings(self):
        response = self.call(mock={"scenario": "success_with_warnings"})
        self.assertEqual(response["status"], "OK_WITH_WARNINGS")
        self.assertEqual({w["code"] for w in response["warnings"]}, {"LEAD_LOW_QUALITY", "MEASUREMENT_UNRELIABLE"})
        self.assertEqual(response["result"]["details"]["measurements"]["qt_ms"]["confidence"], "low")

    def test_missing_leads_warn(self):
        signal = {"sampling_rate_hz": 500, "leads": [l for l in self.signal["leads"] if l["name"] == "II"]}
        response = self.call(signal=signal)
        self.assertEqual(response["status"], "OK_WITH_WARNINGS")
        self.assertEqual(len([w for w in response["warnings"] if w["code"] == "LEAD_MISSING"]), 11)

    def test_failure_codes(self):
        self.assertError(self.call(mock={"scenario": "failure"}), "INTERP_NO_QRS_DETECTED")
        self.assertError(self.call(mock={"scenario": "failure", "code": "INTERP_SIGNAL_TOO_NOISY"}),
                         "INTERP_SIGNAL_TOO_NOISY")
        self.assertError(self.call(mock={"scenario": "refused"}), "MOCK_SCENARIO_INVALID")

    def test_flat_signal_has_no_qrs(self):
        flat = {"sampling_rate_hz": 500, "leads": [{"name": "II", "samples": [0.0] * 1000}]}
        result = self.call(signal=flat)["result"]
        self.assertEqual(result["summary"]["alert_level"], "review")
        checks = {c["code"]: c for c in result["details"]["checks"]}
        self.assertEqual(checks["HEART_RATE"]["status"], "not_possible")
        self.assertEqual(checks["QT_QTC"]["status"], "not_implemented")
        self.assertTrue(any(r.startswith("Could not check: Heart rate") for r in result["summary"]["alert_reasons"]))

    def test_validation_errors(self):
        cases = [
            ({"patient": None}, "SEX_MISSING"),
            ({"patient": {"sex": "female"}}, "SEX_INVALID"),
            ({"signal": None}, "SIGNAL_MISSING"),
            ({"signal": "x"}, "SIGNAL_INVALID"),
            ({"signal": {"sampling_rate_hz": 500, "leads": []}}, "SIGNAL_MISSING"),
            ({"signal": {"sampling_rate_hz": 50, "leads": [{"name": "II", "samples": [0] * 600}]}}, "SIGNAL_SAMPLING_RATE_INVALID"),
            ({"signal": {"sampling_rate_hz": 500, "units": "uV", "leads": [{"name": "II", "samples": [0] * 600}]}}, "SIGNAL_INVALID"),
            ({"signal": {"sampling_rate_hz": 500, "leads": [{"name": "V7", "samples": [0] * 600}]}}, "SIGNAL_LEAD_UNKNOWN"),
            ({"signal": {"sampling_rate_hz": 500, "leads": [{"name": "II", "samples": [0, "a"]}]}}, "SIGNAL_INVALID"),
            ({"signal": {"sampling_rate_hz": 500, "leads": [{"name": "II", "samples": [0.1] * 100}]}}, "SIGNAL_TOO_SHORT"),
            ({"request_id": None}, "REQUEST_ID_INVALID"),
        ]
        for overrides, code in cases:
            with self.subTest(code=code):
                self.assertError(self.call(**overrides), code)


class CatalogTests(unittest.TestCase):
    def test_every_error_has_category_and_message(self):
        for code, (category, retryable, message) in ERRORS.items():
            self.assertIn(category, ("validation", "refused", "processing", "internal"), code)
            self.assertIsInstance(retryable, bool)
            self.assertTrue(message)

    def test_contract_doc_lists_every_code(self):
        doc = open(os.path.join(os.path.dirname(ROOT), "LAMBDA_CONTRACT.md"), encoding="utf-8").read()
        missing = [code for code in [*ERRORS, *WARNINGS] if f"`{code}`" not in doc]
        self.assertEqual(missing, [], "codes missing from LAMBDA_CONTRACT.md")


if __name__ == "__main__":
    unittest.main()


class TestRhythmDoubleDetection(unittest.TestCase):
    def test_wide_t_wave_next_to_qrs_is_not_a_second_beat(self):
        import math
        import diagnoser
        fs, n = 200, 2000
        x = [0.0] * n
        for k in range(15):
            i = int((0.3 + k * 0.63) * fs)
            x[i], x[i + 1] = -0.3, 0.26
            for j in range(int(0.2 * fs)):
                x[i + int(0.12 * fs) + j] -= 0.35 * math.sin(math.pi * j / (0.2 * fs))
        peaks = diagnoser.detect_beats(x, fs)
        rr = [b - a for a, b in zip(peaks, peaks[1:])]
        self.assertLess(max(rr) - min(rr), 0.1 * fs)
