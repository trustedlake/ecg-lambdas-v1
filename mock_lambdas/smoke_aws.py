"""Smoke test of the DEPLOYED lambdas: all scenarios and validations.

Uploads a test photo to mock-smoke/ in the images bucket, invokes both
lambdas in each scenario and checks status/code against LAMBDA_CONTRACT.md.
Without --image the reading success comes from the mock scenario (a blank image is refused by the real reader).
Interpretation receives the leads.json of the best OK method, as the backend does (contract 2.0).

    python mock_lambdas/smoke_aws.py [--image foto.jpg] [--profile trustedlake]
"""

import argparse
import io
import json
import sys
import uuid

import boto3
from botocore.config import Config

IMAGES_BUCKET = "ecg-lambdas-images-785236377332-us-east-1"
READING, INTERPRETATION = "ecg-reading", "ecg-interpretation"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", help="ECG photo (JPG/PNG); without it a synthetic image is used")
    parser.add_argument("--profile", default="trustedlake")
    parser.add_argument("--region", default="us-east-1")
    args = parser.parse_args()

    session = boto3.Session(profile_name=args.profile, region_name=args.region)
    s3 = session.client("s3", config=Config(signature_version="s3v4"))
    lam = session.client("lambda", config=Config(read_timeout=900, retries={"max_attempts": 0}))  # a real read takes minutes
    if args.image:
        body = open(args.image, "rb").read()
    else:
        from PIL import Image
        buf = io.BytesIO()
        Image.new("RGB", (1600, 1000), (255, 235, 235)).save(buf, format="PNG")
        body = buf.getvalue()
    key = f"mock-smoke/{uuid.uuid4()}.img"
    s3.put_object(Bucket=IMAGES_BUCKET, Key=key, Body=body)
    image = {"bucket": IMAGES_BUCKET, "key": key}

    def invoke(function, payload):
        response = lam.invoke(FunctionName=function, Payload=json.dumps(payload).encode())
        if response.get("FunctionError"):
            raise SystemExit(f"FunctionError in {function}: {response['Payload'].read()}")
        return json.loads(response["Payload"].read())

    def read(**extra):
        return invoke(READING, {"request_id": str(uuid.uuid4()), "image": image, "layout": "3x4+1",
                                "calibration": {"speed_mm_s": 25, "gain_mm_mv": 10}, **extra})

    ok = read() if args.image else read(mock={"scenario": "success"})
    methods = [m for m in (ok.get("result") or {}).get("methods", []) if m.get("status") == "OK"]
    if not methods:
        raise SystemExit(f"reading returned no OK method: {ok.get('status')} {(ok.get('error') or {}).get('code')}")
    best = max(methods, key=lambda m: m.get("selector_score") or 0)
    leads_file = best["files"]["leads"]
    signal = json.loads(s3.get_object(Bucket=leads_file["bucket"], Key=leads_file["key"])["Body"].read())
    uncertain = [l["name"] for l in signal.get("leads", []) if l.get("signal_source") == "traced_unvalidated"]
    print(f"best method {best['method']} (score {best.get('selector_score')}); uncertain leads: {uncertain or 'none'}")

    def interpret(**extra):
        payload = {"request_id": str(uuid.uuid4()), "contract_version": "2.0", "operation": "interpret",
                   "patient": {"sex": "M"}, "signal": signal, **extra}
        return invoke(INTERPRETATION, payload)

    real = interpret()
    summary = (real.get("result") or {}).get("summary") or {}
    print(f"interpretation: alert {summary.get('alert_level')}; reasons: {summary.get('alert_reasons')}")

    cases = [
        ("reading success", lambda: ok, ("OK", "OK_WITH_WARNINGS"), None),
        ("reading success_with_warnings", lambda: read(mock={"scenario": "success_with_warnings"}), "OK_WITH_WARNINGS", None),
        ("reading failure", lambda: read(mock={"scenario": "failure"}), "ERROR", "READ_TRACE_EXTRACTION_FAILED"),
        ("reading failure (grid)", lambda: read(mock={"scenario": "failure", "code": "READ_GRID_NOT_FOUND"}), "ERROR", "READ_GRID_NOT_FOUND"),
        ("reading refused", lambda: read(mock={"scenario": "refused"}), "REFUSED", "READ_REFUSED_LOW_CONTRAST"),
        ("reading refused (cropped)", lambda: read(mock={"scenario": "refused", "code": "READ_REFUSED_CROPPED"}), "REFUSED", "READ_REFUSED_CROPPED"),
        ("reading invalid layout", lambda: read(layout="5x5"), "ERROR", "LAYOUT_UNSUPPORTED"),
        ("reading without image", lambda: read(image=None), "ERROR", "IMAGE_MISSING"),
        ("reading missing image", lambda: read(image={"bucket": IMAGES_BUCKET, "key": "mock-smoke/does-not-exist.jpg"}), "ERROR", "IMAGE_NOT_FOUND"),
        ("reading invalid calibration", lambda: read(calibration={"speed_mm_s": 30}), "ERROR", "CALIBRATION_INVALID"),
        ("interpretation success", lambda: real, ("OK", "OK_WITH_WARNINGS"), None),
        ("interpretation success_with_warnings", lambda: interpret(mock={"scenario": "success_with_warnings"}), "OK_WITH_WARNINGS", None),
        ("interpretation failure", lambda: interpret(mock={"scenario": "failure"}), "ERROR", "INTERP_NO_QRS_DETECTED"),
        ("interpretation failure (noise)", lambda: interpret(mock={"scenario": "failure", "code": "INTERP_SIGNAL_TOO_NOISY"}), "ERROR", "INTERP_SIGNAL_TOO_NOISY"),
        ("interpretation without sex", lambda: interpret(patient={}), "ERROR", "SEX_MISSING"),
        ("interpretation invalid sex", lambda: interpret(patient={"sex": "X"}), "ERROR", "SEX_INVALID"),
        ("interpretation without signal", lambda: interpret(signal=None), "ERROR", "SIGNAL_MISSING"),
    ]
    failures = 0
    for name, call, status, code in cases:
        response = call()
        got_code = (response.get("error") or {}).get("code")
        passed = response.get("status") in ((status,) if isinstance(status, str) else status) and got_code == code
        failures += not passed
        print(f"{'PASS' if passed else 'FAIL'}  {name:38} {response.get('status'):17} {got_code or '':30} "
              f"{response['meta']['duration_ms']} ms")

    image = best["files"]["image"]
    url = s3.generate_presigned_url("get_object", Params={"Bucket": image["bucket"], "Key": image["key"]},
                                    ExpiresIn=3600)
    print(f"\npreview (1h): {url}")
    try:
        s3.delete_object(Bucket=IMAGES_BUCKET, Key=key)
    except Exception as exc:  # cleanup is best-effort
        print(f"warning: could not remove {key}: {exc}")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
