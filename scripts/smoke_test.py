#!/usr/bin/env python3
"""Call a deployed endpoint and require a real MP4 response."""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import time
from pathlib import Path

import requests


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint-id", default=os.environ.get("RUNPOD_LTX25_ENDPOINT_ID"))
    parser.add_argument("--api-key", default=os.environ.get("RUNPOD_API_KEY"))
    parser.add_argument("--payload", type=Path, default=Path("examples/t2v.json"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/ltx25_smoke.mp4"))
    parser.add_argument("--timeout", type=int, default=3600)
    args = parser.parse_args()
    if not args.endpoint_id or not args.api_key:
        raise SystemExit("RUNPOD_LTX25_ENDPOINT_ID and RUNPOD_API_KEY are required")
    payload = json.loads(args.payload.read_text(encoding="utf-8"))
    headers = {"Authorization": f"Bearer {args.api_key}", "Content-Type": "application/json"}
    base = f"https://api.runpod.ai/v2/{args.endpoint_id}"
    submitted = requests.post(f"{base}/run", headers=headers, json=payload, timeout=60)
    submitted.raise_for_status()
    job_id = submitted.json().get("id")
    if not job_id:
        raise SystemExit(f"RunPod did not return a job id: {submitted.text[:1000]}")
    deadline = time.monotonic() + args.timeout
    result = None
    while time.monotonic() < deadline:
        response = requests.get(f"{base}/status/{job_id}", headers=headers, timeout=60)
        response.raise_for_status()
        result = response.json()
        status = result.get("status")
        print(json.dumps({"job_id": job_id, "status": status}))
        if status in {"COMPLETED", "FAILED", "CANCELLED", "TIMED_OUT"}:
            break
        time.sleep(5)
    if not result or result.get("status") != "COMPLETED":
        print(json.dumps(result or {}, indent=2))
        return 1
    output = (result.get("output") or {}).get("output") or result.get("output") or {}
    if output.get("type") == "base64":
        raw = base64.b64decode(output["data"])
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_bytes(raw)
    elif output.get("url"):
        downloaded = requests.get(output["url"], timeout=180)
        downloaded.raise_for_status()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_bytes(downloaded.content)
    else:
        raise SystemExit(f"Completed response did not contain a URL or base64 MP4: {json.dumps(result)[:2000]}")
    data = args.output.read_bytes()
    if not data.startswith(b"\x00\x00\x00") and b"ftyp" not in data[:64]:
        raise SystemExit("Output is not an MP4 container")
    print(json.dumps({"status": "REAL_MP4_SAVED", "path": str(args.output.resolve()), "bytes": len(data)}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
