#!/usr/bin/env python3
"""Idempotently verify/download the approved LTX-2.5 runtime files."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "model_manifest.json"


def model_root() -> Path:
    return Path(os.environ.get("RUNPOD_NETWORK_VOLUME_PATH", "/runpod-volume")) / "models"


def entries_for(manifest: dict, profile: str | None = None, loras: list[str] | None = None):
    entries = []
    seen: set[tuple[str, str, str]] = set()
    if profile:
        selected = manifest["profiles"].get(profile)
        if not selected:
            raise SystemExit(f"Unknown model profile: {profile}")
        entries.extend(selected.get("required", []))
        loras = list(selected.get("required_loras", [])) + list(loras or [])
    for logical_id in loras or []:
        item = manifest["lora_repositories"].get(logical_id)
        if not item:
            raise SystemExit(f"Unknown approved LoRA: {logical_id}")
        entries.append(item)

    unique = []
    for entry in entries:
        key = (entry["repo"], entry.get("path", ""), entry["directory"])
        if key not in seen:
            seen.add(key)
            unique.append(entry)
    return unique


def target_path(root: Path, entry: dict) -> Path:
    return root / entry["directory"] / Path(entry["path"]).name


def cached_snapshot_path(cache_root: Path, entry: dict) -> Path | None:
    repo = entry.get("repo", "")
    if "/" not in repo:
        return None
    org, name = repo.split("/", 1)
    model_root = cache_root / f"models--{org}--{name}"
    revision = entry.get("revision", "main")
    ref = model_root / "refs" / revision
    snapshot_id = ref.read_text(encoding="utf-8").strip() if ref.is_file() else ""
    candidates = []
    if snapshot_id:
        candidates.append(model_root / "snapshots" / snapshot_id / entry["path"])
    candidates.extend(sorted((model_root / "snapshots").glob(f"*/{entry['path']}")))
    return next((candidate for candidate in candidates if candidate.is_file()), None)


def map_cached_entries(entries: list[dict], root: Path, cache_root: Path) -> list[str]:
    """Expose cached HF files through ComfyUI without copying model weights."""
    mapped = []
    for entry in entries:
        destination = target_path(root, entry)
        if entry_matches(destination, entry):
            continue
        source = cached_snapshot_path(cache_root, entry)
        if source is None:
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists() or destination.is_symlink():
            destination.unlink()
        destination.symlink_to(source)
        mapped.append(str(destination))
    return mapped


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def entry_matches(path: Path, entry: dict, verify_hash: bool = False) -> bool:
    if not path.is_file() or path.stat().st_size < 1024:
        return False
    expected_bytes = entry.get("expected_bytes")
    if expected_bytes and path.stat().st_size != expected_bytes:
        return False
    if verify_hash and entry.get("sha256") and sha256_file(path) != entry["sha256"].lower():
        return False
    return True


def verify(entries: list[dict], root: Path, verify_hash: bool = False) -> list[dict]:
    missing = []
    for entry in entries:
        path = target_path(root, entry)
        if entry_matches(path, entry, verify_hash=verify_hash):
            continue
        record = {
            "filename": path.name,
            "path": str(path),
            "repo": entry["repo"],
        }
        if path.is_file():
            record["actual_bytes"] = path.stat().st_size
        if entry.get("expected_bytes"):
            record["expected_bytes"] = entry["expected_bytes"]
        if verify_hash and entry.get("sha256") and path.is_file():
            record["sha256"] = sha256_file(path)
            record["expected_sha256"] = entry["sha256"]
        missing.append(record)
    return missing


def download(entries: list[dict], root: Path) -> None:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise SystemExit(f"huggingface_hub is required for model bootstrap: {exc}")

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    if not token:
        raise SystemExit("HF_TOKEN is required only when MODEL_BOOTSTRAP_DOWNLOAD=true; no token was printed or stored.")

    for entry in entries:
        destination = target_path(root, entry)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if entry_matches(destination, entry, verify_hash=bool(entry.get("sha256"))):
            print(f"model present: {destination.name}")
            continue
        print(f"downloading approved model file: {entry['repo']}::{entry['path']}")
        try:
            downloaded = hf_hub_download(
                repo_id=entry["repo"],
                filename=entry["path"],
                revision=entry.get("revision", "main"),
                token=token,
                local_dir=str(root),
            )
        except Exception as exc:
            message = str(exc)
            if "gated" in message.lower() or "access" in message.lower() or "403" in message:
                raise SystemExit(
                    f"HF_MODEL_ACCESS_REQUIRED: accept the LTX-2 community license and grant the read token access for {entry['repo']}"
                ) from exc
            raise SystemExit(f"MODEL_DOWNLOAD_FAILED for {entry['repo']}::{entry['path']}: {message}") from exc
        downloaded_path = Path(downloaded)
        if downloaded_path != destination and downloaded_path.exists():
            shutil.copy2(downloaded_path, destination)
        if not entry_matches(destination, entry, verify_hash=bool(entry.get("sha256"))):
            actual = destination.stat().st_size if destination.is_file() else 0
            expected = entry.get("expected_bytes", "unknown")
            raise SystemExit(f"Downloaded file failed verification at {destination}: {actual} bytes; expected {expected}")
        print(f"verified: {destination.name} ({destination.stat().st_size} bytes)")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", default="ltx25_int8_distilled")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--lora", action="append", default=[])
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--verify-sha256", action="store_true")
    args = parser.parse_args()

    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    root = model_root()
    root.mkdir(parents=True, exist_ok=True)
    entries = entries_for(manifest, args.profile, args.lora)
    cache_root = Path(os.environ.get("LTX25_CACHE_ROOT", "/runpod-volume/huggingface-cache/hub"))
    map_cached_entries(entries, root, cache_root)
    verify_hash = args.verify_sha256 or os.environ.get("LTX_VERIFY_MODEL_HASHES", "false").lower() == "true"
    missing = verify(entries, root, verify_hash=verify_hash)
    if missing and args.download and not args.verify_only:
        download(entries, root)
        missing = verify(entries, root, verify_hash=verify_hash)
    if missing:
        print(json.dumps({"status": "missing", "model_root": str(root), "missing": missing}, indent=2))
        print("Set MODEL_BOOTSTRAP_DOWNLOAD=true with an accepted HF_TOKEN to download approved files.")
        return 2
    print(json.dumps({"status": "ready", "profile": args.profile, "model_root": str(root), "files": [e["path"] for e in entries]}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())