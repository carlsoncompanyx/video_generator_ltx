#!/usr/bin/env python3
"""Idempotently verify and map the approved LTX-2.5 runtime files."""

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
DEFAULT_CACHE_REPO = "Torchem/LTX-2.5-Production"


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


def optional_entries_for(manifest: dict, profile: str | None = None) -> list[dict]:
    if not profile:
        return []
    selected = manifest["profiles"].get(profile)
    if not selected:
        raise SystemExit(f"Unknown model profile: {profile}")
    return list(selected.get("optional", []))


def prompt_enhancer_entries_for(optional_entries: list[dict]) -> list[dict]:
    """Return only the optional official Gemma 4 prompt-enhancer asset."""
    return [
        entry
        for entry in optional_entries
        if Path(entry.get("path", "")).name == "gemma4_e2b_it_bf16.safetensors"
    ]


def target_path(root: Path, entry: dict) -> Path:
    return root / entry["directory"] / Path(entry["path"]).name


def configured_cache_repo(manifest: dict) -> str:
    return os.environ.get("LTX25_CACHE_REPO") or manifest.get("cache_repo") or DEFAULT_CACHE_REPO


def cache_repo_for(manifest: dict, entry: dict) -> str:
    return entry.get("cache_repo") or configured_cache_repo(manifest)


def cache_model_root(cache_root: Path, repo: str) -> Path | None:
    if "/" not in repo:
        return None
    org, name = repo.split("/", 1)
    return cache_root / f"models--{org}--{name}"


def cached_snapshot_dirs(cache_root: Path, repo: str, revision: str = "main") -> list[Path]:
    model_root = cache_model_root(cache_root, repo)
    if model_root is None:
        return []
    snapshots_root = model_root / "snapshots"
    if not snapshots_root.is_dir():
        return []

    candidates: list[Path] = []
    refs = [revision]
    if revision != "main":
        refs.append("main")
    for ref_name in refs:
        ref = model_root / "refs" / ref_name
        snapshot_id = ref.read_text(encoding="utf-8").strip() if ref.is_file() else ""
        if snapshot_id:
            snapshot = snapshots_root / snapshot_id
            if snapshot.is_dir():
                candidates.append(snapshot)
    candidates.extend(sorted(path for path in snapshots_root.iterdir() if path.is_dir()))

    unique: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique.append(candidate)
    return unique


def cached_snapshot_for_repo(cache_root: Path, repo: str, revision: str = "main") -> Path | None:
    return next(iter(cached_snapshot_dirs(cache_root, repo, revision)), None)


def cached_file_in_snapshot(snapshot: Path, entry_path: str) -> Path | None:
    exact = snapshot / entry_path
    if exact.is_file():
        return exact

    filename = Path(entry_path).name
    preferred_parent = snapshot / Path(entry_path).parent
    if preferred_parent.is_dir():
        preferred = sorted(path for path in preferred_parent.glob(filename) if path.is_file())
        if preferred:
            return preferred[0]

    matches = sorted(path for path in snapshot.rglob(filename) if path.is_file())
    if matches:
        return matches[0]

    folded = filename.casefold()
    folded_matches = sorted(
        path for path in snapshot.rglob("*") if path.is_file() and path.name.casefold() == folded
    )
    return folded_matches[0] if folded_matches else None


def resolve_cached_entry(
    cache_root: Path, manifest: dict, entry: dict
) -> tuple[Path | None, str, Path | None]:
    repo = cache_repo_for(manifest, entry)
    revision = entry.get("revision", "main")
    snapshots = cached_snapshot_dirs(cache_root, repo, revision)
    for snapshot in snapshots:
        source = cached_file_in_snapshot(snapshot, entry["path"])
        if source is not None:
            return source, repo, snapshot
    return None, repo, snapshots[0] if snapshots else None


def cached_snapshot_path(
    cache_root: Path, entry: dict, manifest: dict | None = None
) -> Path | None:
    manifest = manifest or {"cache_repo": DEFAULT_CACHE_REPO}
    source, _, _ = resolve_cached_entry(cache_root, manifest, entry)
    return source


def map_cached_repository(root: Path, cache_root: Path, manifest: dict) -> list[str]:
    """Expose all supported files from the curated cache by symlink, never by copying."""
    repo = configured_cache_repo(manifest)
    snapshot = cached_snapshot_for_repo(cache_root, repo)
    if snapshot is None:
        print(f"cache snapshot: MISSING repo={repo} root={cache_root}")
        return []

    mapped: list[str] = []
    directories = manifest.get(
        "cache_directories",
        ["diffusion_models", "text_encoders", "vae", "latent_upscale_models", "loras", "model_patches"],
    )
    print(f"cache snapshot: FOUND repo={repo} snapshot={snapshot}")
    for directory in directories:
        source_root = snapshot / directory
        if not source_root.is_dir():
            print(f"cache directory: MISSING source={source_root}")
            continue
        count = 0
        for source in sorted(path for path in source_root.rglob("*") if path.is_file()):
            destination = root / source.relative_to(snapshot)
            destination.parent.mkdir(parents=True, exist_ok=True)
            same_target = False
            if destination.is_symlink():
                try:
                    same_target = destination.resolve() == source.resolve()
                except OSError:
                    same_target = False
            if same_target:
                count += 1
                continue
            if destination.exists() or destination.is_symlink():
                destination.unlink()
            destination.symlink_to(source)
            mapped.append(str(destination))
            count += 1
        print(f"cache directory: FOUND source={source_root} mapped={count}")
    return mapped


def map_cached_entries(
    entries: list[dict], root: Path, cache_root: Path, manifest: dict, label: str
) -> list[str]:
    """Expose cached HF files through ComfyUI without copying model weights."""
    mapped = []
    for entry in entries:
        destination = target_path(root, entry)
        if entry_matches(destination, entry):
            print(
                f"{label} model: FOUND filename={destination.name} "
                f"destination={destination} (already mapped)"
            )
            continue
        source, repo, snapshot = resolve_cached_entry(cache_root, manifest, entry)
        if source is None:
            print(
                f"{label} model: MISSING filename={destination.name} "
                f"cache_repo={repo} snapshot={snapshot or 'not found'} destination={destination}"
            )
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists() or destination.is_symlink():
            destination.unlink()
        destination.symlink_to(source)
        mapped.append(str(destination))
        print(
            f"{label} model: FOUND filename={destination.name} "
            f"source={source} destination={destination}"
        )
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
            "cache_repo": entry.get("cache_repo"),
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
    optional_entries = optional_entries_for(manifest, args.profile)
    prompt_enhancer_bootstrap = os.environ.get("LTX_PROMPT_ENHANCER_BOOTSTRAP", "false").strip().lower() in {"1", "true", "yes", "on"}
    prompt_enhancer_entries = prompt_enhancer_entries_for(optional_entries)
    cache_root = Path(os.environ.get("LTX25_CACHE_ROOT", "/runpod-volume/huggingface-cache/hub"))
    print(f"bootstrap profile: {args.profile}")
    print(f"detected RunPod HF cache repo: {configured_cache_repo(manifest)}")
    print(f"cache root: {cache_root}")
    map_cached_repository(root, cache_root, manifest)
    map_cached_entries(entries, root, cache_root, manifest, "required")
    map_cached_entries(optional_entries, root, cache_root, manifest, "optional")
    verify_hash = args.verify_sha256 or os.environ.get("LTX_VERIFY_MODEL_HASHES", "false").lower() == "true"
    missing = verify(entries, root, verify_hash=verify_hash)
    if missing and args.download and not args.verify_only:
        download(entries, root)
        missing = verify(entries, root, verify_hash=verify_hash)

    prompt_enhancer_missing = verify(prompt_enhancer_entries, root, verify_hash=verify_hash)
    if prompt_enhancer_bootstrap and prompt_enhancer_missing and args.download and not args.verify_only:
        print("prompt enhancer: missing; downloading the optional official Gemma 4 enhancer")
        download(prompt_enhancer_entries, root)
        prompt_enhancer_missing = verify(prompt_enhancer_entries, root, verify_hash=verify_hash)
    if missing:
        print(json.dumps({"status": "missing", "model_root": str(root), "missing": missing}, indent=2))
        print("Set MODEL_BOOTSTRAP_DOWNLOAD=true with an accepted HF_TOKEN to download approved files.")
        return 2
    if prompt_enhancer_bootstrap and prompt_enhancer_missing:
        print(json.dumps({"status": "missing_prompt_enhancer", "model_root": str(root), "missing": prompt_enhancer_missing}, indent=2))
        print("Prompt-enhancer bootstrap was requested but the optional official Gemma 4 file could not be mapped or downloaded.")
        return 3
    if prompt_enhancer_bootstrap:
        print("prompt enhancer: ready filename=gemma4_e2b_it_bf16.safetensors")
    print(json.dumps({"status": "ready", "profile": args.profile, "model_root": str(root), "files": [e["path"] for e in entries]}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
