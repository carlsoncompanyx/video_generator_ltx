#!/usr/bin/env python3
"""RunPod Serverless adapter for the current ComfyUI LTX-2.5 templates."""

from __future__ import annotations

import base64
import binascii
import copy
import ipaddress
import json
import logging
import math
import mimetypes
import os
import re
import subprocess
import threading
import time
import uuid
from pathlib import Path
from urllib.parse import urlparse

import requests

try:
    import runpod
except ImportError:  # local contract tests do not install the worker runtime
    runpod = None

try:
    from runpod.serverless.utils import rp_upload
except ImportError:
    rp_upload = None


LOG = logging.getLogger("ltx25-worker")
logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))

ROOT = Path(__file__).resolve().parent
WORKFLOW_DIR = ROOT / "workflows"
LORA_REGISTRY_PATH = ROOT / "config" / "ltx_loras.json"
MODEL_MANIFEST_PATH = ROOT / "model_manifest.json"
COMFY_HOST = os.environ.get("COMFY_HOST", "127.0.0.1:8188")
COMFY_BASE = f"http://{COMFY_HOST}"
COMFY_INPUT_DIR = Path(os.environ.get("COMFY_INPUT_DIR", "/comfyui/input"))
COMFY_TIMEOUT_SECONDS = int(os.environ.get("COMFY_TIMEOUT_SECONDS", "3600"))
MAX_INPUT_BYTES = int(os.environ.get("LTX_MAX_INPUT_BYTES", str(100 * 1024 * 1024)))
MAX_BASE64_OUTPUT_BYTES = int(os.environ.get("LTX_MAX_BASE64_OUTPUT_BYTES", str(20 * 1024 * 1024)))

WORKFLOWS = {
    "t2v": "ltx25_t2v.json",
    "i2v": "ltx25_i2v.json",
    "flf2v": "ltx25_flf2v.json",
    "union_control": "ltx25_union_control.json",
    "motion_track": "ltx25_motion_track.json",
}
CORE_MODES = {"t2v", "i2v", "flf2v"}
CONTROL_MODES = {"union_control", "motion_track"}
UNION_CONTROL_TYPES = {"depth"}
CONTROL_LORA_BY_MODE = {"union_control": "union_control", "motion_track": "motion_track"}
MAX_TRACKS = 16
MAX_TRACK_CONTROL_POINTS = 64
ALL_MODES = CORE_MODES | CONTROL_MODES | {"pixel_upscale", "health"}
MODEL_FILES = {
    "ltx25_int8_distilled": {
        "unet": "ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors",
        "clip": "gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors",
    },
    "ltx25_bf16_control": {
        "unet": "ltx-2.5-22b-distilled-transformer-bf16.safetensors",
        "clip": "gemma4-12b-with-proj-ltx-2.5-bf16.safetensors",
    },
    "ltx25_bf16_core": {
        "unet": "ltx-2.5-22b-dev-transformer-bf16.safetensors",
        "clip": "gemma4-12b-with-proj-ltx-2.5-bf16.safetensors",
    },
}
GEMMA_API_CHECKPOINT = "ltx-2.5-22b-distilled-transformer-bf16.safetensors"


class ContractError(Exception):
    def __init__(self, code: str, message: str, details: dict | None = None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = details or {}


class GpuMemorySampler:
    """Sample aggregate GPU memory while ComfyUI executes a job."""

    def __init__(self, interval_seconds: float = 0.5):
        self.interval_seconds = interval_seconds
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._stopped = False
        self._baseline_mb: int | None = None
        self._peak_mb: int | None = None
        self._final_mb: int | None = None
        self._samples = 0

    @staticmethod
    def _read_used_mb() -> int | None:
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                timeout=3,
                check=False,
            )
            values = [int(float(line.strip())) for line in result.stdout.splitlines() if line.strip()]
            return max(values) if values else None
        except (OSError, ValueError, subprocess.SubprocessError):
            return None

    def _sample_loop(self) -> None:
        while not self._stop.wait(self.interval_seconds):
            current = self._read_used_mb()
            if current is not None:
                self._samples += 1
                self._peak_mb = max(self._peak_mb or current, current)

    def start(self) -> None:
        self._baseline_mb = self._read_used_mb()
        if self._baseline_mb is not None:
            self._peak_mb = self._baseline_mb
        self._thread = threading.Thread(target=self._sample_loop, name="ltx-gpu-sampler", daemon=True)
        self._thread.start()

    def stop(self) -> dict:
        if not self._stopped:
            self._stop.set()
            if self._thread is not None:
                self._thread.join(timeout=5)
            self._final_mb = self._read_used_mb()
            if self._final_mb is not None:
                self._peak_mb = max(self._peak_mb or self._final_mb, self._final_mb)
            self._stopped = True
        return {
            "baseline_used_mb": self._baseline_mb,
            "peak_used_mb": self._peak_mb,
            "final_used_mb": self._final_mb,
            "samples": self._samples,
            "source": "nvidia-smi aggregate GPU memory.used",
        }


def error_result(exc: ContractError, normalized: dict | None = None) -> dict:
    LOG.error('LTX_HANDLER_FAILURE code=%s message=%s details=%s', exc.code, exc.message, exc.details)
    result = {
        "status": "FAILED",
        "error": {"code": exc.code, "message": exc.message, "details": exc.details},
    }
    if normalized is not None:
        result["normalized"] = normalized
    return result


def as_input(job: dict) -> dict:
    value = job.get("input", job)
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ContractError("INVALID_JSON", "input must be valid JSON") from exc
    if not isinstance(value, dict):
        raise ContractError("INVALID_INPUT", "input must be an object")
    action = str(value.get("action", "")).lower()
    if action in {"text_to_video", "image_to_video"}:
        value = {
            **value,
            "mode": "i2v" if action == "image_to_video" else "t2v",
            "duration_seconds": value.get("duration", value.get("duration_seconds", 5)),
        }
        if action == "image_to_video" and not (value.get("image_url") or value.get("image_base64")):
            image = value.get("image")
            if isinstance(image, str):
                if image.startswith(("http://", "https://")):
                    value["image_url"] = image
                else:
                    value["image_base64"] = image
            elif image is not None:
                raise ContractError("INVALID_MEDIA", "image must be an HTTPS URL or base64 image")
    return value

def clamp_dimension(value: object, name: str) -> tuple[int, str | None]:
    try:
        numeric = int(value)
    except (TypeError, ValueError) as exc:
        raise ContractError("INVALID_RESOLUTION", f"{name} must be an integer") from exc
    if numeric < 256 or numeric > 2048:
        raise ContractError("INVALID_RESOLUTION", f"{name} must be between 256 and 2048", {name: numeric})
    normalized = (numeric // 32) * 32
    if normalized < 256:
        normalized = 256
    return normalized, (f"{name} rounded down to a multiple of 32" if normalized != numeric else None)


def normalize_tracks(value: object, width: int, height: int) -> list[list[dict[str, float]]]:
    """Normalize sparse control tracks for the official LTXV track editor."""
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ContractError("INVALID_TRACKS", "tracks must be valid JSON") from exc
    if isinstance(value, dict):
        value = value.get("tracks")
    if not isinstance(value, list):
        raise ContractError("INVALID_TRACKS", "tracks must be a list of point lists")
    if value and isinstance(value[0], dict):
        value = [value]
    if not value or len(value) > MAX_TRACKS:
        raise ContractError("INVALID_TRACKS", f"tracks must contain between 1 and {MAX_TRACKS} tracks")

    normalized: list[list[dict[str, float]]] = []
    for track_index, track in enumerate(value):
        if not isinstance(track, list) or not track or len(track) > MAX_TRACK_CONTROL_POINTS:
            raise ContractError(
                "INVALID_TRACKS",
                f"track {track_index} must contain between 1 and {MAX_TRACK_CONTROL_POINTS} points",
            )
        points: list[dict[str, float]] = []
        for point_index, point in enumerate(track):
            if not isinstance(point, dict) or "x" not in point or "y" not in point:
                raise ContractError("INVALID_TRACKS", f"track {track_index} point {point_index} must contain x and y")
            try:
                x = float(point["x"])
                y = float(point["y"])
            except (TypeError, ValueError) as exc:
                raise ContractError("INVALID_TRACKS", f"track {track_index} point {point_index} is not numeric") from exc
            if not math.isfinite(x) or not math.isfinite(y) or x < 0 or x > width or y < 0 or y > height:
                raise ContractError(
                    "INVALID_TRACKS",
                    f"track {track_index} point {point_index} is outside the normalized image bounds",
                    {"width": width, "height": height, "x": x, "y": y},
                )
            points.append({"x": round(x, 3), "y": round(y, 3)})
        normalized.append(points)
    return normalized

def normalize_request(data: dict) -> dict:
    mode = str(data.get("mode", "t2v")).lower()
    if mode not in ALL_MODES:
        raise ContractError("INVALID_MODE", f"unsupported mode: {mode}", {"supported": sorted(ALL_MODES)})
    if mode == "health":
        return {"mode": "health"}
    if mode == "pixel_upscale":
        raise ContractError(
            "WORKFLOW_NOT_VALIDATED",
            "Pixel Spatial Upscaler IC-LoRA is registered but the current official ComfyUI LTX-2.5 graph is not validated yet.",
            {"next_step": "validate an official Pixel IC-LoRA graph before enabling this mode"},
        )

    control = data.get("control")
    if control is None:
        control = {}
    if not isinstance(control, dict):
        raise ContractError("INVALID_CONTROL", "control must be an object when provided")

    prompt = data.get("prompt", "")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ContractError("INVALID_PROMPT", "prompt is required")
    negative = data.get("negative_prompt", "")
    if not isinstance(negative, str):
        raise ContractError("INVALID_PROMPT", "negative_prompt must be a string")

    width, width_note = clamp_dimension(data.get("width", 768), "width")
    height, height_note = clamp_dimension(data.get("height", 512), "height")
    try:
        fps = int(data.get("fps", 24))
    except (TypeError, ValueError) as exc:
        raise ContractError("INVALID_FPS", "fps must be an integer") from exc
    if fps < 8 or fps > 60:
        raise ContractError("INVALID_FPS", "fps must be between 8 and 60", {"fps": fps})

    frame_value = data.get("frames")
    duration_value = data.get("duration_seconds", 5)
    if frame_value is not None:
        try:
            requested_frames = int(frame_value)
        except (TypeError, ValueError) as exc:
            raise ContractError("INVALID_FRAME_COUNT", "frames must be an integer") from exc
    else:
        try:
            requested_frames = max(1, round(float(duration_value) * fps))
        except (TypeError, ValueError) as exc:
            raise ContractError("INVALID_DURATION", "duration_seconds must be numeric") from exc
    if requested_frames < 9 or requested_frames > 513:
        raise ContractError("INVALID_FRAME_COUNT", "frames must be between 9 and 513", {"frames": requested_frames})
    frames = 1 + 8 * round((requested_frames - 1) / 8)
    if frames < 9:
        frames = 9
    if frames > 513:
        frames = 513

    try:
        seed = int(data.get("seed", 42))
    except (TypeError, ValueError) as exc:
        raise ContractError("INVALID_SEED", "seed must be an integer") from exc
    if seed < 0 or seed > 2**63 - 1:
        raise ContractError("INVALID_SEED", "seed must be between 0 and 2^63-1")

    image_value = data.get("image_url") or data.get("image_base64")
    last_frame_value = data.get("last_frame_url") or data.get("last_frame_base64")
    video_value = data.get("video_url") or data.get("video_base64")
    if mode in {"i2v", "flf2v"} and not image_value:
        raise ContractError("MISSING_IMAGE", f"{mode} requires image_url or image_base64")
    if mode == "flf2v" and not last_frame_value:
        raise ContractError("MISSING_LAST_FRAME", "flf2v requires last_frame_url or last_frame_base64")

    control_type = str(data.get("control_type") or control.get("type") or ("depth" if mode == "union_control" else "motion_track")).lower()
    tracks: list[list[dict[str, float]]] | None = None
    if mode == "union_control":
        if control_type not in UNION_CONTROL_TYPES:
            raise ContractError(
                "UNSUPPORTED_CONTROL_TYPE",
                "the current official Union API graph exposes depth control only",
                {"requested": control_type, "supported": sorted(UNION_CONTROL_TYPES)},
            )
        if not video_value:
            raise ContractError("MISSING_VIDEO", "union_control requires video_url or video_base64")
    elif mode == "motion_track":
        if control_type not in {"motion_track", "sparse_tracks", "tracks"}:
            raise ContractError(
                "UNSUPPORTED_CONTROL_TYPE",
                "motion_track accepts sparse motion tracks only",
                {"requested": control_type, "supported": ["motion_track"]},
            )
        if not image_value:
            raise ContractError("MISSING_IMAGE", "motion_track requires image_url or image_base64")
        raw_tracks = data.get("tracks")
        if raw_tracks is None:
            raw_tracks = data.get("motion_tracks")
        if raw_tracks is None:
            raw_tracks = control.get("tracks")
        if raw_tracks is None:
            raise ContractError("MISSING_TRACKS", "motion_track requires tracks or motion_tracks")
        tracks = normalize_tracks(raw_tracks, width, height)

    loras = data.get("loras", [])
    if not isinstance(loras, list):
        raise ContractError("UNKNOWN_LORA", "loras must be a list")
    registry = json.loads(LORA_REGISTRY_PATH.read_text(encoding="utf-8"))["adapters"]
    normalized_loras = []
    for item in loras:
        logical_id = item.get("id") if isinstance(item, dict) else item
        if not isinstance(logical_id, str) or logical_id not in registry:
            raise ContractError("UNKNOWN_LORA", f"unknown LoRA logical id: {logical_id}")
        adapter = registry[logical_id]
        if not adapter.get("enabled") or mode not in adapter.get("workflow_compatibility", []):
            raise ContractError("LORA_NOT_COMPATIBLE_WITH_WORKFLOW", f"{logical_id} is not compatible with {mode}")
        strength = item.get("strength", adapter["default_strength"]) if isinstance(item, dict) else adapter["default_strength"]
        try:
            strength = float(strength)
        except (TypeError, ValueError) as exc:
            raise ContractError("INVALID_LORA_STRENGTH", f"invalid strength for {logical_id}") from exc
        if not adapter["minimum_strength"] <= strength <= adapter["maximum_strength"]:
            raise ContractError("INVALID_LORA_STRENGTH", f"strength out of range for {logical_id}")
        normalized_loras.append({"id": logical_id, "strength": strength, "filename": adapter["filename"]})

    control_strength_value = data.get("control_strength")
    if control_strength_value is None:
        control_strength_value = control.get("strength")
    if mode in CONTROL_MODES:
        expected_lora = CONTROL_LORA_BY_MODE[mode]
        if normalized_loras and (len(normalized_loras) != 1 or normalized_loras[0]["id"] != expected_lora):
            raise ContractError(
                "LORA_NOT_COMPATIBLE_WITH_WORKFLOW",
                f"{mode} only accepts the official {expected_lora} adapter",
                {"expected": expected_lora},
            )
        if not normalized_loras:
            adapter = registry[expected_lora]
            normalized_loras = [{
                "id": expected_lora,
                "strength": float(adapter["default_strength"]),
                "filename": adapter["filename"],
            }]
        if control_strength_value is not None:
            try:
                control_strength = float(control_strength_value)
            except (TypeError, ValueError) as exc:
                raise ContractError("INVALID_LORA_STRENGTH", "control_strength must be numeric") from exc
            adapter = registry[expected_lora]
            if not adapter["minimum_strength"] <= control_strength <= adapter["maximum_strength"]:
                raise ContractError("INVALID_LORA_STRENGTH", "control_strength is outside the official adapter range")
            normalized_loras[0]["strength"] = control_strength
        control_strength = normalized_loras[0]["strength"]
        if mode == "union_control" and not math.isclose(control_strength, 1.0):
            raise ContractError(
                "UNSUPPORTED_CONTROL_STRENGTH",
                "the current official Union loader does not expose a strength input; use control_strength 1.0",
                {"control_strength": control_strength},
            )
    else:
        control_strength = None

    profile = str(data.get("model_profile", "ltx25_bf16_control" if mode in CONTROL_MODES else "ltx25_bf16_core"))
    if profile not in MODEL_FILES:
        raise ContractError("INVALID_MODEL_PROFILE", f"unsupported model profile: {profile}")
    if mode in CONTROL_MODES and profile != "ltx25_bf16_control":
        raise ContractError("MODEL_PROFILE_REQUIRED", f"{mode} uses the official BF16 control profile")

    notes = [x for x in (width_note, height_note) if x]
    return {
        "mode": mode,
        "prompt": prompt,
        "negative_prompt": negative,
        "seed": seed,
        "width": width,
        "height": height,
        "frames": frames,
        "fps": fps,
        "duration_seconds": round((frames - 1) / fps, 4),
        "generate_audio": bool(data.get("generate_audio", True)),
        "enhance_prompt": bool(data.get("enhance_prompt", False if mode in CORE_MODES else True)),
        "upscale": bool(data.get("upscale", True)),
        "model_profile": profile,
        "control_type": control_type if mode in CONTROL_MODES else None,
        "control_strength": control_strength,
        "tracks": tracks,
        "loras": normalized_loras,
        "control": control,
        "input_keys": {
            "image_url": data.get("image_url"),
            "image_base64": data.get("image_base64"),
            "last_frame_url": data.get("last_frame_url"),
            "last_frame_base64": data.get("last_frame_base64"),
            "video_url": data.get("video_url"),
            "video_base64": data.get("video_base64"),
        },
        "adjustments": notes,
        "steps": "15-step HQ res2s + 4-step distilled refinement" if mode == "t2v" else ("8 distilled motion-track template" if mode == "motion_track" else "8+3 distilled two-stage template"),
        "guidance": {"video_cfg": 1.0, "audio_cfg": 1.0},
    }

def reject_private_url(url: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password:
        raise ContractError("INVALID_MEDIA_URL", "media URLs must be HTTPS URLs without embedded credentials")
    host = parsed.hostname.lower()
    if host in {"localhost", "metadata.google.internal", "169.254.169.254"}:
        raise ContractError("INVALID_MEDIA_URL", "local and cloud metadata hosts are not allowed")
    try:
        address = ipaddress.ip_address(host)
        if address.is_private or address.is_loopback or address.is_link_local or address.is_reserved:
            raise ContractError("INVALID_MEDIA_URL", "private and link-local media addresses are not allowed")
    except ValueError:
        pass


def extension_for(source: str, default: str) -> str:
    if source.startswith("data:"):
        mime = source.split(";", 1)[0].split(":", 1)[-1]
        return mimetypes.guess_extension(mime) or default
    suffix = Path(urlparse(source).path).suffix.lower()
    return suffix if re.fullmatch(r"\.[a-z0-9]{1,5}", suffix) else default


def write_media(value: str, kind: str, job_id: str) -> str:
    default_extension = ".mp4" if kind == "video" else ".png"
    extension = extension_for(value, default_extension)
    target = COMFY_INPUT_DIR / f"ltx25_{job_id}_{uuid.uuid4().hex}{extension}"
    COMFY_INPUT_DIR.mkdir(parents=True, exist_ok=True)
    if value.startswith("data:") or not value.startswith("http"):
        encoded = value.split(",", 1)[1] if "," in value else value
        try:
            payload = base64.b64decode(encoded, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise ContractError("INVALID_MEDIA", f"invalid {kind} base64 payload") from exc
        if not payload or len(payload) > MAX_INPUT_BYTES:
            raise ContractError("INVALID_MEDIA", f"{kind} payload exceeds the input limit")
        target.write_bytes(payload)
        return target.name
    reject_private_url(value)
    try:
        with requests.get(value, stream=True, timeout=(10, 120), headers={"User-Agent": "ltx25-runpod-worker/1"}) as response:
            response.raise_for_status()
            declared = response.headers.get("content-length")
            if declared and int(declared) > MAX_INPUT_BYTES:
                raise ContractError("INVALID_MEDIA", f"{kind} download exceeds the input limit")
            total = 0
            with target.open("wb") as output:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if not chunk:
                        continue
                    total += len(chunk)
                    if total > MAX_INPUT_BYTES:
                        raise ContractError("INVALID_MEDIA", f"{kind} download exceeds the input limit")
                    output.write(chunk)
    except ContractError:
        target.unlink(missing_ok=True)
        raise
    except requests.RequestException as exc:
        target.unlink(missing_ok=True)
        raise ContractError("MEDIA_DOWNLOAD_FAILED", f"could not download {kind} input: {exc}") from exc
    return target.name


def _run_ffmpeg(command: list[str], kind: str) -> None:
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=300, check=False)
    except FileNotFoundError as exc:
        raise ContractError("MEDIA_PREPROCESS_FAILED", "ffmpeg is not installed in the worker image") from exc
    except subprocess.SubprocessError as exc:
        raise ContractError("MEDIA_PREPROCESS_FAILED", f"could not preprocess {kind} input") from exc
    if result.returncode != 0:
        details = {"stderr": (result.stderr or "")[-1200:]}
        raise ContractError("MEDIA_PREPROCESS_FAILED", f"ffmpeg could not normalize {kind} input", details)


def _safe_job_stem(job_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]", "_", job_id)[:96] or uuid.uuid4().hex


def normalize_control_video(input_name: str, settings: dict, job_id: str) -> str:
    output_name = f"ltx25_{_safe_job_stem(job_id)}_{uuid.uuid4().hex}_control.mp4"
    input_path = COMFY_INPUT_DIR / input_name
    output_path = COMFY_INPUT_DIR / output_name
    vf = (
        f"scale={settings['width']}:{settings['height']}:force_original_aspect_ratio=decrease:flags=lanczos,"
        f"pad={settings['width']}:{settings['height']}:(ow-iw)/2:(oh-ih)/2:color=black,fps={settings['fps']}"
    )
    _run_ffmpeg(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(input_path),
            "-vf", vf, "-frames:v", str(settings["frames"]), "-an", "-c:v", "libx264",
            "-pix_fmt", "yuv420p", str(output_path),
        ],
        "control video",
    )
    if not output_path.is_file() or output_path.stat().st_size < 1024:
        raise ContractError("MEDIA_PREPROCESS_FAILED", "normalized control video was not created")
    return output_name


def normalize_control_image(input_name: str, settings: dict, job_id: str) -> str:
    output_name = f"ltx25_{_safe_job_stem(job_id)}_{uuid.uuid4().hex}_control.png"
    input_path = COMFY_INPUT_DIR / input_name
    output_path = COMFY_INPUT_DIR / output_name
    vf = (
        f"scale={settings['width']}:{settings['height']}:force_original_aspect_ratio=decrease:flags=lanczos,"
        f"pad={settings['width']}:{settings['height']}:(ow-iw)/2:(oh-ih)/2:color=black"
    )
    _run_ffmpeg(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(input_path),
            "-vf", vf, "-frames:v", "1", "-c:v", "png", str(output_path),
        ],
        "control image",
    )
    if not output_path.is_file() or output_path.stat().st_size < 1024:
        raise ContractError("MEDIA_PREPROCESS_FAILED", "normalized control image was not created")
    return output_name


NATIVE_LTX2_COMMIT = "a95ab856bf29407b6b066ede0abe1846050db56c"
NATIVE_PIPELINE_NAME = "TI2VidTwoStagesHQPipeline"
NATIVE_DISTILLED_LORA_STAGE_1 = 0.25
NATIVE_DISTILLED_LORA_STAGE_2 = 0.5
_NATIVE_PIPELINE = None
_NATIVE_PIPELINE_SIGNATURE: tuple[str, ...] | None = None
_NATIVE_PIPELINE_LOCK = threading.Lock()


def _native_model_root() -> Path:
    return Path(os.environ.get("RUNPOD_NETWORK_VOLUME_PATH", "/runpod-volume")) / "models"


def _native_file_record(path: Path, role: str) -> dict:
    return {
        "role": role,
        "filename": path.name,
        "path": str(path),
        "bytes": path.stat().st_size,
    }


def _native_resolve_models() -> tuple[dict[str, Path], dict]:
    manifest = json.loads(MODEL_MANIFEST_PATH.read_text(encoding="utf-8"))
    profile_name = "ltx25_bf16_core"
    profile = manifest.get("profiles", {}).get(profile_name)
    if not isinstance(profile, dict):
        raise ContractError("NATIVE_MODEL_MAPPING_MISSING", f"manifest profile is missing: {profile_name}")

    root = _native_model_root()

    def resolve_profile_file(filename: str) -> Path:
        entries = list(profile.get("required", [])) + list(profile.get("optional", []))
        entry = next((item for item in entries if Path(item.get("path", "")).name == filename), None)
        if entry is None:
            raise ContractError(
                "NATIVE_MODEL_MAPPING_MISSING",
                f"{filename} is not declared by {profile_name}",
                {"profile": profile_name, "filename": filename},
            )
        path = root / entry["directory"] / Path(entry["path"]).name
        if not path.is_file() or path.stat().st_size < 1024:
            raise ContractError(
                "NATIVE_MODEL_MISSING",
                f"required native model is not mapped: {filename}",
                {"role": filename, "path": str(path), "cache_repo": entry.get("cache_repo") or manifest.get("cache_repo")},
            )
        return path

    resolved = {
        "transformer": resolve_profile_file("ltx-2.5-22b-dev-transformer-bf16.safetensors"),
        "text_encoder": resolve_profile_file("gemma4-12b-with-proj-ltx-2.5-bf16.safetensors"),
        "video_vae": resolve_profile_file("ltx-2.5-video-vae-bf16.safetensors"),
        "audio_vae": resolve_profile_file("ltx-2.5-audio-vae-bf16.safetensors"),
        "spatial_upsampler": resolve_profile_file("ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors"),
        "root": root,
    }

    lora_entry = manifest.get("lora_repositories", {}).get("distilled_refinement")
    required_loras = profile.get("required_loras", [])
    if "distilled_refinement" not in required_loras or not isinstance(lora_entry, dict):
        raise ContractError(
            "NATIVE_MODEL_MAPPING_MISSING",
            "the native distilled refinement LoRA is not declared as required",
            {"profile": profile_name, "required_loras": required_loras},
        )
    lora_path = root / lora_entry["directory"] / Path(lora_entry["path"]).name
    if not lora_path.is_file() or lora_path.stat().st_size < 1024:
        raise ContractError(
            "NATIVE_MODEL_MISSING",
            f"required native refinement LoRA is not mapped: {lora_path.name}",
            {
                "role": "distilled_refinement",
                "path": str(lora_path),
                "cache_repo": lora_entry.get("cache_repo") or manifest.get("cache_repo"),
            },
        )
    resolved["distilled_lora"] = lora_path

    enhancer_path = None
    for entry in profile.get("optional", []):
        if Path(entry.get("path", "")).name == "gemma4_e2b_it_bf16.safetensors":
            candidate = root / entry["directory"] / Path(entry["path"]).name
            if candidate.is_file() and candidate.stat().st_size >= 1024:
                enhancer_path = candidate
            break
    resolved["prompt_enhancer"] = enhancer_path

    metadata_files = {
        key: _native_file_record(path, key)
        for key, path in resolved.items()
        if isinstance(path, Path) and key not in {"root", "prompt_enhancer"}
    }
    metadata_files["distilled_refinement_lora"] = _native_file_record(lora_path, "distilled_refinement_lora")
    if enhancer_path is not None:
        metadata_files["prompt_enhancer"] = _native_file_record(enhancer_path, "prompt_enhancer")
    metadata = {
        "profile": profile_name,
        "cache_repo": manifest.get("cache_repo"),
        "files": metadata_files,
        "prompt_enhancer_available": enhancer_path is not None,
    }
    return resolved, metadata


def _native_runtime_imports() -> dict:
    try:
        import torch
        from ltx_core.components.guiders import MultiModalGuiderParams
        from ltx_core.loader import LTXV_LORA_COMFY_RENAMING_MAP, LoraPathStrengthAndSDOps
        from ltx_core.model.video_vae import AUTO_TILING, get_video_chunks_number
        from ltx_pipelines.ti2vid_two_stages_hq import TI2VidTwoStagesHQPipeline
        from ltx_pipelines.utils.constants import DEFAULT_NEGATIVE_PROMPT, LTX_2_3_HQ_PARAMS, STAGE_2_DISTILLED_SIGMAS
        from ltx_pipelines.utils.args import ImageConditioningInput
        from ltx_pipelines.utils.media_io import encode_video
        from ltx_pipelines.utils.model_paths import ModelPaths
        from ltx_pipelines.utils.types import OffloadMode
    except Exception as exc:
        raise ContractError(
            "NATIVE_RUNTIME_UNAVAILABLE",
            "the official LTX native pipeline could not be imported",
            {"exception": type(exc).__name__, "message": str(exc)[:1000]},
        ) from exc
    return {
        "torch": torch,
        "pipeline": TI2VidTwoStagesHQPipeline,
        "guider_params": MultiModalGuiderParams,
        "lora": LoraPathStrengthAndSDOps,
        "lora_sd_ops": LTXV_LORA_COMFY_RENAMING_MAP,
        "auto_tiling": AUTO_TILING,
        "get_video_chunks_number": get_video_chunks_number,
        "model_paths": ModelPaths,
        "offload_mode": OffloadMode,
        "hq_params": LTX_2_3_HQ_PARAMS,
        "stage_2_sigmas": STAGE_2_DISTILLED_SIGMAS,
        "image_conditioning": ImageConditioningInput,
        "encode_video": encode_video,
        "default_negative_prompt": DEFAULT_NEGATIVE_PROMPT,
        "upstream_commit": NATIVE_LTX2_COMMIT,
    }


def _native_offload_mode(runtime: dict):
    raw = os.environ.get("LTX_NATIVE_OFFLOAD_MODE", "cpu").strip().lower()
    try:
        return runtime["offload_mode"](raw)
    except ValueError as exc:
        raise ContractError(
            "NATIVE_OFFLOAD_INVALID",
            f"unsupported native offload mode: {raw}",
            {"supported": [item.value for item in runtime["offload_mode"]]},
        ) from exc


def _native_pipeline_for(settings: dict, resolved: dict[str, Path], runtime: dict, offload_mode):
    global _NATIVE_PIPELINE, _NATIVE_PIPELINE_SIGNATURE
    enhancer_path = resolved.get("prompt_enhancer") if settings.get("enhance_prompt") else None
    if settings.get("enhance_prompt") and enhancer_path is None:
        raise ContractError(
            "NATIVE_PROMPT_ENHANCER_UNAVAILABLE",
            "prompt enhancement was requested but gemma4_e2b_it_bf16.safetensors is not mapped",
            {"acceptance_default": False, "capability": "available when the optional E2B model is present"},
        )
    external_loras = []
    for item in settings.get("loras", []):
        filename = item.get("filename")
        path = resolved["root"] / "loras" / filename
        if not path.is_file() or path.stat().st_size < 1024:
            raise ContractError(
                "NATIVE_LORA_MISSING",
                f"approved LoRA is not mapped for native execution: {filename}",
                {"path": str(path), "logical_id": item.get("id")},
            )
        external_loras.append(runtime["lora"](str(path), float(item["strength"]), runtime["lora_sd_ops"]))
    signature = tuple(
        [
            str(resolved["transformer"]),
            str(resolved["text_encoder"]),
            str(resolved["video_vae"]),
            str(resolved["audio_vae"]),
            str(resolved["spatial_upsampler"]),
            str(resolved["distilled_lora"]),
            str(enhancer_path or ""),
            offload_mode.value,
            *[f"{item.get('id')}:{item.get('strength')}" for item in settings.get("loras", [])],
        ]
    )
    with _NATIVE_PIPELINE_LOCK:
        if _NATIVE_PIPELINE is not None and _NATIVE_PIPELINE_SIGNATURE == signature:
            return _NATIVE_PIPELINE
        if _NATIVE_PIPELINE is not None:
            LOG.info("native_t2v: replacing cached pipeline because the model/offload signature changed")
            del _NATIVE_PIPELINE
            _NATIVE_PIPELINE = None
            _NATIVE_PIPELINE_SIGNATURE = None
            try:
                import gc
                gc.collect()
                runtime["torch"].cuda.empty_cache()
            except Exception:
                LOG.debug("native_t2v: cached pipeline cleanup was incomplete", exc_info=True)
        model_paths = runtime["model_paths"].from_split(
            transformer_path=str(resolved["transformer"]),
            text_encoder_path=str(resolved["text_encoder"]),
            video_vae_path=str(resolved["video_vae"]),
            audio_vae_path=str(resolved["audio_vae"]),
        )
        refinement_lora = runtime["lora"](
            str(resolved["distilled_lora"]),
            1.0,
            runtime["lora_sd_ops"],
        )
        LOG.info(
            "native_t2v: constructing %s with dev transformer=%s primary_gemma=%s refinement_lora=%s offload=%s external_loras=%d",
            NATIVE_PIPELINE_NAME,
            resolved["transformer"].name,
            resolved["text_encoder"].name,
            resolved["distilled_lora"].name,
            offload_mode.value,
            len(external_loras),
        )
        _NATIVE_PIPELINE = runtime["pipeline"](
            model_paths=model_paths,
            distilled_lora=[refinement_lora],
            distilled_lora_strength_stage_1=NATIVE_DISTILLED_LORA_STAGE_1,
            distilled_lora_strength_stage_2=NATIVE_DISTILLED_LORA_STAGE_2,
            spatial_upsampler_path=str(resolved["spatial_upsampler"]),
            loras=tuple(external_loras),
            device=runtime["torch"].device("cuda"),
            offload_mode=offload_mode,
            prompt_enhancer_gemma_root=str(enhancer_path) if enhancer_path is not None else None,
        )
        _NATIVE_PIPELINE_SIGNATURE = signature
        return _NATIVE_PIPELINE


def native_t2v_preflight() -> dict:
    try:
        runtime = _native_runtime_imports()
        resolved, metadata = _native_resolve_models()
        offload_mode = _native_offload_mode(runtime)
        return {
            "status": "PASS",
            "pipeline": NATIVE_PIPELINE_NAME,
            "upstream_commit": NATIVE_LTX2_COMMIT,
            "model_profile": metadata["profile"],
            "resolved_files": metadata["files"],
            "prompt_enhancer": {
                "available": metadata["prompt_enhancer_available"],
                "required_for_acceptance": False,
                "default_enabled": False,
            },
            "offload_mode": offload_mode.value,
            "cuda_available": bool(runtime["torch"].cuda.is_available()),
            "frame_rule": "8*k+1",
            "stage_1_steps": runtime["hq_params"].num_inference_steps,
            "stage_2_sigma_values": [float(value) for value in runtime["stage_2_sigmas"].tolist()],
            "distilled_lora_strength_stage_1": NATIVE_DISTILLED_LORA_STAGE_1,
            "distilled_lora_strength_stage_2": NATIVE_DISTILLED_LORA_STAGE_2,
            "old_comfy_workflow_used_for_t2v": False,
        }
    except ContractError as exc:
        return {
            "status": "FAIL",
            "error": {"code": exc.code, "message": exc.message, "details": exc.details},
            "pipeline": NATIVE_PIPELINE_NAME,
            "upstream_commit": NATIVE_LTX2_COMMIT,
        }
    except Exception as exc:
        LOG.exception("native_t2v preflight failed")
        return {
            "status": "FAIL",
            "error": {"code": "NATIVE_PREFLIGHT_FAILED", "message": str(exc)[:1000]},
            "pipeline": NATIVE_PIPELINE_NAME,
            "upstream_commit": NATIVE_LTX2_COMMIT,
        }


def run_native_t2v(settings: dict, job_id: str, image_path: Path | None = None) -> dict:
    runtime = _native_runtime_imports()
    resolved, metadata = _native_resolve_models()
    offload_mode = _native_offload_mode(runtime)
    if settings.get("enhance_prompt") and resolved.get("prompt_enhancer") is None:
        raise ContractError(
            "NATIVE_PROMPT_ENHANCER_UNAVAILABLE",
            "prompt enhancement was requested but gemma4_e2b_it_bf16.safetensors is not mapped",
            {"acceptance_default": False},
        )
    if not runtime["torch"].cuda.is_available():
        raise ContractError("NATIVE_CUDA_REQUIRED", "native LTX inference requires CUDA")
    sampler = GpuMemorySampler()
    sampler.start()
    output_path: Path | None = None
    response: dict | None = None
    native_started = time.perf_counter()
    try:
        pipeline = _native_pipeline_for(settings, resolved, runtime, offload_mode)
        params = runtime["hq_params"]
        negative_prompt = settings["negative_prompt"] or runtime["default_negative_prompt"]
        LOG.info(
            "native_t2v: starting official HQ generation profile=%s frames=%d size=%dx%d fps=%d audio=%s enhance=%s",
            metadata["profile"],
            settings["frames"],
            settings["width"],
            settings["height"],
            settings["fps"],
            settings["generate_audio"],
            settings["enhance_prompt"],
        )
        generation_started = time.perf_counter()
        images = []
        if image_path is not None:
            if not image_path.is_file():
                raise ContractError("MEDIA_NOT_FOUND", "prepared I2V image is missing")
            images = [
                runtime["image_conditioning"](
                    path=str(image_path),
                    frame_idx=0,
                    strength=1.0,
                )
            ]
        result = pipeline(
            prompt=settings["prompt"],
            negative_prompt=negative_prompt,
            seed=settings["seed"],
            height=settings["height"],
            width=settings["width"],
            frame_rate=settings["fps"],
            num_inference_steps=params.num_inference_steps,
            video_guider_params=params.video_guider_params,
            audio_guider_params=params.audio_guider_params,
            images=images,
            num_frames=settings["frames"],
            vae_dtype=runtime["torch"].bfloat16,
            tiling_config=runtime["auto_tiling"],
            enhance_prompt=bool(settings["enhance_prompt"]),
            enhance_static_cache=False,
            max_batch_size=1,
            stage_2_sigmas=runtime["stage_2_sigmas"],
            color_space=None,
            generated_keyframes=0,
        )
        generation_finished = time.perf_counter()
        output_path = Path("/tmp") / f"ltx-native-{re.sub(r'[^A-Za-z0-9_.-]+', '-', job_id)}.mp4"
        encoding_started = time.perf_counter()
        # PipelineOutput may contain lazy VAE decode iterators. Keep the
        # encoder and iterator consumption inside inference mode so PyTorch
        # does not treat inference tensors as autograd inputs.
        with runtime["torch"].inference_mode():
            runtime["encode_video"](
                video=result.video,
                fps=settings["fps"],
                audio=result.audio if settings["generate_audio"] else None,
                output_path=str(output_path),
                video_chunks_number=runtime["get_video_chunks_number"](result.num_frames, result.tiling_config),
                color_space=None,
            )
        encoding_finished = time.perf_counter()
        if not output_path.is_file() or output_path.stat().st_size < 1024:
            raise ContractError("NATIVE_OUTPUT_MISSING", "native pipeline completed without a playable MP4")
        payload = output_path.read_bytes()
        response = {
            "payload": payload,
            "descriptor": {
                "filename": output_path.name,
                "subfolder": "",
                "type": "output",
                "kind": "videos",
            },
            "native": {
                "pipeline": NATIVE_PIPELINE_NAME,
                "upstream_commit": NATIVE_LTX2_COMMIT,
                "model_profile": metadata["profile"],
                "resolved_files": metadata["files"],
                "offload_mode": offload_mode.value,
                "prompt_enhancement": {
                    "requested": bool(settings["enhance_prompt"]),
                    "used": bool(settings["enhance_prompt"]),
                    "available": metadata["prompt_enhancer_available"],
                },
                "negative_prompt_source": "request" if settings["negative_prompt"] else "official_default",
                "stage_1_steps": params.num_inference_steps,
                "stage_2_sigma_values": [float(value) for value in runtime["stage_2_sigmas"].tolist()],
                "distilled_lora_strength_stage_1": NATIVE_DISTILLED_LORA_STAGE_1,
                "distilled_lora_strength_stage_2": NATIVE_DISTILLED_LORA_STAGE_2,
                "num_frames": int(result.num_frames),
                "audio_encoded": bool(settings["generate_audio"]),
                "image_conditionings": len(images),
                "image_conditioning": (
                    {"frame_idx": 0, "strength": 1.0, "source": "request_image"}
                    if images
                    else None
                ),
                "comfy_graph_submitted": False,
            },
            "timing": {
                "native_pipeline_seconds": round(generation_finished - generation_started, 3),
                "native_encode_seconds": round(encoding_finished - encoding_started, 3),
                "native_total_seconds": round(encoding_finished - native_started, 3),
            },
        }
        return response
    finally:
        gpu_memory = sampler.stop()
        if response is not None:
            response["gpu_memory"] = gpu_memory
        if output_path is not None:
            output_path.unlink(missing_ok=True)


def load_workflow(mode: str) -> dict:
    path = WORKFLOW_DIR / WORKFLOWS[mode]
    if not path.is_file():
        raise ContractError("WORKFLOW_NOT_INSTALLED", f"workflow file is missing: {path.name}")
    return json.loads(path.read_text(encoding="utf-8"))


def nodes_of(workflow: dict, class_type: str) -> list[dict]:
    return [node for node in workflow.values() if isinstance(node, dict) and node.get("class_type") == class_type]


def node_source_id(node: dict) -> int | None:
    try:
        return int(node.get("_meta", {}).get("source_id"))
    except (TypeError, ValueError):
        return None

def patch_workflow(workflow: dict, settings: dict, media: dict[str, str], job_id: str) -> dict:
    workflow = copy.deepcopy(workflow)
    mode = settings["mode"]
    # The official graph has its two RandomNoise nodes at different sampler
    # stages; both must receive the requested seed for reproducibility.
    for node in nodes_of(workflow, "RandomNoise"):
        node.setdefault("inputs", {})["noise_seed"] = settings["seed"]
        node["inputs"]["control_after_generate"] = "fixed"

    prompt_nodes = nodes_of(workflow, "PrimitiveStringMultiline")
    positive_prompt_nodes = [node for node in prompt_nodes if node_source_id(node) == 5508]
    negative_prompt_nodes = [node for node in prompt_nodes if node_source_id(node) == 5509]
    if positive_prompt_nodes:
        positive_prompt_nodes[0].setdefault("inputs", {})["value"] = settings["prompt"]
    elif prompt_nodes:
        prompt_nodes[0].setdefault("inputs", {})["value"] = settings["prompt"]
    if negative_prompt_nodes:
        negative_prompt_nodes[0].setdefault("inputs", {})["value"] = settings["negative_prompt"]
    elif len(prompt_nodes) > 1:
        prompt_nodes[1].setdefault("inputs", {})["value"] = settings["negative_prompt"]

    if mode in CORE_MODES:
        boolean_nodes = nodes_of(workflow, "PrimitiveBoolean")
        if boolean_nodes:
            boolean_nodes[0].setdefault("inputs", {})["value"] = settings["enhance_prompt"]
        for node in nodes_of(workflow, "CLIPTextEncode"):
            if isinstance(node.get("inputs", {}).get("text"), str):
                node["inputs"]["text"] = settings["negative_prompt"]
    elif mode in CONTROL_MODES:
        for node in nodes_of(workflow, "GemmaAPITextEncode"):
            if node_source_id(node) == 5504:
                node.setdefault("inputs", {})["enhance_prompt"] = settings["enhance_prompt"]
        for node in nodes_of(workflow, "PrimitiveBoolean"):
            if node_source_id(node) == 5506:
                node.setdefault("inputs", {})["value"] = bool(media.get("image"))
    # ComfyUI 0.33 exposes the TextGenerate dynamic combo as one nested
    # sampling_mode object. Older official LTX graph exports store the same
    # values as flat inputs; normalize that export shape at request time.
    for node in nodes_of(workflow, "TextGenerateLTX2Prompt"):
        inputs = node.setdefault("inputs", {})
        sampling_mode = inputs.get("sampling_mode")
        if isinstance(sampling_mode, str):
            nested = {"sampling_mode": sampling_mode}
            for key in ("temperature", "top_k", "top_p", "min_p", "repetition_penalty", "seed", "presence_penalty"):
                if key in inputs:
                    nested[key] = inputs.pop(key)
            inputs["sampling_mode"] = nested
    # ComfyUI's LTXVSeparateAVLatent exposes video at output 0 and audio at
    # output 1. Some official API-format exports incorrectly connect output 0
    # to the second-stage audio input, which later reaches AudioPatchifier as a
    # 5-D video tensor. Repair that export shape at the API boundary.
    for node in nodes_of(workflow, "LTXVConcatAVLatent"):
        audio_link = node.setdefault("inputs", {}).get("audio_latent")
        if isinstance(audio_link, list) and len(audio_link) == 2:
            source = workflow.get(str(audio_link[0]))
            if isinstance(source, dict) and source.get("class_type") == "LTXVSeparateAVLatent" and audio_link[1] == 0:
                audio_link[1] = 1

    for node in nodes_of(workflow, "LTXVAudioVAEDecode"):
        sample_link = node.setdefault("inputs", {}).get("samples")
        if isinstance(sample_link, list) and len(sample_link) == 2:
            source = workflow.get(str(sample_link[0]))
            if isinstance(source, dict) and source.get("class_type") == "LTXVSeparateAVLatent" and sample_link[1] == 0:
                sample_link[1] = 1

    # Current ComfyUI requires these inputs even though the released LTX
    # API-format I2V export omits them. FLF2V uses a different DynamicCombo
    # branch because its two reference frames must be resized to exact
    # dimensions rather than by the longer side.
    for node in nodes_of(workflow, "ResizeImageMaskNode"):
        inputs = node.setdefault("inputs", {})
        # DynamicCombo values are flattened by the /prompt API. Source
        # exports can retain fields for a different branch (for example
        # resize_type.multiple) and the current node then receives that
        # stale field as an unexpected execute() keyword. Remove all branch
        # fields before selecting the active option.
        for key in list(inputs):
            if key.startswith("resize_type."):
                inputs.pop(key, None)
        if mode == "flf2v":
            inputs["resize_type"] = "scale dimensions"
            inputs["resize_type.width"] = settings["width"]
            inputs["resize_type.height"] = settings["height"]
            inputs["resize_type.crop"] = "center"
            inputs["scale_method"] = "nearest-exact"
        else:
            inputs["resize_type"] = "scale longer dimension"
            inputs["resize_type.longer_size"] = max(settings["width"], settings["height"])
            inputs["scale_method"] = "lanczos"
    for node in nodes_of(workflow, "LTXVPreprocess"):
        node.setdefault("inputs", {}).setdefault("img_compression", 18)
    for index, node in enumerate(nodes_of(workflow, "LTXVImgToVideoInplace")):
        node.setdefault("inputs", {}).setdefault("strength", 0.7 if index == 0 else 1.0)

    if mode == "flf2v":
        # The official FLF2V graph was exported from a UI graph whose output
        # slots were dropped by an older API conversion. Restore the current
        # ComfyUI multi-output ordering at the boundary instead of relying on
        # node IDs in callers:
        #   CONDITIONING positive=0, negative=1
        #   LATENT from AddGuide/CropGuides=2
        def repair_link(node: dict, input_name: str, output_slot: int, source_types: set[str] | None = None) -> None:
            link = node.setdefault("inputs", {}).get(input_name)
            if not isinstance(link, list) or len(link) != 2:
                return
            source = workflow.get(str(link[0]))
            if not isinstance(source, dict):
                return
            if source_types is None or source.get("class_type") in source_types:
                link[1] = output_slot

        for index, node in enumerate(nodes_of(workflow, "LTXVAddGuide")):
            inputs = node.setdefault("inputs", {})
            inputs.setdefault("frame_idx", 0 if index == 0 else -1)
            inputs.setdefault("strength", 0.7)
            repair_link(node, "negative", 1, {"LTXVConditioning", "LTXVAddGuide"})
            repair_link(node, "latent", 2, {"LTXVAddGuide"})

        for node in nodes_of(workflow, "LTXVCropGuides"):
            repair_link(node, "negative", 1, {"LTXVAddGuide"})

        for node in nodes_of(workflow, "LTXVDualCFGGuider"):
            repair_link(node, "negative", 1, {"LTXVAddGuide"})

        for node in nodes_of(workflow, "LTXVConcatAVLatent"):
            repair_link(node, "video_latent", 2, {"LTXVAddGuide"})

        for node in nodes_of(workflow, "VAEDecodeTiled"):
            repair_link(node, "samples", 2, {"LTXVCropGuides"})

        for node in nodes_of(workflow, "SamplerEulerAncestral"):
            inputs = node.setdefault("inputs", {})
            inputs.setdefault("eta", 1.0)
            inputs.setdefault("s_noise", 1.0)

    # Override latent dimensions/frame count directly. This intentionally
    # bypasses UI-only ResolutionSelector/math nodes while retaining the
    # upstream conditioning graph.
    for node in nodes_of(workflow, "EmptyLTXVLatentVideo"):
        inputs = node.setdefault("inputs", {})
        inputs.update(width=settings["width"], height=settings["height"], length=settings["frames"])
    for node in nodes_of(workflow, "LTXVEmptyLatentAudio"):
        inputs = node.setdefault("inputs", {})
        inputs.update(frames_number=settings["frames"], frame_rate=settings["fps"])
    for node in nodes_of(workflow, "LTXVConditioning"):
        node.setdefault("inputs", {})["frame_rate"] = settings["fps"]
    for node in nodes_of(workflow, "CreateVideo"):
        inputs = node.setdefault("inputs", {})
        inputs["fps"] = settings["fps"]
        if not settings["generate_audio"]:
            inputs.pop("audio", None)
    for node in nodes_of(workflow, "LTXVImgToVideo"):
        inputs = node.setdefault("inputs", {})
        for key, value in (("width", settings["width"]), ("height", settings["height"]), ("length", settings["frames"])):
            if key in inputs:
                inputs[key] = value

    if mode in CORE_MODES | CONTROL_MODES:
        model_files = MODEL_FILES[settings["model_profile"]]
        for node in nodes_of(workflow, "UNETLoader"):
            node.setdefault("inputs", {})["unet_name"] = model_files["unet"]
        for node in nodes_of(workflow, "CLIPLoader"):
            inputs = node.setdefault("inputs", {})
            source_id = node_source_id(node)
            name = str(inputs.get("clip_name", ""))
            if source_id in {5572, 5605, 387, 228} or "12b" in name or "proj" in name:
                inputs["clip_name"] = model_files["clip"]
            elif source_id in {5545, 5604, 393, 246}:
                inputs["clip_name"] = "gemma4_e2b_it_bf16.safetensors"
        for node in nodes_of(workflow, "GemmaAPITextEncode"):
            node.setdefault("inputs", {})["ckpt_name"] = GEMMA_API_CHECKPOINT

    if mode == "union_control":
        for node in nodes_of(workflow, "LoadVideoDepthAnythingModel"):
            node.setdefault("inputs", {})["model"] = "video_depth_anything_vits.pth"
    if mode == "motion_track":
        for node in nodes_of(workflow, "PrimitiveFloat"):
            source_id = node_source_id(node)
            if source_id == 9007:
                node.setdefault("inputs", {})["value"] = settings["fps"]
            elif source_id == 9008:
                node.setdefault("inputs", {})["value"] = (settings["frames"] - 1) / settings["fps"]
        for node in nodes_of(workflow, "LTXVSparseTrackEditor"):
            node.setdefault("inputs", {})["points_store"] = json.dumps(settings["tracks"], separators=(",", ":"))
            node["inputs"]["coordinates"] = "[]"
            node["inputs"]["points_to_sample"] = settings["frames"]

    # The official T2V subgraph's ResolutionSelector is a UI-only node and is
    # absent from the API-format export, leaving two dangling `top_409` links.
    # Its outputs are consumed after division by two, so provide the selected
    # full-resolution dimensions here to preserve the API's latent-size
    # contract. The same subgraph also derives frame count and output FPS from
    # stale duration/FPS widget values unless these four primitive inputs are
    # replaced at request time.
    if mode == "t2v":
        t2v_primitive_values = {
            360: settings["height"] * 2,
            372: settings["width"] * 2,
            361: settings["fps"],
            362: settings["duration_seconds"],
        }
        for node in nodes_of(workflow, "PrimitiveInt"):
            source_id = node_source_id(node)
            if source_id in t2v_primitive_values:
                node.setdefault("inputs", {})["value"] = t2v_primitive_values[source_id]
                node["inputs"]["control_after_generate"] = "fixed"
    image_loaders = nodes_of(workflow, "LoadImage")
    if media.get("image"):
        if len(image_loaders) == 1:
            image_loaders[0].setdefault("inputs", {})["image"] = media["image"]
        elif image_loaders:
            ordered = sorted(image_loaders, key=lambda item: int(item.get("_meta", {}).get("source_id", 0)))
            ordered[0].setdefault("inputs", {})["image"] = media["image"]
            if media.get("last_frame") and len(ordered) > 1:
                ordered[1].setdefault("inputs", {})["image"] = media["last_frame"]
    for node in nodes_of(workflow, "LoadVideo"):
        if media.get("video"):
            inputs = node.setdefault("inputs", {})
            # ComfyUI 0.33 renamed the required LoadVideo /prompt field from
            # the older UI-export name video to file.
            inputs.pop("video", None)
            inputs["file"] = media["video"]

    # Control graph loaders are only patched when the official source graph
    # exposes a file socket. Missing/extra custom nodes are surfaced by ComfyUI
    # validation rather than silently loading an arbitrary caller filename.
    for node in workflow.values():
        if not isinstance(node, dict):
            continue
        inputs = node.get("inputs", {})
        if "lora_name" in inputs and settings["loras"]:
            inputs["lora_name"] = settings["loras"][0]["filename"]
        if node.get("class_type") == "LTXICLoRALoaderModelOnly" and settings["loras"]:
            inputs["lora_name"] = settings["loras"][0]["filename"]
            # Current ComfyUI requires this socket even when the official
            # Union export omitted it from the older UI graph.
            inputs["strength_model"] = settings["control_strength"]

    for node in nodes_of(workflow, "SaveVideo"):
        suffix = "_tracks" if node_source_id(node) == 9006 else ""
        node.setdefault("inputs", {})["filename_prefix"] = f"ltx25/{job_id}{suffix}"

    # _meta is useful in checked-in API artifacts but is not needed by /prompt.
    for node in workflow.values():
        if isinstance(node, dict):
            node.pop("_meta", None)
    return workflow


def comfy_reachable() -> bool:
    try:
        response = requests.get(f"{COMFY_BASE}/", timeout=5)
        return response.status_code == 200
    except requests.RequestException:
        return False


def object_info() -> tuple[dict | None, str | None]:
    try:
        response = requests.get(f"{COMFY_BASE}/object_info", timeout=30)
        response.raise_for_status()
        payload = response.json()
        return (payload if isinstance(payload, dict) else None), None
    except (requests.RequestException, ValueError) as exc:
        return None, str(exc)[:500]


def preflight_result() -> dict:
    from scripts import validate_workflows

    manifest = json.loads(MODEL_MANIFEST_PATH.read_text(encoding="utf-8"))
    schema, schema_error = object_info()
    workflow_results = []
    for mode, filename in WORKFLOWS.items():
        path = WORKFLOW_DIR / filename
        if not path.is_file():
            workflow_results.append({"mode": mode, "workflow": filename, "errors": ["workflow file is missing"]})
            continue
        workflow = json.loads(path.read_text(encoding="utf-8"))
        errors = validate_workflows.validate_graph(
            workflow,
            object_info=schema,
            manifest=manifest,
            require_media_placeholders=False,
        )
        workflow_results.append({"mode": mode, "workflow": filename, "errors": errors})
    active_profile = os.environ.get("LTX_BOOTSTRAP_PROFILE", "ltx25_bf16_core")
    storage = model_storage_audit(active_profile)
    native = native_t2v_preflight()
    errors = [error for result in workflow_results for error in result["errors"]]
    errors.extend(storage["missing"])
    if native["status"] != "PASS":
        errors.append(f"native_t2v preflight failed: {native.get('error', native)}")
    if schema_error:
        errors.append(f"object_info unavailable: {schema_error}")
    return {
        "status": "PASS" if not errors else "FAIL",
        "comfyui_reachable": comfy_reachable(),
        "object_info": {"available": schema is not None, "error": schema_error},
        "workflows": workflow_results,
        "native_t2v": native,
        "model_storage": storage,
        "errors": errors,
        "checks": [
            "required inputs",
            "dynamic references and output slots",
            "object_info enum/list selections",
            "manifest model filenames",
            "mapped model files on disk",
            "native LTX package import",
            "native full/dev model resolution",
            "native distilled refinement LoRA resolution",
            "native T2V bypasses the ComfyUI distilled graph",
        ],
    }


def gpu_info() -> dict:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,memory.used", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        rows = []
        for line in result.stdout.strip().splitlines():
            bits = [x.strip() for x in line.split(",")]
            if len(bits) == 3:
                rows.append({"name": bits[0], "vram_mb": int(float(bits[1])), "used_mb": int(float(bits[2]))})
        return {"cuda_available": bool(rows), "gpus": rows}
    except (OSError, ValueError, subprocess.SubprocessError):
        return {"cuda_available": False, "gpus": []}


def hf_access_probe() -> dict:
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    if not token:
        return {"configured": False, "status_code": None, "access": "missing"}
    probe_path = "diffusion_models/ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors"
    url = f"https://huggingface.co/Lightricks/LTX-2.5/resolve/main/{probe_path}"
    try:
        response = requests.get(
            url,
            headers={"Authorization": f"Bearer {token}"},
            allow_redirects=True,
            stream=True,
            timeout=20,
        )
        try:
            status_code = response.status_code
            body = ""
            if status_code >= 400:
                body = response.text[:500]
                body = re.sub(r"hf_[A-Za-z0-9_-]{10,}", "[REDACTED_HF_TOKEN]", body)
                body = re.sub(r"Bearer\s+\S+", "Bearer [REDACTED]", body, flags=re.IGNORECASE)
            return {
                "configured": True,
                "status_code": status_code,
                "access": "granted" if status_code == 200 else "denied_or_unavailable",
                "probe": "Lightricks/LTX-2.5 gated transformer file",
                "body_excerpt": body,
            }
        finally:
            response.close()
    except requests.RequestException as exc:
        return {
            "configured": True,
            "status_code": None,
            "access": "probe_failed",
            "probe": "Lightricks/LTX-2.5 gated transformer file",
            "body_excerpt": str(exc)[:500],
        }

def model_storage_audit(profile: str | None = None) -> dict:
    manifest = json.loads(MODEL_MANIFEST_PATH.read_text(encoding="utf-8"))
    root = Path(os.environ.get("RUNPOD_NETWORK_VOLUME_PATH", "/runpod-volume")) / "models"
    records: dict[str, dict] = {}

    def add_entry(entry: dict, category: str, role: str) -> None:
        target = root / entry["directory"] / Path(entry["path"]).name
        key = str(target)
        record = records.setdefault(
            key,
            {
                "path": str(target.relative_to(root)),
                "bytes": 0,
                "installed": False,
                "roles": [],
            },
        )
        record["roles"].append({"category": category, "role": role, "repo": entry["repo"]})
        if target.is_file():
            record["bytes"] = target.stat().st_size
            record["installed"] = record["bytes"] >= 1024

    profiles = {profile: manifest["profiles"][profile]} if profile else manifest["profiles"]
    for profile_name, profile_data in profiles.items():
        for entry in profile_data.get("required", []):
            add_entry(entry, "profile", f"{profile_name}:required")
        for entry in profile_data.get("optional", []):
            add_entry(entry, "profile", f"{profile_name}:optional")
    for logical_id, entry in manifest.get("lora_repositories", {}).items():
        add_entry(entry, "lora", logical_id)

    files = sorted(records.values(), key=lambda item: item["bytes"], reverse=True)
    by_category: dict[str, int] = {}
    for record in files:
        category = record["roles"][0]["category"]
        by_category[category] = by_category.get(category, 0) + record["bytes"]
    return {
        "root": str(root),
        "exists": root.exists(),
        "approved_file_count": len(files),
        "installed_file_count": sum(1 for item in files if item["installed"]),
        "approved_file_bytes": sum(item["bytes"] for item in files),
        "by_category_bytes": by_category,
        "missing": [item["path"] for item in files if not item["installed"]],
        "files": files,
    }

def health_result() -> dict:
    manifest = json.loads(MODEL_MANIFEST_PATH.read_text(encoding="utf-8"))
    volume = Path(os.environ.get("RUNPOD_NETWORK_VOLUME_PATH", "/runpod-volume")) / "models"
    installed = {}
    missing = {}
    for profile, data in manifest["profiles"].items():
        paths = [volume / entry["directory"] / Path(entry["path"]).name for entry in data.get("required", [])]
        installed[profile] = all(path.is_file() and path.stat().st_size >= 1024 for path in paths)
        missing[profile] = [str(path) for path in paths if not path.is_file()]
    return {
        "status": "READY" if comfy_reachable() else "NOT_READY",
        "worker_ready": True,
        "comfyui_reachable": comfy_reachable(),
        "cuda": gpu_info(),
        "installed_profiles": installed,
        "missing_model_files": missing,
        "workflow_modes": sorted(WORKFLOWS),
        "registered_loras": sorted(json.loads(LORA_REGISTRY_PATH.read_text(encoding="utf-8"))["adapters"]),
        "pixel_upscale": "registered_not_validated",
        "huggingface_access": hf_access_probe(),
        "model_storage": model_storage_audit(),
        "native_t2v": native_t2v_preflight(),
    }


def queue_workflow(workflow: dict, client_id: str) -> str:
    try:
        response = requests.post(
            f"{COMFY_BASE}/prompt",
            json={"prompt": workflow, "client_id": client_id},
            timeout=30,
        )
    except requests.RequestException as exc:
        raise ContractError("COMFYUI_STARTUP_FAILED", f"could not queue workflow: {exc}") from exc
    if response.status_code >= 400:
        try:
            details = response.json()
        except ValueError:
            details = {"raw": response.text[:2000]}
        raise ContractError("COMFYUI_EXECUTION_FAILED", "ComfyUI rejected the workflow", details)
    try:
        prompt_id = response.json().get("prompt_id")
    except ValueError as exc:
        raise ContractError("COMFYUI_EXECUTION_FAILED", "ComfyUI returned a non-JSON queue response") from exc
    if not prompt_id:
        raise ContractError("COMFYUI_EXECUTION_FAILED", "ComfyUI queue response did not include prompt_id")
    return prompt_id


def wait_for_history(prompt_id: str) -> dict:
    deadline = time.monotonic() + COMFY_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        try:
            response = requests.get(f"{COMFY_BASE}/history/{prompt_id}", timeout=30)
            response.raise_for_status()
            history = response.json()
        except requests.RequestException as exc:
            raise ContractError("COMFYUI_EXECUTION_FAILED", f"could not read ComfyUI history: {exc}") from exc
        entry = history.get(prompt_id)
        if entry:
            status = entry.get("status", {})
            if status.get("status_str") in {"error", "failed"} or status.get("completed") is False and status.get("messages"):
                raise ContractError("COMFYUI_EXECUTION_FAILED", "ComfyUI execution failed", {"status": status})
            if status.get("completed") or entry.get("outputs"):
                return entry
        time.sleep(2)
    raise ContractError("COMFYUI_EXECUTION_FAILED", f"ComfyUI job timed out after {COMFY_TIMEOUT_SECONDS} seconds")


def output_descriptors(history: dict) -> list[dict]:
    found = []
    for node_output in (history.get("outputs") or {}).values():
        for key in ("videos", "gifs", "images"):
            for item in node_output.get(key, []) or []:
                filename = item.get("filename")
                if filename and (key in {"videos", "gifs"} or Path(filename).suffix.lower() in {".mp4", ".webm", ".mov"}):
                    found.append({"filename": filename, "subfolder": item.get("subfolder", ""), "type": item.get("type", "output"), "kind": key})
    if not found:
        raise ContractError("OUTPUT_NOT_FOUND", "ComfyUI completed without an MP4/video output")
    return found


def fetch_output(descriptor: dict) -> bytes:
    try:
        response = requests.get(
            f"{COMFY_BASE}/view",
            params={"filename": descriptor["filename"], "subfolder": descriptor["subfolder"], "type": descriptor["type"]},
            timeout=180,
        )
        response.raise_for_status()
        return response.content
    except requests.RequestException as exc:
        raise ContractError("OUTPUT_NOT_FOUND", f"could not fetch generated MP4: {exc}") from exc


def publish_output(job_id: str, descriptor: dict, payload: bytes) -> dict:
    if os.environ.get("BUCKET_ENDPOINT_URL") and rp_upload is not None:
        temporary = Path("/tmp") / f"{job_id}_{Path(descriptor['filename']).name}"
        temporary.write_bytes(payload)
        try:
            url = rp_upload.upload_image(job_id, str(temporary))
            return {"url": url, "filename": Path(descriptor["filename"]).name, "mime_type": "video/mp4", "type": "s3_url"}
        except Exception as exc:
            raise ContractError("OUTPUT_UPLOAD_FAILED", f"S3-compatible output upload failed: {exc}") from exc
        finally:
            temporary.unlink(missing_ok=True)
    if len(payload) > MAX_BASE64_OUTPUT_BYTES:
        raise ContractError(
            "OUTPUT_UPLOAD_REQUIRED",
            "generated MP4 is too large for inline RunPod output; configure BUCKET_ENDPOINT_URL and S3 credentials",
            {"bytes": len(payload), "limit": MAX_BASE64_OUTPUT_BYTES},
        )
    return {
        "filename": Path(descriptor["filename"]).name,
        "mime_type": "video/mp4",
        "type": "base64",
        "data": base64.b64encode(payload).decode("ascii"),
    }


def handler(job: dict) -> dict:
    job_id = str(job.get("id", uuid.uuid4().hex))
    started = time.perf_counter()
    sampler: GpuMemorySampler | None = None
    try:
        data = as_input(job)
        if str(data.get("mode", "")).lower() == "health" or data.get("action") == "health":
            return health_result()
        if str(data.get("mode", "")).lower() == "preflight" or data.get("action") == "preflight":
            return preflight_result()
        settings = normalize_request(data)

        # Core T2V and I2V both use the official native full/dev pipeline.
        # I2V passes the downloaded image as the native frame-0 conditioning
        # input; it must not fall through to the stale ComfyUI graph.
        if settings["mode"] in {"t2v", "i2v"}:
            native_settings = copy.deepcopy(settings)
            keys = native_settings.pop("input_keys")
            native_image_path: Path | None = None
            native_media_files: list[str] = []
            try:
                if settings["mode"] == "i2v":
                    image_source = keys["image_base64"] or keys["image_url"]
                    image_name = write_media(image_source, "image", job_id)
                    native_media_files.append(image_name)
                    native_image_path = COMFY_INPUT_DIR / image_name
                native_result = run_native_t2v(native_settings, job_id, native_image_path)
                publish_started = time.perf_counter()
                output = publish_output(job_id, native_result["descriptor"], native_result["payload"])
                finished = time.perf_counter()
                timing = dict(native_result["timing"])
                timing["output_publish_seconds"] = round(finished - publish_started, 3)
                timing["handler_total_seconds"] = round(finished - started, 3)
                timing["gpu_memory"] = native_result["gpu_memory"]
                return {
                    "status": "COMPLETED",
                    "job_id": job_id,
                    "mode": native_settings["mode"],
                    "workflow": f"native:{NATIVE_PIPELINE_NAME}",
                    "model_profile": native_settings["model_profile"],
                    "normalized": native_settings,
                    "native": native_result["native"],
                    "timing": timing,
                    "output_size_bytes": len(native_result["payload"]),
                    "output": output,
                    "video": output,
                    "inference_ms": round(native_result["timing"]["native_pipeline_seconds"] * 1000),
                }
            finally:
                for filename in native_media_files:
                    (COMFY_INPUT_DIR / filename).unlink(missing_ok=True)

        input_prep_started = time.perf_counter()
        media = {}
        media_files: list[str] = []
        keys = settings.pop("input_keys")
        has_image_input = bool(keys["image_base64"] or keys["image_url"])
        if settings["mode"] in {"i2v", "flf2v", "motion_track"} or (settings["mode"] == "union_control" and has_image_input):
            image_name = write_media(keys["image_base64"] or keys["image_url"], "image", job_id)
            media_files.append(image_name)
            media["image"] = normalize_control_image(image_name, settings, job_id) if settings["mode"] == "motion_track" else image_name
            if media["image"] != image_name:
                media_files.append(media["image"])
        if settings["mode"] == "flf2v":
            media["last_frame"] = write_media(keys["last_frame_base64"] or keys["last_frame_url"], "image", job_id)
            media_files.append(media["last_frame"])
        if settings["mode"] == "union_control":
            video_name = write_media(keys["video_base64"] or keys["video_url"], "video", job_id)
            media_files.append(video_name)
            media["video"] = normalize_control_video(video_name, settings, job_id)
            media_files.append(media["video"])
            if "image" not in media:
                # The official Union graph still evaluates its optional image
                # encoder even when the UI toggle is off. Derive a valid
                # opening frame from the normalized control video so the API
                # path never hands LoadImage an empty filename.
                media["image"] = normalize_control_image(media["video"], settings, job_id)
                media_files.append(media["image"])
        input_prep_finished = time.perf_counter()

        if not comfy_reachable():
            raise ContractError("COMFYUI_STARTUP_FAILED", f"ComfyUI is not reachable at {COMFY_BASE}")

        workflow_patch_started = time.perf_counter()
        workflow = patch_workflow(load_workflow(settings["mode"]), settings, media, job_id)
        workflow_patch_finished = time.perf_counter()

        sampler = GpuMemorySampler()
        sampler.start()
        queue_started = time.perf_counter()
        prompt_id = queue_workflow(workflow, str(uuid.uuid4()))
        queue_finished = time.perf_counter()
        execution_started = queue_finished
        history = wait_for_history(prompt_id)
        execution_finished = time.perf_counter()

        descriptors = output_descriptors(history)
        descriptor = next((item for item in descriptors if "_tracks" not in Path(item["filename"]).stem.lower()), descriptors[0])
        output_fetch_started = time.perf_counter()
        payload = fetch_output(descriptor)
        output_fetch_finished = time.perf_counter()
        output_publish_started = output_fetch_finished
        output = publish_output(job_id, descriptor, payload)
        output_publish_finished = time.perf_counter()

        finished = output_publish_finished
        gpu_memory = sampler.stop()
        return {
            "status": "COMPLETED",
            "job_id": job_id,
            "prompt_id": prompt_id,
            "mode": settings["mode"],
            "workflow": WORKFLOWS[settings["mode"]],
            "model_profile": settings["model_profile"],
            "normalized": settings,
            "timing": {
                "input_prep_seconds": round(input_prep_finished - input_prep_started, 3),
                "workflow_patch_seconds": round(workflow_patch_finished - workflow_patch_started, 3),
                "comfy_queue_seconds": round(queue_finished - queue_started, 3),
                "comfy_execution_seconds": round(execution_finished - execution_started, 3),
                "generation_seconds": round(execution_finished - execution_started, 3),
                "output_fetch_seconds": round(output_fetch_finished - output_fetch_started, 3),
                "output_publish_seconds": round(output_publish_finished - output_publish_started, 3),
                "queue_to_output_seconds": round(finished - queue_started, 3),
                "handler_total_seconds": round(finished - started, 3),
                "gpu_memory": gpu_memory,
            },
            "output_size_bytes": len(payload),
            "output": output,
            "video": output,
            "inference_ms": round((execution_finished - execution_started) * 1000),
        }
    except ContractError as exc:
        return error_result(exc)
    except Exception as exc:  # keep the RunPod response structured without leaking a traceback
        LOG.exception("unhandled LTX-2.5 worker error")
        return error_result(ContractError("INTERNAL_ERROR", str(exc)))
    finally:
        if sampler is not None:
            sampler.stop()
        for filename in locals().get("media_files", []):
            (COMFY_INPUT_DIR / filename).unlink(missing_ok=True)

if __name__ == "__main__":
    if runpod is None:
        raise SystemExit("runpod package is required to run the serverless handler")
    runpod.serverless.start({"handler": handler})

