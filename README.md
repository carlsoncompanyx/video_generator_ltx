# LTX-2.5 ComfyUI RunPod Serverless worker

This directory is an isolated RunPod Serverless worker for the current LTX-2.5 ComfyUI workflows. It does not modify the existing VACE/SkyReels endpoint or application production paths.

## Validation pins

Validated upstream on 2026-08-15:

- ComfyUI: `0f1fa67ad8a68b62c65ebc97a7bf485df2459c3a` (`0.33.0`)
- LTX-2 upstream: `fd4ded7f2d88d3da713abcdd4ad41ecc4a9314ca`
- ComfyUI-LTXVideo installed/pinned: `ac4d99839020b983e956a8ab67ec38aec1b6e65a`
- ComfyUI-Video-Depth-Anything installed/pinned: `a0db08e63d1ea571601c45cde4aaee0acdd0544d`
- runpod-workers/worker-comfyui inspected: tag `5.8.7`, commit `a1981e99b1f5a7201f387653420ad1f275b97d0a`
- Python: `3.12`
- CUDA image: `nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04`
- Torch: `2.11.0+cu128`; torchvision `0.26.0+cu128`; torchaudio `2.11.0+cu128`
- RunPod package: `1.7.13` (within the `1.7.12` worker requirement)

The image installs the pinned ComfyUI-LTXVideo custom node and the pinned Video Depth Anything custom node. The Union API graph intentionally keeps only the official depth branch; its disconnected Canny and DWPose UI alternatives are not advertised by this endpoint.

## Model storage

The intended network volume is `ltx25-comfyui-models-20260815` in `US-CA-2`, HIGH_PERFORMANCE, 160 GB. It mounts at `/runpod-volume` and stores models under `/runpod-volume/models`.

The approved files and their exact target directories are in `model_manifest.json`. The normal profile is `ltx25_int8_distilled`. Control workflows use the separate BF16 profile because the official Union Control and Motion Track examples use the BF16 distilled transformer.

Set `MODEL_BOOTSTRAP_DOWNLOAD=true` only on a worker with an accepted, read-enabled `HF_TOKEN`. The LTX-2.5 Hugging Face repositories are gated by the LTX-2 community license. Bootstrap is idempotent, only downloads files in the manifest, never prints the token, and fails with `HF_MODEL_ACCESS_REQUIRED` when access is not granted.

```bash
python scripts/bootstrap_models.py --profile ltx25_int8_distilled --verify-only
python scripts/bootstrap_models.py --profile ltx25_int8_distilled --download
python scripts/bootstrap_models.py --profile ltx25_bf16_control --download --verify-sha256
```

## Build and deploy

The image uses a direct pinned ComfyUI install because the published `runpod/worker-comfyui:5.8.7-base` tag was not available during validation. The worker handler follows the current worker-comfyui `/prompt`, `/history`, `/view`, and optional S3 upload interfaces.

```bash
docker build -t carlsoncompanyx/ltx25-comfyui:0.1.17 .
docker push carlsoncompanyx/ltx25-comfyui:0.1.17
```

The existing retained endpoint still uses its previously published image. The control-capable image defined here is `carlsoncompanyx/ltx25-comfyui:0.1.17`; record its registry digest after the build/push completes before updating the endpoint.

Create a Serverless endpoint with:

- image: `carlsoncompanyx/ltx25-comfyui:0.1.17`
- GPU pool: `ADA_80_PRO` (H100 80 GB) or `AMPERE_80` (A100 80 GB)
- network volume: the 160 GB volume above
- `workersMin=0`, `workersMax=1`
- `containerDiskInGb=40` (the pinned ComfyUI/Torch image is approximately 22 GB)
- `executionTimeoutMs=3600000`
- `LTX_BOOTSTRAP_PROFILE=ltx25_bf16_control`
- `MODEL_BOOTSTRAP_DOWNLOAD=true` only when the volume is not already populated
- `RUNPOD_NETWORK_VOLUME_PATH=/runpod-volume`
- `COMFY_TIMEOUT_SECONDS=3600`

Configure `HF_TOKEN` through a RunPod Secret reference, for example `{{ RUNPOD_SECRET_HF_TOKEN }}`. The referenced Hugging Face token must belong to an account that has accepted the LTX-2 community license and has read access to `Lightricks/LTX-2.5`; a syntactically valid secret reference without that access still fails during bootstrap.

For large MP4 output configure the worker-comfyui S3-compatible variables (`BUCKET_ENDPOINT_URL` plus its access credentials and bucket setting) through RunPodâ€™s secure environment configuration. Without S3 output, responses include base64 only up to `LTX_MAX_BASE64_OUTPUT_BYTES` (20 MB by default).

## API contract

RunPod accepts a normal `input` object; callers do not send a raw ComfyUI graph.

```json
{
  "input": {
    "mode": "i2v",
    "prompt": "A woman walks through a rainy city street at night, cinematic live action.",
    "negative_prompt": "distorted face, extra fingers, warped motion",
    "seed": 42,
    "width": 768,
    "height": 512,
    "duration_seconds": 5,
    "fps": 24,
    "generate_audio": true,
    "enhance_prompt": true,
    "image_url": "https://example.com/reference.png",
    "loras": []
  }
}
```

Supported request modes are `t2v`, `i2v`, `flf2v`, `union_control`, `motion_track`, `pixel_upscale`, and `health`. `health` does not start a generation. Every generation response includes normalized width, height, frames, FPS, seed, model profile, workflow, and timing.

- `t2v`: prompt only.
- `i2v`: one HTTPS image URL or base64 image.
- `flf2v`: first image plus last image.
- `union_control`: control video plus a registered `union_control` adapter; BF16 profile required.
- `motion_track`: one reference image plus normalized sparse tracks and a registered `motion_track` adapter; BF16 profile required.
- `pixel_upscale`: registered in the manifest but intentionally rejected until a current official ComfyUI graph is validated. Native latent spatial upscaling is present in the normal graph and is separate from the Pixel IC-LoRA.

Dimensions are normalized down to multiples of 32. Frame counts are normalized to `1 + 8n`, from 9 through 513. Seconds are converted using the requested FPS. URLs must be HTTPS and cannot resolve to private, loopback, link-local, or metadata addresses. Callers can use only manifest logical LoRA IDs; raw filenames are not accepted.

## n8n HTTP Request example

See `examples/n8n-http-request.json`. Submit to:

```text
POST https://api.runpod.ai/v2/<ENDPOINT_ID>/run
Authorization: Bearer <RUNPOD_API_KEY>
Content-Type: application/json
```

The response contains `id`. Poll:

```text
GET https://api.runpod.ai/v2/<ENDPOINT_ID>/status/<JOB_ID>
Authorization: Bearer <RUNPOD_API_KEY>
```

Use `output.output.url` for S3 output or decode `output.output.data` when the response is inline base64.

## Testing

Local contract tests do not fake GPU generation:

```bash
py -3 -m unittest discover -s tests -v
py -3 -m py_compile handler.py scripts/bootstrap_models.py scripts/convert_workflows.py scripts/smoke_test.py
```

`smoke_test.py` submits a real RunPod job and validates that the returned result is an MP4 container. GPU integration is intentionally separate because it consumes RunPod time. The next deployment test should call `health`, verify the BF16 control files and custom-node load, then run one small official Union depth job. Motion Track should be attempted only after that succeeds.

## Current validation status

- API normalization and error contracts: local tests passing.
- ComfyUI 0.33.0 CPU import/startup: passing in the pinned image; LTX nodes are native ComfyUI nodes, so `ComfyUI-LTXVideo` was not required for this run.
- Real GPU validation: T2V, I2V, and FLF2V each returned an MP4 with H.264 video and synchronized AAC audio, and each passed `ffprobe`.
- Native latent spatial upscaling is active in the T2V/I2V graph. A requested 256x256 latent rendered as 512x512 in those graphs; FLF2V rendered at 256x256.
- Pixel Spatial Upscaler IC-LoRA remains manifest-only/not validated because no current official ComfyUI API graph was available.
- Union Control and Motion Track are statically validated against the pinned official graphs but remain GPU-unvalidated. No control mode is marked operational until a real MP4 completes.
- The official control examples require BF16; int8/convrot compatibility is not claimed without a completed control MP4.

### RunPod validation (2026-08-15)

- Temporary endpoint: `m431efhc55q6r4`; validation image `carlsoncompanyx/ltx25-comfyui:0.1.12`, derived from the existing `0.1.3` image only to publish the handler and startup fixes.
- GPU: NVIDIA H100 80GB HBM3 in `US-CA-2`; ComfyUI reported 81,559 MB VRAM. Peak allocation was not instrumented.
- HF access: the RunPod Secret reference `{{ RUNPOD_SECRET_HF_TOKEN }}` resolved without printing the token. Required int8 files were verified, and the BF16 control transformer/text encoder plus Union adapter were also verified on the persistent volume.
- Base MP4 artifacts: `artifacts/ltx25_t2v_audio_smoke.mp4`, `artifacts/ltx25_i2v_hallway_smoke.mp4`, and `artifacts/ltx25_flf2v_hallway_smoke.mp4`. Each is 9 frames at 24 FPS with an AAC 48 kHz stereo audio stream; video durations are 0.375 seconds and audio duration is 0.330 seconds.
- The endpoint was deleted after validation. Network volume `73qcj6r6nm` was retained.
- The converter now preserves source output slots, and the handler keeps compatibility repairs for older API exports.
Do not add arbitrary LTX-2.3 LoRAs, Ingredients IC-LoRA, identity LoRAs, or community adapters to the registry without an LTX-2.5 workflow validation.


### Official control graph boundaries

The checked-in control graphs are derived from Lightricks/ComfyUI-LTXVideo commit `ac4d99839020b983e956a8ab67ec38aec1b6e65a`:

- Union Control uses BF16 LTX-2.5, the Union IC-LoRA, the `ltx-2.3-spatial-upscaler-x2-1.1.safetensors` loader, and Video Depth Anything `video_depth_anything_vits.pth`.
- The API contract exposes Union `control_type: "depth"` only. Canny and DWPose remain disconnected UI alternatives in the released workflow.
- Motion Track uses BF16 LTX-2.5, the Motion Track IC-LoRA, one reference image, and caller-supplied sparse tracks. The input shape is `tracks: [[{"x": 120, "y": 220}, {"x": 130, "y": 230}]]`.
- The control adapters are registered by logical ID, but are not called operational until a GPU job returns and passes MP4 validation.
## Official sources

- https://github.com/Lightricks/LTX-2
- https://github.com/Lightricks/ComfyUI-LTXVideo
- https://docs.ltx.io/open-source-model/integration-tools/comfy-ui
- https://huggingface.co/Lightricks/LTX-2.5
- https://github.com/runpod-workers/worker-comfyui
