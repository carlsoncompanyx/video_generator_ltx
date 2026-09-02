#!/usr/bin/env bash
set -euo pipefail

PROFILE="${LTX_BOOTSTRAP_PROFILE:-ltx25_bf16_core}"
BOOTSTRAP_ARGS=(--profile "${PROFILE}")
if [[ "${MODEL_BOOTSTRAP_DOWNLOAD:-false}" == "true" ]]; then
  BOOTSTRAP_ARGS+=(--download)
fi
if [[ -n "${LTX_BOOTSTRAP_LORAS:-}" ]]; then
  IFS=',' read -ra LORAS <<< "${LTX_BOOTSTRAP_LORAS}"
  for logical_id in "${LORAS[@]}"; do
    [[ -n "${logical_id}" ]] && BOOTSTRAP_ARGS+=(--lora "${logical_id}")
  done
fi

python /worker/scripts/bootstrap_models.py "${BOOTSTRAP_ARGS[@]}"

cd /comfyui
python main.py \
  --listen 127.0.0.1 \
  --port 8188 \
  --extra-model-paths-config /comfyui/extra_model_paths.yaml \
  --disable-auto-launch \
  > /tmp/comfyui.log 2>&1 &
COMFY_PID=$!
echo "${COMFY_PID}" > /tmp/comfyui.pid
tail -n +1 -f /tmp/comfyui.log &
COMFY_LOG_TAIL_PID=$!
trap 'kill "${COMFY_PID}" "${COMFY_LOG_TAIL_PID}" 2>/dev/null || true' EXIT

for attempt in $(seq 1 240); do
  if ! kill -0 "${COMFY_PID}" 2>/dev/null; then
    echo "ComfyUI exited during startup" >&2
    tail -n 160 /tmp/comfyui.log >&2 || true
    exit 1
  fi
  if python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8188/', timeout=2).read(1)" >/dev/null 2>&1; then
    break
  fi
  sleep 1
  if [[ "${attempt}" == "240" ]]; then
    echo "ComfyUI did not become reachable" >&2
    tail -n 160 /tmp/comfyui.log >&2 || true
    exit 1
  fi
done

echo "LTX-2.5 worker ready; ComfyUI log is /tmp/comfyui.log"
exec python /worker/handler.py
