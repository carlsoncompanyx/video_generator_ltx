ARG BASE_IMAGE=carlsoncompanyx/ltx25-comfyui:0.1.3
FROM ${BASE_IMAGE}

ARG LTXVIDEO_GIT_COMMIT=ac4d99839020b983e956a8ab67ec38aec1b6e65a
ARG LTX2_GIT_COMMIT=a95ab856bf29407b6b066ede0abe1846050db56c
ARG VDA_GIT_COMMIT=a0db08e63d1ea571601c45cde4aaee0acdd0544d

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_PREFER_BINARY=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_INPUT=1 \
    COMFY_LOG_LEVEL=INFO \
    COMFY_API_AVAILABLE_MAX_RETRIES=0

# Reuse the published 0.1.3 CUDA/PyTorch/ComfyUI layer. Only the official
# custom nodes required by the control graphs are added here. BuildKit fetches
# the exact pinned public commit archives without Git credentials.
ADD https://codeload.github.com/Lightricks/ComfyUI-LTXVideo/tar.gz/ac4d99839020b983e956a8ab67ec38aec1b6e65a /tmp/ltxvideo.tar.gz
ADD https://codeload.github.com/yuvraj108c/ComfyUI-Video-Depth-Anything/tar.gz/a0db08e63d1ea571601c45cde4aaee0acdd0544d /tmp/vda.tar.gz
ADD https://codeload.github.com/Lightricks/LTX-2/tar.gz/a95ab856bf29407b6b066ede0abe1846050db56c /tmp/ltx2.tar.gz

RUN mkdir -p \
      /comfyui/custom_nodes/ComfyUI-LTXVideo \
      /comfyui/custom_nodes/ComfyUI-Video-Depth-Anything \
    && tar -xzf /tmp/ltxvideo.tar.gz \
         --strip-components=1 \
         -C /comfyui/custom_nodes/ComfyUI-LTXVideo \
    && tar -xzf /tmp/vda.tar.gz \
         --strip-components=1 \
         -C /comfyui/custom_nodes/ComfyUI-Video-Depth-Anything \
    && rm -f /tmp/ltxvideo.tar.gz /tmp/vda.tar.gz

RUN mkdir -p /tmp/ltx2-src     && tar -xzf /tmp/ltx2.tar.gz --strip-components=1 -C /tmp/ltx2-src     && pip install --no-cache-dir --no-deps /tmp/ltx2-src/packages/ltx-core /tmp/ltx2-src/packages/ltx-pipelines     && rm -rf /tmp/ltx2-src /tmp/ltx2.tar.gz

COPY requirements.txt /worker-requirements.txt

RUN pip install --no-cache-dir -r /comfyui/custom_nodes/ComfyUI-LTXVideo/requirements.txt \
    && pip install --no-cache-dir -r /comfyui/custom_nodes/ComfyUI-Video-Depth-Anything/requirements.txt \
    && pip install --no-cache-dir -r /worker-requirements.txt \
    && pip install --no-cache-dir "kornia==0.7.4"

# Import-time smoke check; model weights stay on the attached volume.
# VDA imports ComfyUI's device manager and is exercised by ComfyUI startup on a RunPod GPU.
# Docker Desktop build workers do not have an NVIDIA driver.
RUN cd /comfyui && python -c "import torch; import comfy; import folder_paths; print(torch.__version__)" \
    && cd /comfyui && python -c "import sys; import comfy.options as options; options.args_parsing=True; sys.argv.append('--cpu'); import importlib.util; path='/comfyui/custom_nodes/ComfyUI-LTXVideo'; spec=importlib.util.spec_from_file_location('custom_nodes.ComfyUI_LTXVideo', path + '/__init__.py', submodule_search_locations=[path]); module=importlib.util.module_from_spec(spec); sys.modules[spec.name]=module; spec.loader.exec_module(module); print('ltxvideo-runtime-import-ok')"


# Import the pinned official LTX native packages without loading model weights.
RUN python -c "from ltx_pipelines.ti2vid_two_stages_hq import TI2VidTwoStagesHQPipeline; from ltx_pipelines.utils.model_paths import ModelPaths; from ltx_core.loader import LoraPathStrengthAndSDOps; print('native-ltx-runtime-import-ok')"
COPY handler.py /worker/handler.py
COPY model_manifest.json /worker/model_manifest.json
COPY config /worker/config
COPY scripts /worker/scripts
COPY workflows /worker/workflows
COPY extra_model_paths.yaml /comfyui/extra_model_paths.yaml
COPY start.sh /worker/start.sh
RUN chmod +x /worker/start.sh /worker/scripts/*.py

WORKDIR /worker
CMD ["/bin/bash", "/worker/start.sh"]
