#!/usr/bin/env python3
"""Convert current ComfyUI v3/subgraph workflows into API-format graphs.

The upstream LTX-2.5 templates are intentionally kept in ``workflows/source``.
ComfyUI's /prompt endpoint accepts the older node-id -> {class_type, inputs}
shape, so this small converter flattens only the subgraphs we have validated.
It also writes a parameter map consumed by the RunPod handler.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "workflows" / "source"
OUTPUT = ROOT / "workflows"
OFFICIAL_LTXVIDEO_COMMIT = "ac4d99839020b983e956a8ab67ec38aec1b6e65a"
LTX25_API_CHECKPOINT_BF16 = "ltx-2.5-22b-distilled-transformer-bf16.safetensors"
LTX25_API_CHECKPOINT_BF16 = "ltx-2.5-22b-distilled-transformer-bf16.safetensors"


def as_key(value: Any) -> str:
    return str(value)


def link_record(link: Any) -> dict[str, Any]:
    if isinstance(link, dict):
        return link
    return {
        "id": link[0],
        "origin_id": link[1],
        "origin_slot": link[2],
        "target_id": link[3],
        "target_slot": link[4],
        "type": link[5] if len(link) > 5 else None,
    }


def index_links(links: list[Any]) -> dict[str, dict[str, Any]]:
    return {as_key(link_record(link)["id"]): link_record(link) for link in links}


def widget_value(node: dict[str, Any], name: str | None, fallback: Any = None) -> Any:
    values = node.get("widgets_values") or []
    known = {
        "RandomNoise": {"noise_seed": 0, "control_after_generate": 1},
        "KSamplerSelect": {"sampler_name": 0},
        "ManualSigmas": {"sigmas": 0},
        "UNETLoader": {"unet_name": 0, "weight_dtype": 1},
        "VAELoader": {"vae_name": 0},
        "CLIPLoader": {"clip_name": 0, "type": 1, "device": 2},
        "LatentUpscaleModelLoader": {"model_name": 0},
        "SaveVideo": {"filename_prefix": 0, "format": 1, "codec": 2},
        "CreateVideo": {"fps": 0, "bit_depth": 1},
        "CLIPTextEncode": {"text": 0},
        "VAEDecodeTiled": {
            "tile_size": 0,
            "overlap": 1,
            "temporal_size": 2,
            "temporal_overlap": 3,
        },
        "LTXVDualCFGGuider": {"video_cfg": 0, "audio_cfg": 1},
        "LTXVConditioning": {"frame_rate": 0},
        "LTXVEmptyLatentAudio": {"frames_number": 0, "frame_rate": 1, "batch_size": 2},
        "EmptyLTXVLatentVideo": {"width": 0, "height": 1, "length": 2, "batch_size": 3},
        "CFGGuider": {"cfg": 0},
        "PrimitiveInt": {"value": 0, "control_after_generate": 1},
        "PrimitiveFloat": {"value": 0, "control_after_generate": 1},
        "PrimitiveBoolean": {"value": 0},
        "PrimitiveString": {"value": 0},
        "PrimitiveStringMultiline": {"value": 0},
        "ComfyMathExpression": {"expression": 0},
        "TextGenerate": {
            "max_length": 1,
            "sampling_mode": 2,
            "temperature": 3,
            "top_k": 4,
            "top_p": 5,
            "min_p": 6,
            "repetition_penalty": 7,
            "seed": 8,
            "presence_penalty": 9,
            "thinking": 10,
            "use_default_template": 11,
        },
        "TextGenerateLTX2Prompt": {
            "max_length": 1,
            "sampling_mode": 2,
            "temperature": 3,
            "top_k": 4,
            "top_p": 5,
            "min_p": 6,
            "repetition_penalty": 7,
            "seed": 8,
            "presence_penalty": 9,
            "thinking": 10,
            "use_default_template": 11,
        },
        "LoadImage": {"image": 0, "upload": 1},
        "LoadVideo": {
            "video": 0,
            "force_rate": 1,
            "custom_width": 2,
            "custom_height": 3,
            "frame_load_cap": 4,
            "skip_first_frames": 5,
            "select_every_nth": 6,
        },
        "LoadVideoDepthAnythingModel": {"model": 0},
        "VideoDepthAnythingProcess": {"input_size": 0, "max_res": 1, "precision": 2},
        "VideoDepthAnythingOutput": {"colormap": 0},
        "StringContains": {"substring": 1, "case_sensitive": 2},
        "LTXICLoRALoaderModelOnly": {"lora_name": 0, "strength_model": 1},
        "LTXAddVideoICLoRAGuide": {
            "frame_idx": 0,
            "strength": 1,
            "latent_downscale_factor": 2,
            "crop": 3,
            "use_tiled_encode": 4,
            "tile_size": 5,
            "tile_overlap": 6,
        },
        "LTXVSparseTrackEditor": {"points_store": 0, "coordinates": 1, "points_to_sample": 2},
        "LTXVDrawTracks": {"tracks": 0, "width": 1, "height": 2},
    }
    slots = known.get(node.get("type"), {})
    if name in slots and slots[name] < len(values):
        return values[slots[name]]
    if name is None and values:
        return values[0]
    return fallback


def node_inputs(node: dict[str, Any]) -> dict[str, Any]:
    """Return widget-only inputs represented in a UI node's widget array."""
    result: dict[str, Any] = {}
    typ = node.get("type")
    names: list[str] = []
    for item in node.get("inputs") or []:
        widget = item.get("widget")
        if widget and widget.get("name"):
            names.append(widget["name"])
    explicit = {
        "RandomNoise": ["noise_seed", "control_after_generate"],
        "KSamplerSelect": ["sampler_name"],
        "ManualSigmas": ["sigmas"],
        "UNETLoader": ["unet_name", "weight_dtype"],
        "VAELoader": ["vae_name"],
        "CLIPLoader": ["clip_name", "type", "device"],
        "LatentUpscaleModelLoader": ["model_name"],
        "SaveVideo": ["filename_prefix", "format", "codec"],
        "CreateVideo": ["fps", "bit_depth"],
        "CLIPTextEncode": ["text"],
        "VAEDecodeTiled": ["tile_size", "overlap", "temporal_size", "temporal_overlap"],
        "LTXVDualCFGGuider": ["video_cfg", "audio_cfg"],
        "LTXVConditioning": ["frame_rate"],
        "LTXVEmptyLatentAudio": ["frames_number", "frame_rate", "batch_size"],
        "EmptyLTXVLatentVideo": ["width", "height", "length", "batch_size"],
        "CFGGuider": ["cfg"],
        "PrimitiveInt": ["value", "control_after_generate"],
        "PrimitiveFloat": ["value", "control_after_generate"],
        "PrimitiveBoolean": ["value"],
        "PrimitiveString": ["value"],
        "PrimitiveStringMultiline": ["value"],
        "ComfyMathExpression": ["expression"],
        "TextGenerate": [
            "max_length", "sampling_mode", "temperature", "top_k", "top_p", "min_p",
            "repetition_penalty", "seed", "presence_penalty", "thinking", "use_default_template",
        ],
        "TextGenerateLTX2Prompt": [
            "max_length", "sampling_mode", "temperature", "top_k", "top_p", "min_p",
            "repetition_penalty", "seed", "presence_penalty", "thinking", "use_default_template",
        ],
        "LoadImage": ["image", "upload"],
        "LoadVideo": [
            "video", "force_rate", "custom_width", "custom_height", "frame_load_cap",
            "skip_first_frames", "select_every_nth",
        ],
        "LoadVideoDepthAnythingModel": ["model"],
        "VideoDepthAnythingProcess": ["input_size", "max_res", "precision"],
        "VideoDepthAnythingOutput": ["colormap"],
        "StringContains": ["substring", "case_sensitive"],
        "LTXICLoRALoaderModelOnly": ["lora_name", "strength_model"],
        "LTXAddVideoICLoRAGuide": [
            "frame_idx", "strength", "latent_downscale_factor", "crop",
            "use_tiled_encode", "tile_size", "tile_overlap",
        ],
        "LTXVSparseTrackEditor": ["points_store", "coordinates", "points_to_sample"],
        "LTXVDrawTracks": ["tracks", "width", "height"],
    }
    for name in explicit.get(typ, []):
        if name not in names:
            names.append(name)
    for name in names:
        value = widget_value(node, name)
        if value is not None:
            result[name] = value
    return result


def clone_source(source: Any, prefix: str) -> list[Any]:
    if source is None:
        return []
    if isinstance(source, list):
        return [prefix + as_key(source[0]), source[1]]
    return source


class Flattener:
    def __init__(self, graph: dict[str, Any]):
        definitions = graph.get("definitions") or {}
        self.subgraphs = {as_key(x["id"]): x for x in definitions.get("subgraphs") or []}
        self.output: dict[str, dict[str, Any]] = {}
        self.parameter_map: dict[str, Any] = {"nodes": {}, "inputs": {}, "outputs": {}}
        self._top_links = index_links(graph.get("links") or [])

    def node_id(self, prefix: str, node_id: Any) -> str:
        return f"{prefix}{as_key(node_id)}"

    def add_parameter_node(self, node: dict[str, Any], output_id: str, prefix: str) -> None:
        title = (node.get("title") or node.get("properties", {}).get("Node name for S&R") or "").lower()
        typ = node.get("type")
        key = output_id
        if typ in {"PrimitiveStringMultiline", "PrimitiveString"} and "prompt" in title:
            key = "prompt"
        elif typ == "PrimitiveBoolean" and "enhance" in title:
            key = "prompt_enhance"
        elif typ == "PrimitiveInt":
            for candidate in ("width", "height", "frame_rate", "duration"):
                if candidate in title:
                    key = candidate
        elif typ == "RandomNoise":
            key = "seed"
        self.parameter_map["nodes"].setdefault(key, []).append({"id": output_id, "class_type": typ, "prefix": prefix})

    def source_for_link(
        self,
        link_id: Any,
        links: dict[str, dict[str, Any]],
        sources: dict[tuple[Any, int], Any],
    ) -> Any:
        record = links.get(as_key(link_id))
        if not record:
            return None
        origin = (record["origin_id"], int(record.get("origin_slot", 0)))
        return sources.get(origin)

    def flatten_subgraph(
        self,
        instance: dict[str, Any],
        subgraph: dict[str, Any],
        prefix: str,
        external_inputs: dict[int, Any],
    ) -> dict[int, Any]:
        links = index_links(subgraph.get("links") or [])
        regular: dict[Any, dict[str, Any]] = {n["id"]: n for n in subgraph.get("nodes") or []}
        sources: dict[tuple[Any, int], Any] = {}
        for record in links.values():
            if record["origin_id"] in (-10, -20):
                continue
            origin_id = record["origin_id"]
            origin_node = regular.get(origin_id)
            if origin_node is not None:
                origin_slot = int(record.get("origin_slot", 0))
                sources[(origin_id, origin_slot)] = [self.node_id(prefix, origin_id), origin_slot]

        # Reroute nodes are UI-only. Preserve their source while flattening so
        # downstream API links point at the real producer rather than a node
        # that is intentionally omitted from the /prompt graph.
        for node in subgraph.get("nodes") or []:
            if node.get("type") != "Reroute":
                continue
            input_link = next(
                (item.get("link") for item in node.get("inputs") or [] if item.get("link") is not None),
                None,
            )
            if input_link is None:
                continue
            record = links.get(as_key(input_link))
            if not record:
                continue
            source = None
            if record["origin_id"] == -10:
                source = external_inputs.get(int(record.get("origin_slot", 0)))
            else:
                source = sources.get((record["origin_id"], int(record.get("origin_slot", 0))))
            if source is None:
                continue
            for output_link in links.values():
                if output_link["origin_id"] == node["id"]:
                    sources[(node["id"], int(output_link.get("origin_slot", 0)))] = source

        # Resolve each node's inputs and emit ordinary API-format nodes.
        for node in subgraph.get("nodes") or []:
            if node.get("type") in {"MarkdownNote", "Note", "Reroute"}:
                continue
            out_id = self.node_id(prefix, node["id"])
            inputs: dict[str, Any] = node_inputs(node)
            for item in node.get("inputs") or []:
                name = item.get("name")
                if not name:
                    continue
                link_id = item.get("link")
                if link_id is not None:
                    value = self.source_for_link(link_id, links, sources)
                    if value is not None:
                        inputs[name] = value
                    else:
                        record = links.get(as_key(link_id))
                        if record and record["origin_id"] == -10:
                            slot = int(record.get("origin_slot", 0))
                            if slot in external_inputs:
                                inputs[name] = external_inputs[slot]
                elif item.get("widget"):
                    value = widget_value(node, item["widget"].get("name"), None)
                    if value is not None:
                        inputs[name] = value
            # The upstream graph uses links from the subgraph input boundary;
            # inputs above already handled those. Avoid carrying UI-only widget
            # values for missing dynamic sockets.
            self.output[out_id] = {"class_type": node["type"], "inputs": inputs, "_meta": {"source_id": node["id"], "prefix": prefix}}
            self.add_parameter_node(node, out_id, prefix)

        result: dict[int, Any] = {}
        for record in links.values():
            if record["target_id"] == -20:
                source = sources.get((record["origin_id"], int(record.get("origin_slot", 0))))
                if source is not None:
                    result[int(record.get("target_slot", 0))] = source
        return result

    def convert(self, graph: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        top_nodes = {n["id"]: n for n in graph.get("nodes") or []}
        top_sources: dict[tuple[Any, int], Any] = {}
        for link in self._top_links.values():
            origin = (link["origin_id"], int(link.get("origin_slot", 0)))
            if origin in top_sources:
                continue
            top_sources[origin] = [self.node_id("top_", link["origin_id"]), int(link.get("origin_slot", 0))]

        subgraph_outputs: dict[Any, dict[int, Any]] = {}
        for node in graph.get("nodes") or []:
            typ = as_key(node.get("type"))
            if typ not in self.subgraphs:
                continue
            subgraph = self.subgraphs[typ]
            incoming: dict[int, Any] = {}
            for link in self._top_links.values():
                if link["target_id"] != node["id"]:
                    continue
                origin_id = link["origin_id"]
                incoming[int(link.get("target_slot", 0))] = [self.node_id("top_", origin_id), int(link.get("origin_slot", 0))]
            # Supply unlinked widget defaults through the subgraph boundary
            # before flattening. This matters when a UI Reroute sits between a
            # group input (for example the checkpoint or LoRA filename) and a
            # real node; the later API patch cannot target an omitted Reroute.
            widget_position = 0
            for index, item in enumerate(node.get("inputs") or []):
                if not item.get("widget"):
                    continue
                values = node.get("widgets_values") or []
                value = values[widget_position] if widget_position < len(values) else None
                widget_position += 1
                if item.get("link") is None and value is not None:
                    incoming.setdefault(index, value)

            instance_prefix = f"sg_{node['id']}_"
            subgraph_outputs[node["id"]] = self.flatten_subgraph(node, subgraph, instance_prefix, incoming)

        for node in graph.get("nodes") or []:
            typ = as_key(node.get("type"))
            if typ in self.subgraphs or typ in {"MarkdownNote", "Note", "ResolutionSelector"}:
                continue
            out_id = self.node_id("top_", node["id"])
            inputs = node_inputs(node)
            for item in node.get("inputs") or []:
                link_id = item.get("link")
                if link_id is None:
                    continue
                link = self._top_links.get(as_key(link_id))
                if not link:
                    continue
                source_id = link["origin_id"]
                source_node = top_nodes.get(source_id)
                if source_node and as_key(source_node.get("type")) in self.subgraphs:
                    source = subgraph_outputs.get(source_id, {}).get(int(link.get("origin_slot", 0)))
                else:
                    source = [self.node_id("top_", source_id), int(link.get("origin_slot", 0))]
                if source is not None:
                    inputs[item["name"]] = source
            self.output[out_id] = {"class_type": node["type"], "inputs": inputs, "_meta": {"source_id": node["id"], "prefix": "top_"}}

        # API-format top-level graph inputs from subgraph instances that have
        # no external link are represented by the instance widget defaults.
        for node in graph.get("nodes") or []:
            typ = as_key(node.get("type"))
            if typ not in self.subgraphs:
                continue
            subgraph = self.subgraphs[typ]
            widget_position = 0
            for index, item in enumerate(node.get("inputs") or []):
                if not item.get("widget"):
                    continue
                values = node.get("widgets_values") or []
                value = values[widget_position] if widget_position < len(values) else None
                widget_position += 1
                if item.get("link") is not None or value is None:
                    continue
                # Locate the internal boundary target and patch the flattened
                # node input by subgraph input name, rather than relying on an
                # opaque UI instance node id in the handler.
                for boundary in subgraph.get("links") or []:
                    record = link_record(boundary)
                    if record["origin_id"] == -10 and int(record.get("origin_slot", -1)) == index:
                        target_id = self.node_id(f"sg_{node['id']}_", record["target_id"])
                        target = self.output.get(target_id)
                        if target is not None:
                            target_name = next((x.get("name") for x in (subgraph.get("nodes") or []) if x.get("id") == record["target_id"] for _ in [0]), None)
                            # The name is the socket on the target node; find
                            # it by matching the boundary link target slot.
                            target_node = next(x for x in subgraph.get("nodes") or [] if x.get("id") == record["target_id"])
                            socket = (target_node.get("inputs") or [])[int(record.get("target_slot", 0))]
                            target["inputs"][socket["name"]] = value

        # Wire any ordinary top-level links whose origin is a subgraph output
        # after all flattened nodes exist.
        for link in self._top_links.values():
            target_node = self.output.get(self.node_id("top_", link["target_id"]))
            if target_node is None:
                continue
            origin_id = link["origin_id"]
            source_node = top_nodes.get(origin_id)
            if source_node and as_key(source_node.get("type")) in self.subgraphs:
                source = subgraph_outputs.get(origin_id, {}).get(int(link.get("origin_slot", 0)))
            else:
                source = [self.node_id("top_", origin_id), int(link.get("origin_slot", 0))]
            target_node["inputs"][next((item["name"] for item in (top_nodes[link["target_id"]].get("inputs") or []) if item.get("link") == link["id"]), "video")] = source

        # Group instances are flattened into ordinary API nodes. During the
        # first pass, a link from one group instance to another is represented
        # as a temporary ["top_<group_id>", slot] reference because the
        # producing group's flattened output is not known yet. ComfyUI's
        # /prompt API cannot resolve that virtual group node, so replace those
        # references with the actual flattened source after every subgraph has
        # been expanded.
        def resolve_group_reference(value: Any) -> Any:
            if isinstance(value, list) and len(value) == 2 and isinstance(value[0], str):
                prefix, slot = value
                if prefix.startswith("top_"):
                    try:
                        group_id = int(prefix[4:])
                        resolved = subgraph_outputs.get(group_id, {}).get(int(slot))
                    except (TypeError, ValueError):
                        resolved = None
                    if resolved is not None:
                        return resolved
            if isinstance(value, dict):
                return {key: resolve_group_reference(item) for key, item in value.items()}
            return value

        for node in self.output.values():
            if isinstance(node, dict):
                node["inputs"] = {
                    key: resolve_group_reference(value)
                    for key, value in (node.get("inputs") or {}).items()
                }

        return self.output, self.parameter_map


def _contains_reference(value: Any, node_ids: set[str]) -> bool:
    if isinstance(value, list) and len(value) == 2 and isinstance(value[0], str):
        return value[0] in node_ids
    if isinstance(value, dict):
        return any(_contains_reference(item, node_ids) for item in value.values())
    return False


def _prune_union_to_depth(api_graph: dict[str, dict[str, Any]]) -> None:
    """Keep the official Union graph's wired Depth branch only.

    The released graph also contains disconnected Canny and DW-Pose branches
    intended for manual UI rewiring. They are not part of this API contract
    yet; keeping them would force unrelated preprocessor packages into the
    worker and would make a caller believe those control types were supported.
    """
    removable = {
        node_id
        for node_id, node in api_graph.items()
        if node.get("class_type") in {"CannyEdgePreprocessor", "DWPreprocessor"}
    }
    if not removable:
        return
    for node_id, node in api_graph.items():
        if node_id in removable:
            continue
        if _contains_reference(node.get("inputs", {}), removable):
            raise ValueError(f"cannot prune Union control preprocessor {node_id}: it is still wired")
    for node_id in removable:
        del api_graph[node_id]


def _apply_current_ltx_schema_defaults(api_graph: dict[str, dict[str, Any]]) -> None:
    """Bridge current ComfyUI schemas where the released UI export is older.

    The official LTX control templates are exported from the UI and omit some
    widget sockets that the current /prompt schema now marks as required.
    These values are the defaults from the pinned official templates and are
    deliberately applied here, at conversion time, rather than scattered
    through the request handler or hand-edited generated JSON.
    """
    defaults: dict[str, dict[str, Any]] = {
        "StringContains": {
            "substring": "ltxv_",
            "case_sensitive": True,
        },
        "GemmaAPITextEncode": {
            "enhance_prompt": False,
        },
        "LTXICLoRALoaderModelOnly": {
            "strength_model": 1,
        },
        "LTXAddVideoICLoRAGuide": {
            "frame_idx": 0,
            "strength": 1,
            "crop": "disabled",
            "use_tiled_encode": False,
            "tile_size": 256,
            "tile_overlap": 64,
        },
        "CFGGuider": {
            "cfg": 1,
        },
    }
    for node in api_graph.values():
        if node.get("class_type") == "GemmaAPITextEncode":
            # GemmaAPITextEncode reads metadata from a checkpoint or diffusion-model file.
            # The Gemma encoder files are loaded by the native CLIPLoader branch.
            node.setdefault("inputs", {})["ckpt_name"] = LTX25_API_CHECKPOINT_BF16
        class_defaults = defaults.get(node.get("class_type"))
        if not class_defaults:
            continue
        inputs = node.setdefault("inputs", {})
        for name, value in class_defaults.items():
            inputs.setdefault(name, copy.deepcopy(value))


def _normalize_current_api_input_names(api_graph: dict[str, dict[str, Any]]) -> None:
    """Normalize names that changed between the UI export and /prompt API."""
    for node in api_graph.values():
        if node.get("class_type") != "LoadVideo":
            continue
        inputs = node.setdefault("inputs", {})
        # The official template calls this widget video; ComfyUI 0.33's
        # current node schema calls the required /prompt field file.
        if "file" not in inputs and "video" in inputs:
            inputs["file"] = inputs.pop("video")
        if not inputs.get("file"):
            inputs["file"] = "__LTX_CONTROL_VIDEO__.mp4"


def _repair_removed_resolution_selector(api_graph: dict[str, dict[str, Any]]) -> None:
    """Replace API references to the UI-only ResolutionSelector."""
    for node in api_graph.values():
        if node.get("class_type") != "PrimitiveInt":
            continue
        source_id = (node.get("_meta") or {}).get("source_id")
        value = (node.get("inputs") or {}).get("value")
        if source_id in {360, 372} and isinstance(value, list) and len(value) == 2:
            if str(value[0]) not in api_graph:
                node["inputs"]["value"] = 576 if source_id == 360 else 1024

def _validate_api_graph(api_graph: dict[str, dict[str, Any]]) -> None:
    dangling: list[dict[str, Any]] = []
    for node_id, node in api_graph.items():
        for input_name, value in (node.get("inputs") or {}).items():
            if isinstance(value, list) and len(value) == 2 and isinstance(value[0], str):
                if value[0] not in api_graph:
                    dangling.append({"node": node_id, "input": input_name, "reference": value})
    if dangling:
        raise ValueError(f"API graph contains dangling references: {dangling[:5]}")


def convert(path: Path, output_path: Path) -> None:
    graph = json.loads(path.read_text(encoding="utf-8"))
    flattener = Flattener(graph)
    api_graph, parameter_map = flattener.convert(graph)
    if "union_control" in path.name:
        _prune_union_to_depth(api_graph)
    _apply_current_ltx_schema_defaults(api_graph)
    _normalize_current_api_input_names(api_graph)
    _repair_removed_resolution_selector(api_graph)
    if "ac4d998" in path.name:
        _validate_api_graph(api_graph)
    clean_graph = {k: v for k, v in api_graph.items()}
    output_path.write_text(json.dumps(clean_graph, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    map_path = output_path.with_suffix(".parameters.json")
    parameter_map["source"] = str(path.name)
    if "ac4d998" in path.name:
        parameter_map["source_commit"] = OFFICIAL_LTXVIDEO_COMMIT
        parameter_map["control_preprocessors"] = ["depth"] if "union_control" in path.name else []
    map_path.write_text(json.dumps(parameter_map, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    pairs = {
        "comfy_org_video_ltx2_5_t2v.json": "ltx25_t2v.json",
        "lightricks_ltx25_t2v_main.json": "ltx25_t2v.json",
        "comfy_org_video_ltx2_5_i2v.json": "ltx25_i2v.json",
        "comfy_org_video_ltx2_5_flf2v.json": "ltx25_flf2v.json",
        "lightricks_ltx25_union_control_ac4d998.json": "ltx25_union_control.json",
        "lightricks_ltx25_motion_track_ac4d998.json": "ltx25_motion_track.json",
    }
    fallbacks = {
        "lightricks_ltx25_union_control.json": "ltx25_union_control.json",
        "lightricks_ltx25_motion_track.json": "ltx25_motion_track.json",
    }
    if args.input:
        convert(args.input, args.output or OUTPUT / args.input.name)
        return
    for source_name, output_name in pairs.items():
        source = SOURCE / source_name
        if source.exists():
            convert(source, OUTPUT / output_name)
    for source_name, output_name in fallbacks.items():
        source = SOURCE / source_name
        official_name = source_name.replace(".json", "_ac4d998.json")
        if source.exists() and not (SOURCE / official_name).exists():
            convert(source, OUTPUT / output_name)


if __name__ == "__main__":
    main()
