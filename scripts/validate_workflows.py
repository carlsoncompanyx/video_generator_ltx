#!/usr/bin/env python3
"""Validate LTX API graphs against the pinned ComfyUI object_info schema.

This is intentionally independent of the request handler. It catches stale
UI-export fields, missing current /prompt inputs, dangling links, and model
filenames that are not registered in the worker manifest before a paid job is
started.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "model_manifest.json"


def load_json(source: str | Path) -> Any:
    value = str(source)
    if urlparse(value).scheme in {"http", "https"}:
        request = Request(value, headers={"Accept": "application/json"})
        with urlopen(request, timeout=30) as response:
            return json.load(response)
    return json.loads(Path(value).read_text(encoding="utf-8"))


def iter_references(value: Any):
    if isinstance(value, list) and len(value) == 2 and isinstance(value[0], str):
        yield value
        return
    if isinstance(value, dict):
        for nested in value.values():
            yield from iter_references(nested)
    elif isinstance(value, list):
        for nested in value:
            yield from iter_references(nested)


def schema_required_inputs(schema_entry: dict[str, Any]) -> dict[str, Any]:
    input_schema = schema_entry.get("input") or {}
    required = input_schema.get("required") or {}
    if isinstance(required, dict):
        return required
    if isinstance(required, list):
        return {str(name): None for name in required}
    return {}


def schema_output_count(schema_entry: dict[str, Any]) -> int | None:
    output = schema_entry.get("output")
    if isinstance(output, list):
        return len(output)
    if isinstance(output, tuple):
        return len(output)
    return None


def manifest_filenames(manifest: dict[str, Any]) -> set[str]:
    filenames: set[str] = set()
    for profile in (manifest.get("profiles") or {}).values():
        for group in ("required", "optional"):
            for entry in profile.get(group, []) or []:
                path = entry.get("path")
                if path:
                    filenames.add(Path(path).name)
    for entry in (manifest.get("lora_repositories") or {}).values():
        path = entry.get("path")
        if path:
            filenames.add(Path(path).name)
    return filenames


def schema_allowed_values(schema_entry: dict[str, Any], name: str) -> list[str] | None:
    spec = schema_required_inputs(schema_entry).get(name)
    if not isinstance(spec, list) or not spec or not isinstance(spec[0], list):
        return None
    values = spec[0]
    return values if all(isinstance(value, str) for value in values) else None


def validate_graph(
    workflow: dict[str, Any],
    object_info: dict[str, Any] | None = None,
    manifest: dict[str, Any] | None = None,
    require_media_placeholders: bool = True,
) -> list[str]:
    errors: list[str] = []
    node_ids = set(workflow)
    known_models = manifest_filenames(manifest or {})

    model_inputs_by_class = {
        "UNETLoader": {"unet_name"},
        "CLIPLoader": {"clip_name"},
        "VAELoader": {"vae_name"},
        "LatentUpscaleModelLoader": {"model_name"},
        "LTXICLoRALoaderModelOnly": {"lora_name"},
        "GemmaAPITextEncode": {"ckpt_name"},
        "LoadVideoDepthAnythingModel": {"model"},
    }

    for node_id, node in workflow.items():
        if not isinstance(node, dict):
            errors.append(f"{node_id}: node is not an object")
            continue
        class_type = node.get("class_type")
        if not class_type:
            errors.append(f"{node_id}: missing class_type")
            continue
        inputs = node.get("inputs")
        if not isinstance(inputs, dict):
            errors.append(f"{node_id} ({class_type}): inputs is not an object")
            inputs = {}

        schema_entry = None
        if object_info is not None:
            schema_entry = object_info.get(class_type)
            if not isinstance(schema_entry, dict):
                errors.append(f"{node_id} ({class_type}): class is absent from /object_info")
            else:
                for name in schema_required_inputs(schema_entry):
                    has_direct_value = name in inputs and inputs[name] is not None
                    has_dynamic_children = any(
                        key.startswith(f"{name}.") for key in inputs
                    )
                    if not has_direct_value and not has_dynamic_children:
                        errors.append(f"{node_id} ({class_type}): required input missing: {name}")
                    elif has_direct_value:
                        allowed = schema_allowed_values(schema_entry, name)
                        value = inputs.get(name)
                        if allowed and isinstance(value, str) and value not in allowed:
                            errors.append(
                                f"{node_id} ({class_type}): invalid selection {name}={value!r}; "
                                f"allowed values are {allowed!r}"
                            )

        for reference in iter_references(inputs):
            source = workflow.get(reference[0])
            if source is None:
                errors.append(f"{node_id} ({class_type}): dangling reference {reference}")
                continue
            if object_info is not None:
                source_schema = object_info.get(source.get("class_type"))
                output_count = (
                    schema_output_count(source_schema)
                    if isinstance(source_schema, dict)
                    else None
                )
                slot = reference[1]
                if output_count is not None and (
                    not isinstance(slot, int) or slot < 0 or slot >= output_count
                ):
                    errors.append(
                        f"{node_id} ({class_type}): reference {reference} exceeds "
                        f"{reference[0]} output count {output_count}"
                    )

        if class_type == "LoadVideo":
            file_name = inputs.get("file")
            if require_media_placeholders and (
                not isinstance(file_name, str) or not file_name
            ):
                errors.append(f"{node_id} (LoadVideo): file placeholder is empty")
        if class_type == "LoadImage":
            image_name = inputs.get("image")
            if require_media_placeholders and (
                not isinstance(image_name, str) or not image_name
            ):
                errors.append(f"{node_id} (LoadImage): image placeholder is empty")

        for input_name in model_inputs_by_class.get(class_type, set()):
            value = inputs.get(input_name)
            if not isinstance(value, str) or not value or value.startswith("["):
                continue
            if value.startswith("__LTX_"):
                continue
            if value not in known_models:
                errors.append(
                    f"{node_id} ({class_type}): unregistered model filename: {input_name}={value}"
                )

    return errors


def validate_workflow_file(
    path: Path,
    object_info: dict[str, Any] | None,
    manifest: dict[str, Any] | None,
    require_media_placeholders: bool,
) -> list[str]:
    workflow = load_json(path)
    if not isinstance(workflow, dict):
        return [f"{path.name}: workflow root is not an object"]
    return validate_graph(
        workflow,
        object_info=object_info,
        manifest=manifest,
        require_media_placeholders=require_media_placeholders,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workflow", action="append", type=Path, required=True)
    parser.add_argument("--object-info", help="Path or URL for the pinned ComfyUI /object_info response")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--allow-empty-media",
        action="store_true",
        help="Allow empty media fields when validating an intentionally unpatched template",
    )
    args = parser.parse_args()

    try:
        object_info = load_json(args.object_info) if args.object_info else None
        manifest = load_json(args.manifest)
    except Exception as exc:
        print(f"SCHEMA_LOAD_FAILED: {exc}", file=sys.stderr)
        return 2

    failed = False
    for workflow_path in args.workflow:
        errors = validate_workflow_file(
            workflow_path,
            object_info=object_info,
            manifest=manifest,
            require_media_placeholders=not args.allow_empty_media,
        )
        if errors:
            failed = True
            print(f"FAIL {workflow_path}")
            for error in errors:
                print(f"  - {error}")
        else:
            print(f"PASS {workflow_path}")

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
