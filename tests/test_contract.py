import base64
import json
import unittest
from pathlib import Path

import handler


ROOT = Path(__file__).parents[1]


class LtxContractTests(unittest.TestCase):
    def test_duration_normalizes_to_ltx_frame_rule(self):
        result = handler.normalize_request({"mode": "t2v", "prompt": "x", "duration_seconds": 5, "fps": 24})
        self.assertEqual(result["frames"], 121)

    def test_resolution_is_multiple_of_32(self):
        result = handler.normalize_request({"mode": "t2v", "prompt": "x", "width": 769, "height": 513})
        self.assertEqual((result["width"], result["height"]), (768, 512))

    def test_i2v_requires_image(self):
        with self.assertRaises(handler.ContractError) as raised:
            handler.normalize_request({"mode": "i2v", "prompt": "x"})
        self.assertEqual(raised.exception.code, "MISSING_IMAGE")

    def test_flf2v_requires_last_frame(self):
        with self.assertRaises(handler.ContractError) as raised:
            handler.normalize_request({"mode": "flf2v", "prompt": "x", "image_base64": "a"})
        self.assertEqual(raised.exception.code, "MISSING_LAST_FRAME")

    def test_unknown_lora_is_rejected(self):
        with self.assertRaises(handler.ContractError) as raised:
            handler.normalize_request({"mode": "t2v", "prompt": "x", "loras": ["not-approved"]})
        self.assertEqual(raised.exception.code, "UNKNOWN_LORA")

    def test_pixel_mode_is_explicitly_blocked_until_validated(self):
        with self.assertRaises(handler.ContractError) as raised:
            handler.normalize_request({"mode": "pixel_upscale", "prompt": "x"})
        self.assertEqual(raised.exception.code, "WORKFLOW_NOT_VALIDATED")

    def test_union_requires_video_and_depth_type(self):
        with self.assertRaises(handler.ContractError) as raised:
            handler.normalize_request({"mode": "union_control", "prompt": "x"})
        self.assertEqual(raised.exception.code, "MISSING_VIDEO")
        with self.assertRaises(handler.ContractError) as raised:
            handler.normalize_request({"mode": "union_control", "prompt": "x", "video_url": "https://example.com/a.mp4", "control_type": "canny"})
        self.assertEqual(raised.exception.code, "UNSUPPORTED_CONTROL_TYPE")

    def test_union_uses_bf16_and_official_lora(self):
        result = handler.normalize_request({
            "mode": "union_control",
            "prompt": "x",
            "video_url": "https://example.com/a.mp4",
            "frames": 9,
            "fps": 8,
        })
        self.assertEqual(result["model_profile"], "ltx25_bf16_control")
        self.assertEqual(result["loras"][0]["id"], "union_control")
        self.assertEqual(result["control_type"], "depth")
        self.assertEqual(result["control_strength"], 1.0)

    def test_motion_requires_tracks_and_normalizes_points(self):
        with self.assertRaises(handler.ContractError) as raised:
            handler.normalize_request({"mode": "motion_track", "prompt": "x", "image_base64": "a"})
        self.assertEqual(raised.exception.code, "MISSING_TRACKS")
        result = handler.normalize_request({
            "mode": "motion_track",
            "prompt": "x",
            "image_base64": "a",
            "frames": 9,
            "fps": 8,
            "tracks": [[{"x": 1.25, "y": 2.5}, {"x": 3, "y": 4}]],
        })
        self.assertEqual(result["model_profile"], "ltx25_bf16_control")
        self.assertEqual(result["loras"][0]["id"], "motion_track")
        self.assertEqual(result["tracks"][0][0], {"x": 1.25, "y": 2.5})

    def test_real_api_workflow_has_video_output(self):
        workflow = json.loads((ROOT / "workflows" / "ltx25_t2v.json").read_text())
        save_nodes = [node for node in workflow.values() if node.get("class_type") == "SaveVideo"]
        self.assertEqual(len(save_nodes), 1)
        self.assertIn("video", save_nodes[0]["inputs"])

    def test_t2v_selects_native_full_dev_profile(self):
        settings = handler.normalize_request({
            "action": "text_to_video",
            "prompt": "test",
            "duration_seconds": 2,
            "width": 704,
            "height": 1280,
            "fps": 24,
            "seed": 20260901,
            "generate_audio": True,
        })
        self.assertEqual(settings["model_profile"], "ltx25_bf16_core")
        self.assertEqual(settings["frames"], 49)
        self.assertFalse(settings["enhance_prompt"])
        self.assertEqual(
            handler.MODEL_FILES["ltx25_bf16_core"]["unet"],
            "ltx-2.5-22b-dev-transformer-bf16.safetensors",
        )
        self.assertEqual(settings["steps"], "15-step HQ res2s + 4-step distilled refinement")

    def test_native_t2v_preflight_is_explicitly_separate_from_comfy_graph(self):
        self.assertEqual(handler.NATIVE_PIPELINE_NAME, "TI2VidTwoStagesHQPipeline")
        self.assertEqual(handler.NATIVE_LTX2_COMMIT, "a95ab856bf29407b6b066ede0abe1846050db56c")
        self.assertEqual(handler.NATIVE_DISTILLED_LORA_STAGE_1, 0.25)
        self.assertEqual(handler.NATIVE_DISTILLED_LORA_STAGE_2, 0.5)


    def test_union_graph_is_depth_only_and_patched_to_bf16(self):
        workflow = json.loads((ROOT / "workflows" / "ltx25_union_control.json").read_text())
        settings = handler.normalize_request({
            "mode": "union_control",
            "prompt": "test",
            "video_base64": base64.b64encode(b"video").decode(),
            "frames": 9,
            "fps": 8,
        })
        patched = handler.patch_workflow(workflow, settings, {"video": "reference.mp4"}, "static-union")
        class_types = {node["class_type"] for node in patched.values()}
        self.assertNotIn("CannyEdgePreprocessor", class_types)
        self.assertNotIn("DWPreprocessor", class_types)
        self.assertIn("VideoDepthAnythingProcess", class_types)
        self.assertEqual(
            next(node for node in patched.values() if node["class_type"] == "UNETLoader")["inputs"]["unet_name"],
            "ltx-2.5-22b-distilled-transformer-bf16.safetensors",
        )
        lora_node = next(node for node in patched.values() if node["class_type"] == "LTXICLoRALoaderModelOnly")
        self.assertEqual(lora_node["inputs"]["lora_name"], "ltx-2.3-22b-ic-lora-union-control-ref0.5.safetensors")
        self.assertEqual(lora_node["inputs"]["strength_model"], 1.0)
        guide_node = next(node for node in patched.values() if node["class_type"] == "LTXAddVideoICLoRAGuide")
        self.assertEqual(guide_node["inputs"]["frame_idx"], 0)
        self.assertEqual(guide_node["inputs"]["crop"], "disabled")
        self.assertFalse(guide_node["inputs"]["use_tiled_encode"])
        negative_gemma = next(
            node for node in patched.values()
            if node["class_type"] == "GemmaAPITextEncode"
            and node["inputs"].get("enhance_prompt") is False
        )
        self.assertFalse(negative_gemma["inputs"]["enhance_prompt"])
        load_video = next(node for node in patched.values() if node["class_type"] == "LoadVideo")
        self.assertEqual(load_video["inputs"]["file"], "reference.mp4")

    def test_union_image_branch_is_bound_when_reference_frame_is_available(self):
        workflow = json.loads((ROOT / "workflows" / "ltx25_union_control.json").read_text())
        settings = handler.normalize_request({
            "mode": "union_control",
            "prompt": "test",
            "video_base64": base64.b64encode(b"video").decode(),
            "frames": 9,
            "fps": 8,
        })
        patched = handler.patch_workflow(
            workflow,
            settings,
            {"video": "reference.mp4", "image": "derived_first_frame.png"},
            "static-union-image",
        )
        load_image = next(node for node in patched.values() if node["class_type"] == "LoadImage")
        self.assertEqual(load_image["inputs"]["image"], "derived_first_frame.png")

        # _meta is removed by patch_workflow, so locate the surviving boolean
        # by its value and confirm the input path was not left empty.
        self.assertTrue(any(
            node.get("class_type") == "PrimitiveBoolean" and node.get("inputs", {}).get("value") is True
            for node in patched.values()
        ))
    def test_validator_accepts_comfy_dynamic_inputs(self):
        import sys
        sys.path.insert(0, str(ROOT / "scripts"))
        import validate_workflows

        schema = {
            "Source": {"input": {"required": {}}, "output": ["FLOAT"]},
            "ComfyMathExpression": {
                "input": {"required": {"expression": ["STRING", {}], "values": ["COMFY_AUTOGROW_V3", {}]}},
                "output": ["FLOAT"],
            },
        }
        graph = {
            "source": {"class_type": "Source", "inputs": {}},
            "math": {
                "class_type": "ComfyMathExpression",
                "inputs": {"expression": "a*32", "values.a": ["source", 0]},
            },
        }
        self.assertEqual(validate_workflows.validate_graph(graph, object_info=schema), [])

    def test_motion_graph_patches_tracks_and_frame_timing(self):
        workflow = json.loads((ROOT / "workflows" / "ltx25_motion_track.json").read_text())
        settings = handler.normalize_request({
            "mode": "motion_track",
            "prompt": "test",
            "image_base64": base64.b64encode(b"image").decode(),
            "frames": 9,
            "fps": 8,
            "tracks": [[{"x": 10, "y": 20}, {"x": 12, "y": 22}]],
        })
        patched = handler.patch_workflow(workflow, settings, {"image": "reference.png"}, "static-motion")
        editor = next(node for node in patched.values() if node["class_type"] == "LTXVSparseTrackEditor")
        self.assertEqual(json.loads(editor["inputs"]["points_store"]), settings["tracks"])
        self.assertEqual(editor["inputs"]["points_to_sample"], settings["frames"])
        self.assertIn(8, [node["inputs"].get("value") for node in patched.values() if node["class_type"] == "PrimitiveFloat"])
        self.assertIn(1.0, [node["inputs"].get("value") for node in patched.values() if node["class_type"] == "PrimitiveFloat"])
        save_nodes = [node for node in patched.values() if node["class_type"] == "SaveVideo"]
        self.assertEqual({node["inputs"]["filename_prefix"] for node in save_nodes}, {"ltx25/static-motion", "ltx25/static-motion_tracks"})


if __name__ == "__main__":
    unittest.main()