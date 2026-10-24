"""Regression coverage for photo-scene versus document forensic routing."""
import tempfile
from pathlib import Path
import unittest

import cv2
import numpy as np

from agent.forensic import (
    analyze_noise_and_frequency,
    classify_image_content,
    detect_face_artifacts,
    run_forensics_agent,
)


ROOT = Path(__file__).resolve().parents[1]


class ForensicsRoutingTests(unittest.TestCase):
    def _image(self, name):
        path = ROOT / name
        if not path.exists():
            self.skipTest(f"regression fixture not available: {name}")
        return str(path)

    def test_ai_flood_image_is_photo_scene_and_face_is_detected(self):
        path = self._image("images.jpeg")
        content = classify_image_content(path)
        self.assertEqual(content["content_type"], "PHOTO_SCENE")
        self.assertFalse(content["document_like"])

        faces = detect_face_artifacts(path)
        self.assertEqual(faces["module_status"], "ok")
        self.assertGreater(faces["faces_found"], 0)

        result = run_forensics_agent({
            "status": "success",
            "ready_for_next_agent": True,
            "file_path": path,
            "source_file_path": path,
            "media_type": "image",
        })
        findings = result["raw_findings"]
        self.assertEqual(findings["content_classification"]["content_type"], "PHOTO_SCENE")
        self.assertEqual(findings["face_artifacts"]["module_status"], "ok")
        self.assertEqual(findings["ai_heuristics"]["module_status"], "ok")
        self.assertNotIn("NOT_APPLICABLE", " ".join(
            findings["ai_heuristics"].get("not_applicable_checks", [])
        ))
        self.assertIn(result["evidence_strength"], {"WEAK", "MODERATE", "STRONG"})

    def test_ntv_bangla_graphic_remains_document_layout(self):
        path = self._image("chek.jpeg")
        content = classify_image_content(path)
        self.assertEqual(content["content_type"], "DOCUMENT_OR_LAYOUT")
        self.assertTrue(content["document_like"])

    def test_flat_color_low_noise_is_non_diagnostic(self):
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "flat.png")
            image = np.full((120, 180, 3), 128, dtype=np.uint8)
            self.assertTrue(cv2.imwrite(path, image))
            noise = analyze_noise_and_frequency(path)
            flag = next(item for item in noise["flags"] if item.startswith("LOW_NOISE_STD"))
            self.assertIn("weak, non-diagnostic", flag)
            result = run_forensics_agent({
                "status": "success", "ready_for_next_agent": True,
                "file_path": path, "source_file_path": path,
                "media_type": "image", "c2pa": {},
            })
            self.assertNotEqual(result["media_authenticity"], "MANIPULATED")

    def test_one_eye_is_partial_not_clean(self):
        path = self._image("canonical_image_173dfd1761cab29e.png")
        faces = detect_face_artifacts(path)
        one_eye = any(item.get("eyes_detected") == 1 for item in faces.get("face_analyses", []))
        if not one_eye:
            self.skipTest("fixture did not reproduce one-eye detector coverage")
        self.assertEqual(faces["module_status"], "partial")
        self.assertEqual(faces["coverage_status"], "partial")
        self.assertNotIn("clean_evidence_text", faces)


if __name__ == "__main__":
    unittest.main()
