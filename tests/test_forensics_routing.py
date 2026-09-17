"""Regression coverage for photo-scene versus document forensic routing."""
from pathlib import Path
import unittest

from agent.forensic import (
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


if __name__ == "__main__":
    unittest.main()
