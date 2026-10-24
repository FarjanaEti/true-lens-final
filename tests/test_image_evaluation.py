import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("run_image_evaluation", ROOT / "tests" / "run_image_evaluation.py")
evaluation = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(evaluation)


class ImageEvaluationTests(unittest.TestCase):
    def test_manifest_never_turns_filename_into_ground_truth(self):
        manifest = json.loads(evaluation.MANIFEST_PATH.read_text(encoding="utf-8"))
        fixtures = {item["filename"]: item for item in manifest["fixtures"]}
        self.assertEqual(fixtures["images.jpeg"]["label_status"], "unverified")
        self.assertIsNone(fixtures["images.jpeg"]["category"])
        self.assertEqual(fixtures["newsntvfake.jpg"]["label_status"], "unverified")
        self.assertIsNone(fixtures["newsntvfake.jpg"]["ground_truth"]["source_url"])

    def test_sha256_file_is_deterministic(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "fixture.bin"
            path.write_bytes(b"TrueLens")
            self.assertEqual(
                evaluation.sha256_file(path),
                "dc1c44166046c7833a106718142a614c5eeca7bb5c79f9f3112780ee8ec3838e",
            )

    def test_summary_preserves_queries_and_decision_trigger(self):
        result = {
            "status": "completed",
            "agents": {
                "ingestor": {"status": "completed"},
                "provenance": {},
                "forensics": {},
                "cross_reference": {
                    "claim_results": [{"claim": "x", "search_queries": ["x"], "search_trace": [{"provider": "test"}]}],
                    "external_searches_performed": 1,
                    "search_status": "COMPLETED",
                },
            },
            "final_decision": {"category": "Needs Review", "decision_state": {"baseline_trigger": "weak_evidence"}},
        }
        summary = evaluation.summarize_pipeline(result)
        self.assertEqual(summary["cross_reference"]["queries"], ["x"])
        self.assertEqual(summary["cross_reference"]["search_traces"], [{"provider": "test"}])
        self.assertEqual(summary["decision"]["trigger"], "weak_evidence")


if __name__ == "__main__":
    unittest.main()
