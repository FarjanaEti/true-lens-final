from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

from PIL import Image

import image_pipeline


class CanonicalLifecycleTests(unittest.TestCase):
    def _make_source(self, directory: str, name: str = "upload.jpg") -> Path:
        source = Path(directory) / name
        Image.new("RGB", (24, 16), "navy").save(source, "JPEG")
        return source

    def _successful_agents(self, observed: list[Path], lock=None):
        def remember(path: Path) -> None:
            self.assertTrue(path.is_file())
            self.assertGreater(len(path.read_bytes()), 0)
            if lock is None:
                observed.append(path)
            else:
                with lock:
                    observed.append(path)

        def provenance(ingestor):
            canonical = Path(ingestor["normalized_file_path"])
            remember(canonical)
            return {
                "status": "success",
                "ready_for_next_agent": True,
                "file_path": str(canonical),
                "source_file_path": ingestor["source_file_path"],
                "media_type": ingestor["media_type"],
            }

        def forensics(provenance_output):
            canonical = Path(provenance_output["file_path"])
            remember(canonical)
            return {
                "status": "success",
                "ready_for_next_agent": True,
                "file_path": str(canonical),
                "media_type": provenance_output["media_type"],
                "media_authenticity": "UNCERTAIN",
                "confidence": 50,
            }

        def cross_reference(provenance_output, forensics_output):
            canonical = Path(forensics_output["file_path"])
            self.assertEqual(canonical, Path(provenance_output["file_path"]))
            remember(canonical)
            return {
                "status": "completed",
                "claims_checked": 0,
                "file_path": str(canonical),
            }

        def decision(ingestor, provenance_output, forensics_output, crossref_output):
            canonical = Path(ingestor["normalized_file_path"])
            self.assertEqual(canonical, Path(provenance_output["file_path"]))
            self.assertEqual(canonical, Path(forensics_output["file_path"]))
            self.assertEqual(canonical, Path(crossref_output["file_path"]))
            remember(canonical)
            return {
                "category": "Needs Review",
                "decision_state": {},
            }

        return provenance, forensics, cross_reference, decision

    def test_canonical_is_available_to_every_stage_then_cleaned(self):
        observed = []
        agents = self._successful_agents(observed)
        with tempfile.TemporaryDirectory() as upload_dir:
            source = self._make_source(upload_dir)
            original = source.read_bytes()
            with patch.object(image_pipeline, "run_provenance_agent", side_effect=agents[0]), \
                    patch.object(image_pipeline, "run_forensics_agent", side_effect=agents[1]), \
                    patch.object(image_pipeline, "run_crossref_agent", side_effect=agents[2]), \
                    patch.object(image_pipeline, "run_decision_engine", side_effect=agents[3]):
                result = image_pipeline.run_local_image_pipeline(str(source))

            canonical = Path(result["agents"]["ingestor"]["normalized_file_path"])
            self.assertEqual(observed, [canonical, canonical, canonical, canonical])
            self.assertFalse(canonical.exists())
            self.assertFalse(canonical.parent.exists())
            self.assertTrue(source.exists())
            self.assertEqual(source.read_bytes(), original)

    def test_cleanup_happens_when_a_downstream_agent_raises(self):
        observed = []

        def provenance(ingestor):
            canonical = Path(ingestor["normalized_file_path"])
            observed.append(canonical)
            self.assertTrue(canonical.exists())
            return {
                "status": "success",
                "ready_for_next_agent": True,
                "file_path": str(canonical),
                "source_file_path": ingestor["source_file_path"],
                "media_type": "image",
            }

        def fail_forensics(provenance_output):
            self.assertTrue(Path(provenance_output["file_path"]).exists())
            raise RuntimeError("forced downstream failure")

        with tempfile.TemporaryDirectory() as upload_dir:
            source = self._make_source(upload_dir)
            original = source.read_bytes()
            with patch.object(image_pipeline, "run_provenance_agent", side_effect=provenance), \
                    patch.object(image_pipeline, "run_forensics_agent", side_effect=fail_forensics):
                with self.assertRaisesRegex(RuntimeError, "forced downstream failure"):
                    image_pipeline.run_pipeline(str(source))

            self.assertEqual(len(observed), 1)
            self.assertFalse(observed[0].exists())
            self.assertFalse(observed[0].parent.exists())
            self.assertTrue(source.exists())
            self.assertEqual(source.read_bytes(), original)

    def test_repeated_analyses_leave_no_generated_canonical_files(self):
        observed = []
        agents = self._successful_agents(observed)
        with tempfile.TemporaryDirectory() as upload_dir:
            source = self._make_source(upload_dir)
            with patch.object(image_pipeline, "run_provenance_agent", side_effect=agents[0]), \
                    patch.object(image_pipeline, "run_forensics_agent", side_effect=agents[1]), \
                    patch.object(image_pipeline, "run_crossref_agent", side_effect=agents[2]), \
                    patch.object(image_pipeline, "run_decision_engine", side_effect=agents[3]):
                for _ in range(3):
                    image_pipeline.run_pipeline(str(source))

        canonical_paths = set(observed)
        self.assertEqual(len(canonical_paths), 3)
        self.assertTrue(all(not path.exists() for path in canonical_paths))
        self.assertTrue(all(not path.parent.exists() for path in canonical_paths))

    def test_concurrent_analyses_use_isolated_directories(self):
        observed = []
        lock = threading.Lock()
        agents = self._successful_agents(observed, lock)
        with tempfile.TemporaryDirectory() as upload_dir:
            sources = [self._make_source(upload_dir, f"upload-{index}.jpg") for index in range(4)]
            originals = {source: source.read_bytes() for source in sources}
            with patch.object(image_pipeline, "run_provenance_agent", side_effect=agents[0]), \
                    patch.object(image_pipeline, "run_forensics_agent", side_effect=agents[1]), \
                    patch.object(image_pipeline, "run_crossref_agent", side_effect=agents[2]), \
                    patch.object(image_pipeline, "run_decision_engine", side_effect=agents[3]):
                with ThreadPoolExecutor(max_workers=4) as executor:
                    results = list(executor.map(image_pipeline.run_pipeline, map(str, sources)))

            canonical_paths = [
                Path(result["agents"]["ingestor"]["normalized_file_path"])
                for result in results
            ]
            self.assertEqual(len({path.parent for path in canonical_paths}), len(sources))
            self.assertTrue(all(not path.exists() for path in canonical_paths))
            self.assertTrue(all(not path.parent.exists() for path in canonical_paths))
            for source, original in originals.items():
                self.assertTrue(source.exists())
                self.assertEqual(source.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
