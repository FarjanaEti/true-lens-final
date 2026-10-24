"""Focused C2PA byte-flow and validation regressions."""
import hashlib
from pathlib import Path
import tempfile
import unittest

from agent.ingestor import run_ingestor_agent
from agent.provinence import check_c2pa_signature, run_provenance_agent
from decision_engine import run_decision_engine


ROOT = Path(__file__).resolve().parents[1]
PUBLIC_TEST_ASSET = ROOT / "adobe-20220124-C.jpg"


class C2PAProvenanceTests(unittest.TestCase):
    def test_original_jpeg_is_verified_not_normalized_derivative(self):
        ingestor = run_ingestor_agent(str(PUBLIC_TEST_ASSET))
        self.assertEqual(ingestor["source_file_path"], str(PUBLIC_TEST_ASSET))
        self.assertTrue(ingestor["normalized_file_path"].endswith(".png"))
        self.assertNotEqual(ingestor["source_file_hash"], ingestor["normalized_file_hash"])
        provenance = run_provenance_agent(ingestor)
        self.assertEqual(provenance["source_file_path"], str(PUBLIC_TEST_ASSET))
        self.assertEqual(provenance["integrity_check"]["recomputed_hash"],
                         ingestor["source_file_hash"])
        c2pa = provenance["c2pa"]
        self.assertTrue(c2pa["signature_valid"])
        self.assertTrue(c2pa["asset_binding_valid"])
        self.assertFalse(c2pa["credential_trusted"])
        self.assertTrue(c2pa["c2pa_valid"])
        self.assertTrue(c2pa["applicable_verification"])
        self.assertEqual(c2pa["c2pa_status"], "C2PA_VALID_UNTRUSTED")
        self.assertIn("signingCredential.untrusted",
                      {item["code"] for item in c2pa["validation_failures"]})
        decision = run_decision_engine(
            ingestor, provenance,
            {"status": "success", "ready_for_next_agent": True,
             "media_authenticity": "UNCERTAIN", "confidence": 50,
             "explainability": []},
            {"status": "completed", "claims_checked": 0},
        )
        self.assertEqual(decision["media_authenticity_assessment"]["status"], "SUPPORTED_AUTHENTIC")
        self.assertEqual(decision["confidence"], "High")
        self.assertGreaterEqual(decision["likelihood"]["authentic"], 80)

    def test_manifest_stripped_derivative_loses_supported_authentic_state(self):
        ingestor = run_ingestor_agent(str(PUBLIC_TEST_ASSET))
        stripped_ingestor = {
            **ingestor,
            "source_file_path": ingestor["normalized_file_path"],
            "source_file_hash": ingestor["normalized_file_hash"],
            "file_hash": ingestor["normalized_file_hash"],
        }
        provenance = run_provenance_agent(stripped_ingestor)
        decision = run_decision_engine(
            stripped_ingestor, provenance,
            {"status": "success", "ready_for_next_agent": True,
             "media_authenticity": "UNCERTAIN", "confidence": 50,
             "explainability": []},
            {"status": "completed", "claims_checked": 0},
        )
        self.assertEqual(provenance["c2pa"]["c2pa_status"], "NO_C2PA")
        self.assertNotEqual(decision["media_authenticity_assessment"]["status"], "SUPPORTED_AUTHENTIC")
        self.assertLess(decision["likelihood"]["authentic"], 80)

    def test_absent_and_tampered_credentials_are_not_verified(self):
        absent = ROOT / "c2pa_adobe_positive.jpg"
        self.assertEqual(check_c2pa_signature(str(absent))["c2pa_status"], "NO_C2PA")
        with tempfile.TemporaryDirectory() as directory:
            tampered = Path(directory) / "tampered.jpg"
            data = bytearray(PUBLIC_TEST_ASSET.read_bytes())
            data[-100] ^= 1
            tampered.write_bytes(data)
            result = check_c2pa_signature(str(tampered))
            self.assertFalse(result["applicable_verification"])
            self.assertNotEqual(result["c2pa_status"], "C2PA_VERIFIED")

    def test_public_fixture_hash_is_pinned(self):
        digest = hashlib.sha256(PUBLIC_TEST_ASSET.read_bytes()).hexdigest()
        self.assertEqual(digest, "75a8da33f6eaf1e16bf3b42cd78913b22b2e6a671fda217a508b1ba4230ce864")


if __name__ == "__main__":
    unittest.main()
