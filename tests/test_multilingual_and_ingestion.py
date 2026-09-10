"""Deterministic regressions for multilingual image claims and unified ingestion."""
import os
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image

from agent import crossReff
from agent.ingestor import classify_input, run_ingestor_agent
from agent.provinence import calculate_provenance_score


BANGLA_HEADLINE = "ঢাকার দুই সিটিতে ত্রিমুখী লড়াই দেখার অপেক্ষায় রাজধানীবাসী"


class MultilingualCrossReferenceTests(unittest.TestCase):
    def test_quality_prefers_meaningful_text_over_long_garbage(self):
        meaningful = "Officials announced an election in Dhaka on 12 June 2026."
        garbage = "#@$% " * 100
        self.assertGreater(crossReff._ocr_quality(meaningful)["score"],
                           crossReff._ocr_quality(garbage)["score"])

    def test_duplicate_ocr_lines_keep_unicode(self):
        merged = crossReff._merge_ocr_passes([BANGLA_HEADLINE, BANGLA_HEADLINE])
        self.assertEqual(merged, BANGLA_HEADLINE)

    def test_mixed_search_keeps_original_unicode(self):
        text = f"{BANGLA_HEADLINE} DHAKA 2026"
        queries = crossReff.build_search_queries(text, crossReff.extract_entities(text))
        self.assertEqual(queries[0], text)
        self.assertTrue(any("ঢাকা" in query for query in queries))

    def test_missing_visual_database_is_not_checked_not_not_applicable(self):
        result = crossReff.check_visual_similarity("unused.png", "image", known_hash_db={})
        self.assertFalse(result["checked"])
        self.assertEqual(result["verdict"], "NOT_CHECKED")
        self.assertEqual(result["module_status"], "NOT_CHECKED")
        self.assertEqual(result["reason"], "NO_REFERENCE_DATABASE")

    def test_bangla_claim_is_unicode_safe_and_fact_checkable(self):
        claims, stats = crossReff.extract_claims(BANGLA_HEADLINE, "", return_stats=True)
        self.assertGreater(stats["ocr_claim_candidates"], 0)
        self.assertGreater(stats["reliable_claims"], 0)
        self.assertEqual(claims[0]["language"], "bn")
        self.assertEqual(claims[0]["original_ocr_text"], BANGLA_HEADLINE)
        self.assertIn("ঢাকা", claims[0]["entities"]["locations"])

    def test_mixed_bangla_english_and_duplicate_claims(self):
        mixed = f"NTV Online\n{BANGLA_HEADLINE}\n{BANGLA_HEADLINE}"
        claims, stats = crossReff.extract_claims(mixed, "", return_stats=True)
        self.assertEqual(len(claims), 1)
        self.assertEqual(claims[0]["language"], "bn")
        self.assertIn("NTV Online", crossReff.extract_entities(mixed)["organizations"])
        self.assertGreater(stats["filtered_claims"], 0)

    def test_english_claim_still_works(self):
        claims = crossReff.extract_claims("Officials announced an election in Dhaka today.", "")
        self.assertTrue(claims)
        self.assertEqual(claims[0]["language"], "en")

    def test_ocr_reports_missing_bangla_pack(self):
        with patch.object(crossReff, "_available_tesseract_languages", return_value=["eng"]), \
             patch.object(crossReff, "_prepare_image_ocr_variants", return_value=[("original", object(), "--psm 3")]), \
             patch.object(crossReff.pytesseract, "image_to_string", return_value="NTV Online report from Dhaka") as ocr:
            result = crossReff.extract_text_via_ocr("unused.png", "image")
        self.assertEqual(result["module_status"], "partial")
        self.assertIn("ben", result["missing_languages"])
        self.assertTrue(any("OCR_LANGUAGE_UNAVAILABLE" in flag for flag in result["flags"]))
        self.assertEqual(ocr.call_args.kwargs["lang"], "eng")

    def test_mixed_language_ocr_uses_ben_plus_eng(self):
        with patch.object(crossReff, "_available_tesseract_languages", return_value=["eng", "ben"]), \
             patch.object(crossReff, "_prepare_image_ocr_variants", return_value=[("headline", object(), "--psm 11")]), \
             patch.object(crossReff.pytesseract, "image_to_string", return_value=f"NTV Online {BANGLA_HEADLINE}") as ocr:
            result = crossReff.extract_text_via_ocr("unused.png", "image")
        self.assertEqual(result["language_detected"], "bn+en")
        self.assertEqual(ocr.call_args.kwargs["lang"], "ben+eng")

    def test_meaningful_bangla_claim_reaches_external_search(self):
        claims = crossReff.extract_claims(BANGLA_HEADLINE, "")
        failed_search = {"status": "SEARCH_FAILED", "results": [], "requests_made": 1}
        with patch.object(crossReff, "search_trusted_sources", return_value=failed_search):
            result = crossReff.verify_claims(claims)
        self.assertEqual(result["claims_checked"], 1)
        self.assertGreater(result["external_searches_performed"], 0)
        self.assertIn(result["claim_results"][0]["verdict"], {"SEARCH_FAILED", "INSUFFICIENT_EVIDENCE"})


class IngestionTests(unittest.TestCase):
    def test_five_input_classification(self):
        with tempfile.TemporaryDirectory() as temporary:
            image = Path(temporary) / "local.jpg"
            video = Path(temporary) / "local.mp4"
            image.write_bytes(b"x")
            video.write_bytes(b"x")
            self.assertEqual(classify_input(str(image)), "local_file")
            self.assertEqual(classify_input(str(video)), "local_file")
        self.assertEqual(classify_input("https://example.org/photo.jpg?size=large"), "direct_image_url")
        self.assertEqual(classify_input("https://youtu.be/abc"), "youtube_url")
        self.assertEqual(classify_input("https://www.facebook.com/public/posts/1"), "facebook_post_url")
        self.assertEqual(classify_input("not a url"), "unsupported")

    def test_local_source_bytes_and_normalized_hash_are_distinct(self):
        with tempfile.TemporaryDirectory() as temporary:
            previous = Path.cwd()
            try:
                os.chdir(temporary)
                source = Path(temporary) / "photo.jpg"
                Image.new("RGB", (32, 24), "red").save(source, "JPEG")
                result = run_ingestor_agent(str(source))
            finally:
                os.chdir(previous)
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["source_file_path"], str(source.resolve()))
        self.assertNotEqual(result["source_file_hash"], result["normalized_file_hash"])
        self.assertEqual(result["file_hash"], result["source_file_hash"])

    def test_missing_metadata_and_c2pa_are_score_neutral(self):
        score, _ = calculate_provenance_score(
            [{"flag": "NO_EXIF: absent", "confidence": "LOW_CONFIDENCE"}],
            {"c2pa_status": "NO_C2PA"},
            {"hash_match": False},
            {"hash_mismatch": False},
        )
        self.assertEqual(score, 50)

    def test_direct_image_url_validates_bytes_and_preserves_source(self):
        payload = io.BytesIO()
        Image.new("RGB", (20, 20), "blue").save(payload, "PNG")

        class Response:
            headers = {"content-length": str(len(payload.getvalue()))}
            def raise_for_status(self):
                return None
            def iter_content(self, chunk_size=8192):
                yield payload.getvalue()

        with tempfile.TemporaryDirectory() as temporary:
            previous = Path.cwd()
            try:
                os.chdir(temporary)
                with patch("agent.ingestor.requests.get", return_value=Response()):
                    result = run_ingestor_agent("https://example.org/news.jpg")
                source_exists = Path(result["source_file_path"]).exists()
                normalized_exists = Path(result["normalized_file_path"]).exists()
            finally:
                os.chdir(previous)
        self.assertEqual(result["input_type"], "direct_image_url")
        self.assertEqual(result["image_info"]["detected_format"], "PNG")
        self.assertTrue(source_exists and normalized_exists)

    def test_corrupt_image_fails_structurally(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "broken.jpg"
            source.write_bytes(b"not an image")
            result = run_ingestor_agent(str(source))
        self.assertEqual(result["status"], "error")
        self.assertFalse(result["ready_for_next_agent"])
        self.assertEqual(result["error_code"], "INGESTION_FAILED")


if __name__ == "__main__":
    unittest.main()
