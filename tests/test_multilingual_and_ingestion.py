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
    def test_ocr_normalization_preserves_raw_unicode_and_isolates_noise(self):
        raw = "নির্বাচন ১৩ ফেব্রুয়ারি ২০২৬\n||||||||\nDhaka election announced"
        result = crossReff.normalize_ocr_text(raw)
        self.assertIn("নির্বাচন ১৩ ফেব্রুয়ারি ২০২৬", result["text"])
        self.assertEqual(result["rejected_lines"][0]["raw_text"], "||||||||")
        self.assertEqual(raw.splitlines()[0], result["kept_lines"][0]["raw_text"])

    def test_incomplete_headline_is_marked_not_guessed(self):
        result = crossReff.reconstruct_news_headline("১৩ ফেব্রুয়ারি ২০২৬\nসন্ধ্যা ৭টা")
        self.assertIn(result["status"], {"INCOMPLETE", "NOT_FOUND"})
        self.assertTrue(result["reasons"])
        self.assertNotIn("নির্বাচন", result["text"])

    def test_numbers_and_fused_ocr_do_not_make_complete_headline(self):
        noisy = "চি ve 9 Lens ৪০ এ ৩৮ ৪১৮ ১\nRaiThousands rallied KolkatayIndiagin 2"
        result = crossReff.reconstruct_news_headline(noisy)
        self.assertIn(result["status"], {"INCOMPLETE", "NOT_FOUND"})

    def test_mixed_garbage_line_is_not_joined_to_bangla_headline(self):
        text = "নির্বাচনের সময় সন্ধ্যা ৭টা\n১৩ ফেব্রুয়ারি ২০২৬\nco ENT EDU উঠ 1 eee"
        result = crossReff.reconstruct_news_headline(text)
        self.assertNotIn("ENT EDU", result["text"])

    def test_date_location_event_headline_retains_anchors(self):
        text = "ঢাকায় নির্বাচন অনুষ্ঠিত হবে ১৩ ফেব্রুয়ারি ২০২৬"
        result = crossReff.reconstruct_news_headline(text)
        self.assertEqual(result["status"], "COMPLETE")
        self.assertIn("ঢাকায়", result["text"])
        self.assertIn("২০২৬", result["text"])
        self.assertEqual(result["source_lines"], [text])

    def test_bangla_date_on_adjacent_line_stays_with_election_claim(self):
        text = "নির্বাচনের সময় : সন্ব্যা ৭টা\n১৩ ফেব্রুয়ারি ২০২৬"
        claims = crossReff.extract_claims(text, "")
        self.assertTrue(claims)
        self.assertIn("১৩ ফেব্রুয়ারি ২০২৬", claims[0]["text"])
        self.assertIn("১৩ ফেব্রুয়ারি ২০২৬", claims[0]["entities"]["dates"])
        queries = crossReff.build_search_queries(claims[0]["text"], claims[0]["entities"])
        self.assertIn("১৩ ফেব্রুয়ারি ২০২৬", queries[0])

    def test_bangla_reconstruction_keeps_raw_normalized_headline_and_date_separate(self):
        raw = "NTV\nনির্বাচনের সময় : সন্ধ্যা ৭টা\n১৩ ফেব্রুয়ারি ২০২৬\n|||"
        normalized = crossReff.normalize_ocr_text(raw)
        headline = crossReff.reconstruct_news_headline(normalized["text"])
        self.assertEqual(raw.splitlines()[1], normalized["kept_lines"][1]["raw_text"])
        self.assertEqual(headline["status"], "COMPLETE")
        self.assertIn("১৩ ফেব্রুয়ারি ২০২৬", headline["text"])
        self.assertEqual(headline["date_text"], "১৩ ফেব্রুয়ারি ২০২৬")

    def test_publisher_requires_exact_ocr_branding_and_can_remain_unknown(self):
        unknown = crossReff.detect_publisher("বাংলা সংবাদ নির্বাচনের সময়")
        self.assertFalse(unknown["detected"])
        self.assertEqual(unknown["method"], "NO_RELIABLE_PUBLISHER_CLUE")
        # A substring inside another Latin word must not establish NTV.
        self.assertFalse(crossReff.detect_publisher("governmentv report")["detected"])
        detected = crossReff.detect_publisher(
            "বাংলা সংবাদ", [{"name": "logo_region", "raw_text": "NTV"}])
        self.assertTrue(detected["detected"])
        self.assertEqual(detected["domain"], "ntvbd.com")
        self.assertEqual(detected["evidence"][0]["ocr_pass"], "logo_region")

    def test_bangla_claim_corroboration_retains_trusted_source_evidence(self):
        claims = crossReff.extract_claims(
            "ঢাকায় নির্বাচন অনুষ্ঠিত হবে ১৩ ফেব্রুয়ারি ২০২৬", "")[:1]
        search_results = {
            "status": "VERIFIED_RESULTS_FOUND", "requests_made": 1,
            "results": [
                {"title": "নির্বাচন ১৩ ফেব্রুয়ারি ২০২৬", "snippet": "ঢাকায় নির্বাচন অনুষ্ঠিত হবে",
                 "url": "https://reuters.com/bn/election", "domain": "reuters.com"},
                {"title": "ঢাকার নির্বাচন", "snippet": "১৩ ফেব্রুয়ারি ২০২৬ ভোট",
                 "url": "https://apnews.com/election", "domain": "apnews.com"},
            ],
        }
        def corroborated(claim, evidence_text, evidence_source, evidence_url, source_type="trusted_search"):
            return {"source": evidence_source, "source_type": source_type, "url": evidence_url,
                    "trusted_domain": True, "source_trust": {"trusted": True, "rationale": "allowlist"},
                    "matched_claim": claim["text"], "final_verdict": "CORROBORATED",
                    "overlap_score": 90, "text_verdict": "CORROBORATED", "keyword_overlap": 1,
                    "sequence_similarity": 1, "entity_matches": [], "entity_mismatches": []}
        with patch.object(crossReff, "search_trusted_sources", return_value=search_results), \
             patch.object(crossReff, "compare_claim_to_evidence", side_effect=corroborated):
            result = crossReff.verify_claims(claims)
        item = result["claim_results"][0]
        self.assertEqual(item["verdict"], "VERIFIED")
        self.assertEqual(item["evidence_comparisons"][0]["matched_claim"], claims[0]["text"])
        self.assertTrue(item["evidence_comparisons"][0]["source_trust"]["trusted"])
        self.assertIn("https://reuters.com/bn/election",
                      [source["url"] for source in result["source_results"]])

    def test_bangla_claim_contradiction_is_retained_separately(self):
        claims = crossReff.extract_claims(
            "ঢাকায় নির্বাচন অনুষ্ঠিত হবে ১৩ ফেব্রুয়ারি ২০২৬", "")[:1]
        search_results = {"status": "VERIFIED_RESULTS_FOUND", "requests_made": 1, "results": [
            {"title": "নির্বাচনের তারিখ পরিবর্তন", "snippet": "১৩ ফেব্রুয়ারি ভোট হবে না",
             "url": "https://reuters.com/bn/correction", "domain": "reuters.com"}]}
        def contradicted(claim, evidence_text, evidence_source, evidence_url, source_type="trusted_search"):
            return {"source": evidence_source, "source_type": source_type, "url": evidence_url,
                    "trusted_domain": True, "source_trust": {"trusted": True, "rationale": "allowlist"},
                    "matched_claim": claim["text"], "final_verdict": "CONTRADICTED",
                    "overlap_score": 90, "text_verdict": "CONTRADICTED", "keyword_overlap": 1,
                    "sequence_similarity": 1, "entity_matches": [], "entity_mismatches": ["dates"]}
        with patch.object(crossReff, "search_trusted_sources", return_value=search_results), \
             patch.object(crossReff, "compare_claim_to_evidence", side_effect=contradicted):
            result = crossReff.verify_claims(claims)
        self.assertEqual(result["search_status"], "CONTRADICTED")
        self.assertEqual(result["claim_results"][0]["verdict"], "DEBUNKED")

    def test_successful_search_with_no_usable_results_is_not_search_failure(self):
        claims = crossReff.extract_claims(
            "ঢাকায় নির্বাচন অনুষ্ঠিত হবে ১৩ ফেব্রুয়ারি ২০২৬", "")[:1]
        empty = {"status": "INSUFFICIENT_EVIDENCE", "results": [], "requests_made": 1}
        with patch.object(crossReff, "search_trusted_sources", return_value=empty):
            result = crossReff.verify_claims(claims)
        item = result["claim_results"][0]
        self.assertEqual(item["verdict"], "INSUFFICIENT_EVIDENCE")
        self.assertEqual(item["search_status"], "INSUFFICIENT_EVIDENCE")
        self.assertEqual(item["structured_errors"], [])

    def test_low_quality_ocr_is_not_promoted_to_claim(self):
        claims, stats = crossReff.extract_claims(
            "x | ? 22", "", return_stats=True, ocr_result={"quality_score": 12})
        self.assertEqual(claims, [])
        self.assertEqual(stats["status"], "OCR_TOO_POOR")

    def test_search_trace_preserves_queries_results_and_failure(self):
        claims = crossReff.extract_claims("Officials announced an election in Dhaka today.", "")[:1]
        unavailable = {"status": "SEARCH_FAILED", "results": [], "requests_made": 1,
                       "error": "offline"}
        with patch.object(crossReff, "search_trusted_sources", return_value=unavailable):
            result = crossReff.verify_claims(claims)
        item = result["claim_results"][0]
        self.assertEqual(item["verdict"], "SEARCH_FAILED")
        self.assertEqual(item["search_trace"][0]["error"], "offline")
        self.assertEqual(item["source_text"], claims[0]["source_text"])

    def test_mocked_verified_and_contradicted_claims_are_distinct(self):
        claim = crossReff.extract_claims("Officials announced an election in Dhaka today.", "")[:1]
        def search(_query, **_kwargs):
            return {"status": "VERIFIED_RESULTS_FOUND", "requests_made": 1, "results": [
                {"title": "Report", "snippet": "Details", "url": "https://reuters.com/a", "domain": "reuters.com"},
                {"title": "Report", "snippet": "Details", "url": "https://apnews.com/a", "domain": "apnews.com"}]}
        def comparison(claim, evidence_text, evidence_source, evidence_url, source_type="trusted_search"):
            source, url = evidence_source, evidence_url
            verdict = "CONTRADICTED" if source == "reuters.com" else "CORROBORATED"
            return {"source": source, "source_type": source_type, "url": url,
                    "trusted_domain": True, "final_verdict": verdict,
                    "overlap_score": 90, "text_verdict": verdict,
                    "keyword_overlap": 1, "sequence_similarity": 1,
                    "entity_matches": [], "entity_mismatches": []}
        with patch.object(crossReff, "search_trusted_sources", side_effect=search), \
             patch.object(crossReff, "compare_claim_to_evidence", side_effect=comparison):
            conflict = crossReff.verify_claims(claim)
        self.assertEqual(conflict["claim_results"][0]["verdict"], "CONFLICTING_EVIDENCE")
        self.assertTrue(conflict["source_results"])

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
