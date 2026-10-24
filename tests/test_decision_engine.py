"""Decision-policy regression tests. Run: python -m unittest discover -s tests."""
import copy
import json
import unittest
from decision_engine import run_decision_engine


class DecisionTests(unittest.TestCase):
    def setUp(self):
        self.i = {'status': 'success', 'ready_for_next_agent': True, 'media_type': 'image'}
        self.p = {'status': 'success', 'integrity_check': {'hash_mismatch': False}}
        self.f = {'status': 'success', 'media_authenticity': 'UNCERTAIN', 'confidence': 65}
        self.c = {'status': 'completed', 'claims_checked': 0}

    def decide(self):
        result = run_decision_engine(self.i, self.p, self.f, self.c)
        self.assertEqual(sum(result['likelihood'].values()), 100)
        self.assertTrue(all(type(n) is int and 0 <= n <= 100 for n in result['likelihood'].values()))
        self.assertIs(result['is_calibrated_probability'], False)
        self.assertEqual(len(result['agent_summary']), 4)
        json.dumps(result, allow_nan=False)
        return result

    def sign(self):
        self.p['c2pa'] = {
            'c2pa_present': True, 'c2pa_valid': True,
            'c2pa_status': 'C2PA_VERIFIED', 'verification_status': 'VALID',
            'applicable_verification': True, 'signature_valid': True,
            'asset_binding_valid': True, 'credential_trusted': True,
        }

    def claim(self, verdict='DEBUNKED', sources=2, trusted=True):
        self.c.update(claims_checked=1, claim_results=[{
            'claim': 'This photograph was taken today in Dhaka.',
            'is_fact_checkable': True, 'verdict': verdict, 'confidence': .85,
            'evidence_comparisons': [dict(source=f'source{n}.org', url=f'https://source{n}.org/report', trusted_domain=trusted,
                                          final_verdict='CORROBORATED' if verdict == 'VERIFIED' else 'CONTRADICTED') for n in range(sources)]}])

    def test_unknown_and_weak_flags_are_neutral(self):
        baseline = self.decide()
        self.assertTrue(baseline['decision_state']['used_inconclusive_baseline'])
        self.assertEqual(
            baseline['decision_state']['baseline_trigger'],
            'NO_DECISIVE_PROVENANCE_FORENSIC_OR_CONTEXTUAL_EVIDENCE')
        self.p.update(c2pa=None, metadata={'exif_status': 'MISSING'}, provenance_score=1)
        self.f.update(all_flags=['NO_FACE', 'HIGH_ELA_MEAN', 'DOCUMENT_LAYOUT'], forensics_score=1)
        self.c.update(crossref_score=99, modules={'ocr': 'error'}, visual_similarity={'verdict': 'NOT_CHECKED'})
        self.assertEqual(self.decide()['likelihood'], baseline['likelihood'])
        self.assertEqual(self.decide()['category'], 'Needs Review')

    def test_no_c2pa_but_otherwise_inconclusive_uses_neutral_baseline(self):
        self.p.update(
            c2pa={'c2pa_present': False, 'c2pa_valid': False,
                  'c2pa_status': 'NO_C2PA'},
            provenance_score=47, provenance_confidence='LOW',
            hash_matches_ingestor=True,
        )
        result = self.decide()
        self.assertEqual(result['category'], 'Needs Review')
        self.assertGreater(result['likelihood']['authentic'], result['likelihood']['manipulated'])
        self.assertEqual(result['decision_state']['baseline_trigger'],
                         'NO_DECISIVE_PROVENANCE_FORENSIC_OR_CONTEXTUAL_EVIDENCE')
        self.assertTrue(result['decision_state']['integrity_valid'])

    def test_no_c2pa_with_credible_corroboration_keeps_media_unresolved(self):
        self.p['c2pa'] = {'c2pa_present': False, 'c2pa_valid': False,
                          'c2pa_status': 'NO_C2PA'}
        self.c.update(
            publisher={'detected': True, 'domain': 'publisher.example'},
            source_matches=[{
                'source_type': 'official_publisher_search',
                'source': 'publisher.example', 'final_verdict': 'CORROBORATED',
                'entity_mismatches': [],
            }],
        )
        result = self.decide()
        self.assertEqual(result['category'], 'Needs Review')
        self.assertEqual(result['claim_context_assessment']['status'], 'CORROBORATED')
        self.assertEqual(result['media_authenticity_assessment']['status'], 'INCONCLUSIVE')
        self.assertEqual(result['decision_state']['baseline_trigger'],
                         'CONTEXT_CORROBORATED_MEDIA_AUTHENTICITY_UNRESOLVED')

    def test_strong_manipulation_evidence_dominates_missing_c2pa(self):
        self.p['c2pa'] = {'c2pa_present': False, 'c2pa_valid': False,
                          'c2pa_status': 'NO_C2PA'}
        self.f.update(media_authenticity='MANIPULATED', confidence=90)
        result = self.decide()
        self.assertEqual(result['category'], 'Likely Manipulated')
        self.assertTrue(result['decision_state']['strong_manipulation_evidence'])
        self.assertIsNone(result['decision_state']['baseline_trigger'])

    def test_trustworthy_contradiction_is_context_not_media_manipulation(self):
        self.p['c2pa'] = {'c2pa_present': False, 'c2pa_valid': False,
                          'c2pa_status': 'NO_C2PA'}
        self.claim(verdict='DEBUNKED', sources=2, trusted=True)
        result = self.decide()
        self.assertEqual(result['category'], 'Misleading Context')
        self.assertEqual(result['claim_context_assessment']['status'], 'CONTRADICTED')
        self.assertFalse(result['decision_state']['strong_manipulation_evidence'])
        self.assertEqual(result['media_authenticity_assessment']['status'], 'INCONCLUSIVE')

    def test_verified_provenance_with_clean_and_uncertain_forensics(self):
        self.sign()
        self.assertEqual(self.decide()['category'], 'Likely Authentic')
        self.assertEqual(self.decide()['confidence'], 'High')
        self.f['media_authenticity'] = 'AUTHENTIC'
        self.assertEqual(self.decide()['confidence'], 'High')

    def test_strong_forensics_and_signed_conflict(self):
        for confidence in (90, .9, 'High'):
            self.f.update(media_authenticity='MANIPULATED', confidence=confidence)
            self.assertEqual(self.decide()['category'], 'Likely Manipulated')
        self.sign()
        self.assertEqual(self.decide()['confidence'], 'Medium')

    def test_weak_manipulation_classification_is_not_decisive(self):
        self.f.update(media_authenticity='MANIPULATED', confidence=60)
        self.assertEqual(self.decide()['category'], 'Needs Review')

    def test_webcam_weak_signals_cannot_outweigh_authentic(self):
        self.p.update(metadata={'exif_status': 'PRESENT'},
                      metadata_summary={'camera_make': 'Webcam'})
        self.f['explainability'] = [
            {'result': name, 'feature_checked': 'heuristic', 'confidence': 90,
             'contribution_to_score': -20, 'execution_state': 'RAN_AND_FOUND_SOMETHING'}
            for name in ('LOW_NOISE_STD', 'FACE_ASYMMETRY_ANOMALY',
                         'EAR_TEXTURE_SMOOTHING', 'EYE_COLOR_MISMATCH',
                         'LIGHTING_DIRECTION_SEVERE')
        ]
        result = self.decide()
        self.assertLessEqual(result['likelihood']['manipulated'], result['likelihood']['authentic'])

    def test_invalid_c2pa_is_strong_suspicious_evidence(self):
        self.p['c2pa'] = {
            'c2pa_present': True, 'c2pa_valid': False,
            'verification_status': 'INVALID', 'c2pa_status': 'C2PA_NOT_VERIFIED',
            'execution_state': 'RAN_AND_FOUND_SOMETHING',
            'signature_valid': False, 'asset_binding_valid': False,
        }
        result = self.decide()
        self.assertEqual(result['category'], 'Likely Manipulated')
        self.assertTrue(result['decision_state']['c2pa_tampered_evidence'])
        self.assertGreater(result['likelihood']['manipulated'], result['likelihood']['authentic'])

    def test_uncertain_forensic_findings_are_preserved_without_forcing_verdict(self):
        self.f.update(
            evidence_strength='MODERATE',
            forensics_score=50,
            all_flags=['FACE_ASYMMETRY_ANOMALY', 'EYE_COLOR_MISMATCH', 'LIGHTING_DIRECTION_SEVERE'],
            explainability=[
                {'result': 'No significant ELA anomaly', 'contribution_to_score': 0,
                 'confidence': 65, 'execution_state': 'RAN_AND_FOUND_NOTHING'},
                {'result': 'No GAN/frequency anomaly', 'contribution_to_score': 0,
                 'confidence': 55, 'execution_state': 'RAN_AND_FOUND_NOTHING'},
                {'result': 'No face-swap artifacts', 'contribution_to_score': 0,
                 'confidence': 70, 'execution_state': 'RAN_AND_FOUND_NOTHING'},
                {'result': 'FACE_ASYMMETRY_ANOMALY', 'contribution_to_score': -5},
                {'result': 'EYE_COLOR_MISMATCH', 'contribution_to_score': -5},
                {'result': 'LIGHTING_DIRECTION_SEVERE', 'contribution_to_score': -20},
            ],
        )
        result = self.decide()
        self.assertEqual(result['category'], 'Needs Review')
        self.assertFalse(result['decision_state']['strong_manipulation_evidence'])
        self.assertEqual(result['agent_summary']['forensics']['evidence_strength'], 'MODERATE')
        self.assertTrue(any('LIGHTING_DIRECTION_SEVERE' in item for item in result['supporting_evidence']))
        self.assertTrue(any('No significant ELA anomaly' in item for item in result['negative_evidence']))
        self.assertIn('insufficient to establish manipulation', result['decision_reasons'][0])

    def test_evidence_changes_distribution_and_skipped_checks_are_not_clean(self):
        self.p.update(
            metadata={'exif_status': 'PRESENT', 'flags': []},
            metadata_summary={'camera_make': 'Example Camera', 'camera_model': 'X1'},
        )
        self.f['explainability'] = [{
            'feature_checked': 'ELA', 'result': 'No compression anomaly',
            'confidence': 65, 'contribution_to_score': 0,
            'execution_state': 'RAN_AND_FOUND_NOTHING',
        }, {
            'feature_checked': 'Face', 'result': 'FEWER_THAN_TWO_RELIABLE_EYES',
            'confidence': 0, 'contribution_to_score': 0,
            'execution_state': 'DID_NOT_COMPLETE',
        }]
        real_photo = self.decide()
        self.f['explainability'].append({
            'feature_checked': 'ELA', 'result': 'LARGE_SUSPICIOUS_REGION',
            'confidence': 90, 'contribution_to_score': -20,
            'execution_state': 'RAN_AND_FOUND_SOMETHING',
        })
        edited = self.decide()
        self.f['explainability'].append({
            'feature_checked': 'Face', 'result': 'FACE_GENERATION_ARTIFACT',
            'confidence': 90, 'contribution_to_score': -20,
            'execution_state': 'RAN_AND_FOUND_SOMETHING',
        })
        ai_face = self.decide()
        self.assertGreater(real_photo['likelihood']['authentic'], real_photo['likelihood']['manipulated'])
        self.assertGreater(edited['likelihood']['manipulated'], real_photo['likelihood']['manipulated'])
        self.assertGreater(ai_face['likelihood']['manipulated'], edited['likelihood']['manipulated'])
        self.assertNotEqual(real_photo['likelihood'], edited['likelihood'])
        self.assertFalse(any('FEWER_THAN_TWO' in item for item in real_photo['negative_evidence']))

    def test_context_is_distinct_from_manipulation(self):
        self.sign()
        self.claim()
        result = self.decide()
        self.assertEqual(result['category'], 'Misleading Context')
        self.assertEqual(result['confidence'], 'High')
        self.assertFalse(result['decision_state']['strong_manipulation_evidence'])

    def test_untrusted_and_count_only_claims_are_not_decisive(self):
        self.c.update(claims_checked=1, debunked_count=99)
        self.assertEqual(self.decide()['category'], 'Needs Review')
        self.claim(trusted=False)
        self.assertEqual(self.decide()['category'], 'Needs Review')

    def test_duplicate_claims_and_sources_do_not_increase_strength(self):
        self.claim(sources=1)
        baseline = self.decide()
        self.c['claim_results'] *= 4
        self.c['claim_results'][0]['evidence_comparisons'] *= 5
        self.assertEqual(self.decide()['likelihood'], baseline['likelihood'])
        self.assertEqual(self.decide()['confidence'], 'Medium')

    def test_verified_claims_cannot_establish_file_authenticity(self):
        self.claim('VERIFIED')
        self.assertTrue(self.decide()['decision_state']['verified_context'])
        self.assertEqual(self.decide()['category'], 'Needs Review')

    def test_official_publisher_headline_corroboration_does_not_authenticate_media(self):
        self.c.update(
            publisher={'detected': True, 'domain': 'publisher.example'},
            source_matches=[{
                'source_type': 'official_publisher_search', 'source': 'publisher.example',
                'final_verdict': 'CORROBORATED', 'entity_mismatches': [],
            }],
        )
        result = self.decide()
        self.assertEqual(result['category'], 'Needs Review')
        self.assertTrue(result['decision_state']['official_source_corroboration'])
        self.assertEqual(result['claim_context_assessment']['status'], 'CORROBORATED')
        self.assertEqual(result['media_authenticity_assessment']['status'], 'INCONCLUSIVE')
        self.f.update(media_authenticity='MANIPULATED', confidence=90)
        self.assertEqual(self.decide()['category'], 'Likely Manipulated')

    def test_separate_media_and_context_assessments_are_always_present(self):
        result = self.decide()
        self.assertEqual(result['media_authenticity_assessment']['status'], 'INCONCLUSIVE')
        self.assertEqual(result['claim_context_assessment']['status'], 'INSUFFICIENT_EVIDENCE')
        self.assertIn('not calibrated probabilities', result['likelihood_note'])

    def test_conflicting_context_requires_review(self):
        self.sign()
        self.claim('CONFLICTING_EVIDENCE')
        self.assertEqual(self.decide()['category'], 'Needs Review')

    def test_failed_stages_ignore_stale_positive_fields(self):
        self.sign()
        self.p['status'] = 'error'
        self.assertEqual(self.decide()['category'], 'Needs Review')
        self.f.update(status='error', media_authenticity='MANIPULATED', confidence=99)
        self.assertEqual(self.decide()['category'], 'Needs Review')

    def test_hash_mismatch_requires_review(self):
        self.sign()
        self.p['integrity_check']['hash_mismatch'] = True
        self.assertEqual(self.decide()['category'], 'Needs Review')
        self.assertFalse(self.decide()['decision_state']['strong_provenance_evidence'])

    def test_missing_integrity_and_string_booleans_are_not_verified(self):
        self.sign()
        self.p['integrity_check'] = None
        self.assertFalse(self.decide()['decision_state']['strong_provenance_evidence'])
        self.p['integrity_check'] = {'hash_mismatch': False}
        self.p['c2pa']['c2pa_valid'] = 'true'
        self.assertFalse(self.decide()['decision_state']['strong_provenance_evidence'])

    def test_crypto_valid_untrusted_signer_and_asset_mismatch_are_distinct(self):
        self.sign()
        self.p['c2pa'].update(
            c2pa_valid=True, applicable_verification=True,
            credential_trusted=False, verification_status='VALID_UNTRUSTED',
            c2pa_status='C2PA_VALID_UNTRUSTED')
        result = self.decide()
        self.assertEqual(result['category'], 'Likely Authentic')
        self.assertEqual(result['media_authenticity_assessment']['status'], 'SUPPORTED_AUTHENTIC')
        self.sign()
        self.p['c2pa'].update(c2pa_valid=False, applicable_verification=False,
                              asset_binding_valid=False, verification_status='INVALID')
        self.assertFalse(self.decide()['decision_state']['strong_provenance_evidence'])

    def test_none_and_malformed_fields_are_safe(self):
        for value in (None, {}, [], 'NOT_CHECKED'):
            result = run_decision_engine(value, value, value, value)
            self.assertEqual(result['category'], 'Needs Review')
            json.dumps(result, allow_nan=False)
        self.f.update(confidence='nonsense', raw_findings={'ela': None})
        self.c.update(claim_results=[None, {}], modules=None)
        self.decide()

    def test_strict_serialization_and_no_mutation(self):
        self.f['confidence'] = float('nan')
        self.p['metadata'] = {'bytes': b'abc', 'tuple': (True, None)}
        original = copy.deepcopy(self.p)
        self.decide()
        self.assertEqual(self.p, original)

    def test_zero_claims_ignore_stale_results(self):
        self.claim()
        self.c['claims_checked'] = 0
        self.assertEqual(self.decide()['category'], 'Needs Review')

    def test_invalid_confidence_is_neutral(self):
        self.f['media_authenticity'] = 'MANIPULATED'
        for value in (None, True, 'invalid', float('inf'), 101, -1):
            self.f['confidence'] = value
            self.assertEqual(self.decide()['category'], 'Needs Review')

    def test_media_agnostic(self):
        self.sign()
        for media_type in ('image', 'video', 'audio'):
            self.i['media_type'] = media_type
            self.assertEqual(self.decide()['category'], 'Likely Authentic')


if __name__ == '__main__':
    unittest.main()
