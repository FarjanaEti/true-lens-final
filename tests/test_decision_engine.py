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
        self.p['c2pa'] = {'c2pa_present': True, 'c2pa_valid': True, 'c2pa_status': 'C2PA_VERIFIED', 'verification_status': 'VALID'}

    def claim(self, verdict='DEBUNKED', sources=2, trusted=True):
        self.c.update(claims_checked=1, claim_results=[{
            'claim': 'This photograph was taken today in Dhaka.',
            'is_fact_checkable': True, 'verdict': verdict, 'confidence': .85,
            'evidence_comparisons': [dict(source=f'source{n}.org', url=f'https://source{n}.org/report', trusted_domain=trusted,
                                          final_verdict='CORROBORATED' if verdict == 'VERIFIED' else 'CONTRADICTED') for n in range(sources)]}])

    def test_unknown_and_weak_flags_are_neutral(self):
        baseline = self.decide()
        self.p.update(c2pa=None, metadata={'exif_status': 'MISSING'}, provenance_score=1)
        self.f.update(all_flags=['NO_FACE', 'HIGH_ELA_MEAN', 'DOCUMENT_LAYOUT'], forensics_score=1)
        self.c.update(crossref_score=99, modules={'ocr': 'error'}, visual_similarity={'verdict': 'NOT_CHECKED'})
        self.assertEqual(self.decide()['likelihood'], baseline['likelihood'])
        self.assertEqual(self.decide()['category'], 'Needs Review')

    def test_verified_provenance_with_clean_and_uncertain_forensics(self):
        self.sign()
        self.assertEqual(self.decide()['category'], 'Likely Authentic')
        self.assertEqual(self.decide()['confidence'], 'Medium')
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
