"""Run the existing agents on seven local image fixtures without overwriting inputs.

Run from repository root: .venv/Scripts/python.exe tests/run_image_regressions.py
External verification uses the existing agent configuration and may be unavailable.
"""
import contextlib
import io
import json
import logging
import os
from pathlib import Path
import shutil
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from image_pipeline import run_local_image_pipeline
from agent import crossReff as crossref_module

# Media regressions are deterministic and offline. Search routing itself is
# covered separately with a mocked provider in test_multilingual_and_ingestion.
crossref_module.search_trusted_sources = lambda query, max_results=5, **kwargs: {
    'found': False, 'results': [], 'requests_made': 0, 'status': 'NOT_CHECKED'
}

CASES = [
    ('normal_jpeg', 'c2pa_adobe_positive.jpg'),
    ('png', 'canonical_image_9d33d48863ac4f94.png'),
    ('social_screenshot', 'carson-screenshot8_0.png'),
    ('newspaper_document', 'imageprocessor.jpeg'),
    ('metadata_positive', 'c2pa_adobe_positive.jpg'),
    ('c2pa_positive', 'adobe-20220124-C.jpg'),
    ('no_meaningful_claims', 'canonical_image_75a8da33f6eaf1e1.png'),
]
if (ROOT / 'chek.jpeg').exists():
    CASES.append(('bangla_ntv_news_graphic', 'chek.jpeg'))


def main():
    reports = []
    logging.disable(logging.CRITICAL)
    with tempfile.TemporaryDirectory(prefix='truelens-regression-') as temporary:
        previous = Path.cwd()
        try:
            os.chdir(temporary)
            for label, filename in CASES:
                source = Path(temporary) / filename
                shutil.copy2(ROOT / filename, source)
                with contextlib.redirect_stdout(io.StringIO()):
                    result = run_local_image_pipeline(str(source))
                assert result['status'] == 'completed', (label, result)
                decision = result['final_decision']
                json.dumps(decision, allow_nan=False)
                assert sum(decision['likelihood'].values()) == 100
                assert len(decision['agent_summary']) == 4
                agents = result['agents']
                assert all(agents[name]['status'] in {'success', 'completed', 'skipped'} for name in agents), (label, agents)
                if label == 'c2pa_positive':
                    c2pa = agents['provenance']['c2pa']
                    assert c2pa['signature_valid'] is True
                    assert c2pa['asset_binding_valid'] is True
                    assert c2pa['credential_trusted'] is False
                    assert c2pa['c2pa_valid'] is False
                    assert c2pa['c2pa_status'] == 'C2PA_VALID_UNTRUSTED'
                    assert decision['decision_state']['strong_provenance_evidence'] is False
                    assert decision['category'] == 'Needs Review'
                if label == 'metadata_positive':
                    assert agents['provenance']['metadata']['exif_status'] == 'PRESENT'
                if label == 'no_meaningful_claims':
                    assert agents['cross_reference']['claims_checked'] == 0
                if label == 'bangla_ntv_news_graphic':
                    crossref = agents['cross_reference']
                    content = agents['forensics']['raw_findings']['content_classification']
                    assert content['document_like'] is True
                    assert agents['forensics']['raw_findings']['face_artifacts']['module_status'] == 'not_applicable'
                    if 'ben' in crossref.get('ocr', {}).get('missing_languages', []):
                        assert crossref['ocr']['status'] == 'partial'
                        assert any('OCR_LANGUAGE_UNAVAILABLE' in flag for flag in crossref['flags'])
                    else:
                        assert crossref['analysis_summary']['reliable_claims_after_filtering'] > 0
                        assert crossref['external_searches_performed'] > 0
                report = {
                    'case': label, 'file': filename, 'category': decision['category'],
                    'confidence': decision['confidence'], 'likelihood': decision['likelihood'],
                    'c2pa_status': agents['provenance']['c2pa']['c2pa_status'],
                    'forensic_verdict': agents['forensics']['media_authenticity'],
                    'claims_checked': agents['cross_reference']['claims_checked'],
                    'modules': agents['cross_reference'].get('modules'),
                    'decision': decision,
                }
                reports.append(report)
                print(json.dumps({k: v for k, v in report.items() if k != 'decision'}), flush=True)
        finally:
            os.chdir(previous)
    target = ROOT / 'result' / 'decision_engine_regressions.json'
    target.parent.mkdir(exist_ok=True)
    target.write_text(json.dumps(reports, indent=2), encoding='utf-8')
    print(f'PASS: {len(reports)} image cases; full decisions saved to {target}', flush=True)


if __name__ == '__main__':
    main()
