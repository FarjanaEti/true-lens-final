"""Conservative, media-independent evidence fusion; no calibrated probabilities.

Evidence groups have precedence rather than additive scores: broken custody
requires review, strong technical evidence dominates, reliable contradictions
identify context, and verified provenance supports file authenticity only.
"""
import math
import re


def _dict(value):
    return value if isinstance(value, dict) else {}


def _list(value):
    return value if isinstance(value, (list, tuple)) else []


def _status(value):
    return str(value or '').strip().upper()


def _confidence(value):
    if isinstance(value, bool):
        return 0
    if _status(value) in {'HIGH', 'HIGH_CONFIDENCE'}:
        return 90
    try:
        number = float(value)
        return (number * 100 if 0 <= number <= 1 else number) if math.isfinite(number) and 0 <= number <= 100 else 0
    except (TypeError, ValueError):
        return 0


def _has_claims(value):
    try:
        return not isinstance(value, bool) and int(value) > 0
    except (ValueError, TypeError, OverflowError):
        return False


def _available(value):
    return bool(value) and value.get('ready_for_next_agent') is not False and _status(value.get('status')) in {'SUCCESS', 'COMPLETED', 'OK'}


def _safe(value):
    """Copy diagnostic outputs into strict JSON-compatible primitives."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(k): _safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(v) for v in value]
    return str(value)


def _forensic_evidence(forensics):
    """Separate suspicious findings from clean checks without changing verdicts."""
    suspicious, clean = [], []
    for item in _list(forensics.get('explainability')):
        item = _dict(item)
        result = str(item.get('result') or '').strip()
        contribution = item.get('contribution_to_score')
        if not result:
            continue
        if isinstance(contribution, (int, float)) and not isinstance(contribution, bool) and contribution < 0:
            suspicious.append(result)
        elif contribution == 0 and result.lower().startswith(('no ', 'frame-to-frame consistency', 'lip movement')):
            clean.append(result)
    return list(dict.fromkeys(suspicious)), list(dict.fromkeys(clean))


def run_decision_engine(ingestor_output, provenance_output, forensics_output, crossref_output):
    """Return explainable data from four dictionaries, tolerating missing stages.

    Scores are preserved for inspection but never fused. Fixed, coarse evidence
    distributions express policy strength, not measured statistical probability.
    Claim counts and repeated sources do not independently add evidence weight.
    """
    agents = dict(zip(('ingestor', 'provenance', 'forensics', 'cross_reference'),
                      map(_dict, (ingestor_output, provenance_output, forensics_output, crossref_output))))
    i, p, f, c = agents.values()
    available = {name: _available(output) for name, output in agents.items()}
    c2pa = _dict(p.get('c2pa'))
    integrity = _dict(p.get('integrity_check'))
    mismatch = available['provenance'] and (integrity.get('hash_mismatch') is True or p.get('hash_matches_ingestor') is False)
    integrity_ok = available['provenance'] and not mismatch and (
        integrity.get('hash_mismatch') is False or p.get('hash_matches_ingestor') is True)
    signed = (available['provenance'] and c2pa.get('c2pa_present') is True
              and c2pa.get('c2pa_valid') is True
              and _status(c2pa.get('c2pa_status')) == 'C2PA_VERIFIED'
              and _status(c2pa.get('verification_status')) not in {'INVALID', 'NOT_CHECKED', 'NOT_AVAILABLE'})
    strong_provenance = signed and integrity_ok
    verdict = _status(f.get('media_authenticity') or f.get('classification'))
    strong_manipulation = available['forensics'] and verdict in {
        'MANIPULATED', 'LIKELY_MANIPULATED', 'DEEPFAKE', 'AI_GENERATED'
    } and _confidence(f.get('confidence')) >= 85
    clean = available['forensics'] and verdict == 'AUTHENTIC'
    forensic_suspicious, forensic_clean = _forensic_evidence(f) if available['forensics'] else ([], [])

    # Use claim-level evidence, never aggregate debunked_count alone. Group
    # duplicate OCR claims and independent publishers before interpreting them.
    groups = {}
    for claim in _list(c.get('claim_results')) if available['cross_reference'] and _has_claims(c.get('claims_checked')) else []:
        claim = _dict(claim)
        key = re.sub(r'\W+', ' ', str(claim.get('claim') or '').lower()).strip()
        if not key or claim.get('is_fact_checkable') is False:
            continue
        group = groups.setdefault(key, {'support': set(), 'contradict': set(), 'verdicts': set(), 'confidence': 0})
        group['verdicts'].add(_status(claim.get('verdict')))
        group['confidence'] = max(group['confidence'], _confidence(claim.get('confidence')))
        for comparison in _list(claim.get('evidence_comparisons')):
            comparison = _dict(comparison)
            source = str(comparison.get('source') or '').lower().removeprefix('www.').strip()
            if comparison.get('trusted_domain') is not True or not source or not comparison.get('url'):
                continue
            direction = {'CORROBORATED': 'support', 'CONTRADICTED': 'contradict'}.get(_status(comparison.get('final_verdict')))
            if direction:
                group[direction].add(source)
    contradicted = [g for g in groups.values() if g['contradict'] and not g['support']
                   and g['verdicts'] <= {'DEBUNKED', 'CONTRADICTED'} and g['confidence'] >= 65]
    verified = [g for g in groups.values() if len(g['support']) >= 2 and not g['contradict']
                and g['verdicts'] == {'VERIFIED'} and g['confidence'] >= 65]
    context_conflict = any((g['support'] and g['contradict']) or 'CONFLICTING_EVIDENCE' in g['verdicts'] for g in groups.values())
    contradiction = bool(contradicted)
    strong_context = any(len(g['contradict']) >= 2 and g['confidence'] >= 75 for g in contradicted)
    supporting, negative, warnings = [], [], []
    if strong_provenance:
        supporting.append('C2PA provenance was cryptographically validated and file integrity matches Ingestor.')
    if clean:
        supporting.append('Forensics reported no strong manipulation evidence; this is not independent proof of authenticity.')
    supporting.extend(f'Forensic signal: {item}' for item in forensic_suspicious)
    negative.extend(f'Forensic negative finding: {item}' for item in forensic_clean)
    if verified:
        supporting.append('Meaningful claims were corroborated by multiple trusted sources.')
    if strong_manipulation:
        negative.append('Forensics explicitly classified manipulation with high confidence.')
    if contradiction:
        negative.append('Trusted-source evidence contradicts an extracted factual claim.')
    if mismatch:
        negative.append('Source hash differs from Ingestor: evidence may concern different file contents.')
    if context_conflict:
        warnings.append('Contextual evidence conflicts and requires review.')
    if not verified and not contradiction:
        warnings.append('Context remains unresolved; missing claims or search results establish neither truth nor falsity.')
    if signed:
        warnings.append('Valid C2PA supports provenance, not the truth of captions or factual claims, nor absence of declared edits.')
    if not integrity_ok:
        warnings.append('File integrity is unverified or mismatched.')
    for name, output in agents.items():
        if not available[name]:
            warnings.append(f'{name} did not complete; its evidence is unavailable.')
        for field in ('analysis_limitations', 'analysis_notes'):
            warnings.extend(str(note) for note in _list(output.get(field)))
    for name, module in _dict(f.get('raw_findings')).items():
        state = _status(_dict(module).get('module_status'))
        if state != 'OK':
            warnings.append(f'Forensics module {name}: {state or "UNKNOWN"}; no evidence inferred from unavailability.')
    for claim in _list(c.get('claim_results')):
        claim_status = _status(_dict(claim).get('verdict'))
        if claim_status in {'SEARCH_FAILED', 'UNVERIFIED', 'INSUFFICIENT_EVIDENCE'}:
            warnings.append(f'Claim verification: {claim_status}; no authenticity inference was made.')
    for name, state in _dict(c.get('modules')).items():
        if _status(state) != 'OK':
            warnings.append(f'Cross-reference module {name}: {state}; missing coverage remains neutral.')
    hash_record = _dict(p.get('hash_lookup'))
    if hash_record.get('hash_match') is True:
        warnings.append('Hash lookup records are retained for review; the current lookup is a demo database and does not establish manipulation or context by itself.')

    category, confidence, distribution = 'Needs Review', 'Low', (35, 35, 30)
    reasons = []
    if not available['ingestor'] or i.get('ready_for_next_agent') is False or mismatch:
        reasons.append('Input processing or chain of custody is incomplete; the combined evidence requires review.')
    elif strong_manipulation:
        category = 'Likely Manipulated'
        confidence = 'Medium' if strong_provenance or contradiction else 'High'
        distribution = (25, 65, 10) if strong_provenance else ((5, 65, 30) if contradiction else (5, 90, 5))
        reasons.append(negative[0])
        if strong_provenance:
            warnings.append('Verified provenance and strong forensic manipulation evidence conflict; inspect the signed edit history.')
    elif contradiction:
        category = 'Misleading Context'
        confidence = 'High' if strong_context and (strong_provenance or clean) and not context_conflict else 'Medium'
        distribution = (10, 5, 85) if confidence == 'High' else (20, 15, 65)
        reasons.append('Claim contradiction concerns context, not proof of pixel or audio manipulation.')
    elif context_conflict:
        reasons.append('Conflicting contextual evidence prevents a decisive combined category.')
    elif strong_provenance:
        category = 'Likely Authentic'
        confidence = 'High' if clean else 'Medium'
        distribution = (90, 5, 5) if clean and verified else ((85, 5, 10) if clean else (75, 10, 15))
        reasons.append(supporting[0])
    else:
        if forensic_suspicious:
            reasons.append('Suspicious forensic signals were detected, but the available evidence is insufficient to establish manipulation.')
        else:
            reasons.append('Available evidence does not establish media authenticity or strong manipulation; unknown remains neutral.')
    warnings.append('Likelihood values are an evidence distribution, not calibrated probabilities.')
    colors = {'Likely Authentic': 'green', 'Likely Manipulated': 'red', 'Misleading Context': 'orange', 'Needs Review': 'yellow'}
    fields = {
        'ingestor': ('status', 'media_type', 'input_type', 'image_info', 'file_hash', 'ready_for_next_agent'),
        'provenance': ('status', 'c2pa', 'integrity_check', 'metadata_summary', 'provenance_confidence'),
        'forensics': ('status', 'media_authenticity', 'confidence', 'evidence_strength', 'forensics_score', 'evidence', 'all_flags'),
        'cross_reference': ('status', 'claims_checked', 'verified_count', 'debunked_count', 'modules', 'visual_similarity'),
    }
    return {
        'category': category, 'confidence': confidence,
        'likelihood': dict(zip(('authentic', 'manipulated', 'misleading_context'), distribution)),
        'display': {'label': category, 'color': colors[category]},
        'decision_reasons': reasons, 'reasons': reasons.copy(),
        'supporting_evidence': supporting, 'negative_evidence': negative,
        'warnings': list(dict.fromkeys(warnings)),
        'agent_summary': {name: {'available': available[name], **{key: _safe(output[key]) for key in fields[name] if key in output}} for name, output in agents.items()},
        'agent_outputs': _safe(agents),
        'decision_state': {**{name + '_available': value for name, value in available.items()},
                           'strong_provenance_evidence': strong_provenance,
                           'strong_manipulation_evidence': strong_manipulation,
                           'contextual_contradiction': contradiction, 'contextual_conflict': context_conflict,
                           'verified_context': bool(verified), 'integrity_valid': integrity_ok},
        'is_calibrated_probability': False, 'ready_for_display': True,
    }
