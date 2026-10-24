"""Conservative, media-independent evidence fusion; no calibrated probabilities.

Evidence groups have precedence rather than additive scores: broken custody
requires review, strong technical evidence dominates, reliable contradictions
identify context, and verified provenance supports file authenticity only.
"""
import math
import logging
import re


log = logging.getLogger(__name__)


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
        elif (contribution == 0
              and item.get('execution_state') == 'RAN_AND_FOUND_NOTHING'
              and result.lower().startswith(('no ', 'frame-to-frame consistency', 'lip movement'))):
            clean.append(result)
    return list(dict.fromkeys(suspicious)), list(dict.fromkeys(clean))


def _evidence_distribution(provenance, forensics, contradiction=False, verified=False,
                           strong_provenance=False, strong_manipulation=False,
                           c2pa_tampered=False):
    """Fuse available evidence strengths; values are policy weights, not probabilities."""
    p, f = _dict(provenance), _dict(forensics)
    authentic = manipulated = context = 0.0
    trace = []

    metadata = _dict(p.get('metadata'))
    summary = _dict(p.get('metadata_summary'))
    if _status(metadata.get('exif_status')) == 'PRESENT':
        authentic += 10
        trace.append({'source': 'provenance.exif', 'authentic': 10})
    if summary.get('camera_make') or summary.get('camera_model'):
        authentic += 8
        trace.append({'source': 'provenance.camera_identity', 'authentic': 8})
    integrity = _dict(p.get('integrity_check'))
    if integrity.get('hash_mismatch') is False or p.get('hash_matches_ingestor') is True:
        authentic += 4
        trace.append({'source': 'chain_of_custody', 'authentic': 4})

    weak_tokens = {
        'LOW_NOISE_STD', 'FACE_ASYMMETRY_ANOMALY', 'EAR_TEXTURE_SMOOTHING',
        'EYE_COLOR_MISMATCH', 'LIGHTING_DIRECTION_SEVERE',
    }
    weak_manipulated = 0.0
    strong_forensic_signal = False
    for item in _list(f.get('explainability')):
        item = _dict(item)
        contribution = item.get('contribution_to_score')
        confidence = _confidence(item.get('confidence'))
        if isinstance(contribution, (int, float)) and contribution < 0:
            weight = min(20.0, abs(float(contribution)) * max(confidence, 25) / 100)
            result = str(item.get('result') or '').upper()
            if any(token in result for token in weak_tokens):
                weak_manipulated += weight
            else:
                manipulated += weight
                strong_forensic_signal = strong_forensic_signal or weight >= 7
                trace.append({'source': item.get('feature_checked'), 'manipulated': round(weight, 2)})
        elif item.get('execution_state') == 'RAN_AND_FOUND_NOTHING' and confidence > 0:
            weight = min(6.0, confidence / 20)
            authentic += weight
            trace.append({'source': item.get('feature_checked'), 'authentic': round(weight, 2)})
    if weak_manipulated:
        capped = min(8.0, weak_manipulated)
        manipulated += capped
        trace.append({'source': 'bounded_weak_forensic_signals', 'manipulated': round(capped, 2),
                      'uncapped_weight': round(weak_manipulated, 2)})

    provenance_flags = ' '.join(str(v) for v in _list(metadata.get('flags'))).upper()
    if 'AI_SOFTWARE_DETECTED' in provenance_flags:
        manipulated += 18
        trace.append({'source': 'provenance.ai_software', 'manipulated': 18})
    elif 'EDITING_SOFTWARE_DETECTED' in provenance_flags:
        manipulated += 7
        trace.append({'source': 'provenance.editing_software', 'manipulated': 7})
    if contradiction:
        context += 80
        trace.append({'source': 'trusted_context_contradiction', 'misleading_context': 80})
    elif verified:
        context += 3
        trace.append({'source': 'trusted_context_corroboration', 'misleading_context': 3})
    if strong_provenance:
        authentic += 100
        trace.append({'source': 'verified_c2pa_and_integrity', 'authentic': 100})
    if strong_manipulation:
        manipulated += 50
        trace.append({'source': 'high_confidence_forensic_verdict', 'manipulated': 50})
    if c2pa_tampered:
        manipulated += 70
        strong_forensic_signal = True
        trace.append({'source': 'invalid_or_tampered_c2pa', 'manipulated': 70})
    if not (strong_forensic_signal or strong_manipulation or c2pa_tampered):
        manipulated = min(manipulated, authentic)

    usable = authentic + manipulated + context
    if usable == 0:
        return (34, 33, 33), trace, True
    raw = [12 + authentic, 12 + manipulated, 10 + context]
    total = sum(raw)
    values = [int(round(value * 100 / total)) for value in raw]
    values[0] += 100 - sum(values)
    return tuple(values), trace, False


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
              and c2pa.get('applicable_verification') is True
              and c2pa.get('signature_valid') is True
              and c2pa.get('asset_binding_valid') is True
              and _status(c2pa.get('c2pa_status')) in {'C2PA_VERIFIED', 'C2PA_VALID_UNTRUSTED'}
              and _status(c2pa.get('verification_status')) in {'VALID', 'VALID_UNTRUSTED'})
    strong_provenance = signed and integrity_ok
    c2pa_tampered = bool(
        available['provenance'] and c2pa.get('c2pa_present') is True
        and _status(c2pa.get('verification_status')) == 'INVALID'
        and _status(c2pa.get('execution_state')) != 'DID_NOT_COMPLETE'
    )
    c2pa_incomplete = bool(
        c2pa.get('c2pa_present') is True
        and (_status(c2pa.get('execution_state')) == 'DID_NOT_COMPLETE'
             or _status(c2pa.get('verification_status')) in {'FAILED', 'NOT_AVAILABLE'})
    )
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
    publisher = _dict(c.get('publisher'))
    official_source_matches = [
        _dict(match) for match in _list(c.get('source_matches'))
        if _dict(match).get('source_type') == 'official_publisher_search'
        and _dict(match).get('final_verdict') == 'CORROBORATED'
        and _dict(match).get('source') == publisher.get('domain')
        and not _dict(match).get('entity_mismatches')
    ]
    source_image_match = any(_dict(match).get('verdict') == 'NEAR_DUPLICATE_FOUND'
                             for match in _list(c.get('image_matches')))
    strong_source_corroboration = bool(publisher.get('detected') and official_source_matches)
    if strong_provenance:
        supporting.append('C2PA provenance was cryptographically validated and file integrity matches Ingestor.')
        if c2pa.get('credential_trusted') is not True:
            warnings.append('C2PA signature and asset binding are valid, but signer identity is not in the configured trust store.')
        if c2pa.get('generative_ai_declared') is True:
            warnings.append('The valid C2PA manifest declares generative/algorithmic media; provenance is valid but this is not evidence of a camera-original photograph.')
    elif c2pa_tampered:
        negative.append('An embedded C2PA manifest failed signature or asset-binding validation.')
    elif c2pa_incomplete:
        warnings.append('C2PA verification DID_NOT_COMPLETE: ' + str(c2pa.get('note') or 'reason unavailable'))
    elif not c2pa.get('c2pa_present'):
        warnings.append('No C2PA provenance metadata was found; absence is neutral.')
    if clean:
        supporting.append('Forensics reported no strong manipulation evidence; this is not independent proof of authenticity.')
    supporting.extend(f'Forensic signal: {item}' for item in forensic_suspicious)
    negative.extend(f'Forensic negative finding: {item}' for item in forensic_clean)
    if verified:
        supporting.append('Meaningful claims were corroborated by multiple trusted sources.')
    if strong_source_corroboration:
        supporting.append('An official publisher article independently corroborated the reconstructed headline.')
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

    computed_distribution, evidence_trace, no_usable_evidence = _evidence_distribution(
        p if available['provenance'] else {}, f if available['forensics'] else {},
        contradiction=contradiction, verified=bool(verified),
        strong_provenance=strong_provenance,
        strong_manipulation=strong_manipulation,
        c2pa_tampered=c2pa_tampered,
    )
    category, confidence, distribution = 'Needs Review', 'Low', computed_distribution
    baseline_trigger = None
    reasons = []
    if not available['ingestor'] or i.get('ready_for_next_agent') is False or mismatch:
        baseline_trigger = 'INPUT_OR_CHAIN_OF_CUSTODY_UNAVAILABLE'
        reasons.append('Input processing or chain of custody is incomplete; the combined evidence requires review.')
    elif strong_manipulation or c2pa_tampered:
        category = 'Likely Manipulated'
        confidence = 'Medium' if strong_provenance or contradiction else 'High'
        reasons.append(negative[0] if negative else 'Strong manipulation evidence was detected.')
        if strong_provenance:
            warnings.append('Verified provenance and strong forensic manipulation evidence conflict; inspect the signed edit history.')
    elif contradiction:
        category = 'Misleading Context'
        confidence = 'High' if strong_context and (strong_provenance or clean) and not context_conflict else 'Medium'
        reasons.append('Claim contradiction concerns context, not proof of pixel or audio manipulation.')
    elif context_conflict:
        baseline_trigger = 'CONFLICTING_CONTEXTUAL_EVIDENCE'
        reasons.append('Conflicting contextual evidence prevents a decisive combined category.')
    elif strong_provenance:
        category = 'Likely Authentic'
        confidence = 'High'
        reasons.append(supporting[0])
    elif strong_source_corroboration:
        # An official article can corroborate the displayed context, but it
        # cannot authenticate the uploaded bytes. Keep the combined verdict
        # reviewable unless provenance/forensics independently support media.
        baseline_trigger = 'CONTEXT_CORROBORATED_MEDIA_AUTHENTICITY_UNRESOLVED'
        reasons.append('The reconstructed headline is corroborated by the detected publisher, but media authenticity remains unresolved.')
    else:
        baseline_trigger = ('INCONCLUSIVE_SUSPICIOUS_FORENSICS' if forensic_suspicious
                            else 'NO_DECISIVE_PROVENANCE_FORENSIC_OR_CONTEXTUAL_EVIDENCE')
        if forensic_suspicious:
            reasons.append('Suspicious forensic signals were detected, but the available evidence is insufficient to establish manipulation.')
        else:
            reasons.append('Available evidence does not establish media authenticity or strong manipulation; unknown remains neutral.')
    if no_usable_evidence:
        warnings.append('No usable scored evidence was available; the neutral fallback distribution was used explicitly.')
    warnings.append('Likelihood values are an evidence distribution, not calibrated probabilities.')
    log.info(
        'decision category=%s likelihood=%s baseline_trigger=%s evidence: '
        'provenance=%s manipulation=%s contradiction=%s context_conflict=%s '
        'forensic_verdict=%s forensic_confidence=%s claims=%s',
        category, distribution, baseline_trigger, strong_provenance,
        strong_manipulation, contradiction, context_conflict, verdict,
        f.get('confidence'), c.get('claims_checked'),
    )
    colors = {'Likely Authentic': 'green', 'Likely Manipulated': 'red', 'Misleading Context': 'orange', 'Needs Review': 'yellow'}
    media_assessment = {
        'status': ('LIKELY_MANIPULATED' if strong_manipulation or c2pa_tampered else
                   'SUPPORTED_AUTHENTIC' if strong_provenance else
                   'NO_ANOMALY_DETECTED' if clean else 'INCONCLUSIVE'),
        'confidence': confidence if strong_manipulation or c2pa_tampered or strong_provenance else 'Low',
        'reason': ('An embedded C2PA manifest failed cryptographic validation.' if c2pa_tampered else
                   'High-confidence forensic manipulation evidence was found.' if strong_manipulation else
                   'Cryptographically verified provenance and file integrity support authenticity.' if strong_provenance else
                   'Forensics found no strong anomaly, which is not proof of authenticity.' if clean else
                   'No decisive provenance or forensic evidence establishes media authenticity.'),
    }
    context_assessment = {
        'status': ('CONTRADICTED' if contradiction else 'CONFLICTING' if context_conflict else
                   'CORROBORATED' if verified or strong_source_corroboration else 'INSUFFICIENT_EVIDENCE'),
        'confidence': ('High' if strong_context else 'Medium' if contradiction or verified or strong_source_corroboration else 'Low'),
        'reason': ('Trusted evidence contradicts an extracted claim.' if contradiction else
                   'Available sources conflict.' if context_conflict else
                   'Extracted context was corroborated, independently of pixel authenticity.' if verified or strong_source_corroboration else
                   'No claim was verified or contradicted with sufficient trusted evidence.'),
    }
    fields = {
        'ingestor': ('status', 'media_type', 'input_type', 'image_info', 'file_hash', 'source_file_hash', 'normalized_file_hash', 'normalized_file_path', 'ready_for_next_agent'),
        'provenance': ('status', 'c2pa', 'integrity_check', 'metadata_summary', 'provenance_confidence'),
        'forensics': ('status', 'media_authenticity', 'confidence', 'evidence_strength', 'forensics_score', 'evidence', 'all_flags'),
        'cross_reference': ('status', 'crossref_score', 'crossref_note', 'claims_checked', 'verified_count', 'debunked_count', 'modules', 'visual_similarity', 'ocr', 'claim_results', 'sources_used'),
    }
    return {
        'category': category, 'confidence': confidence,
        'likelihood': dict(zip(('authentic', 'manipulated', 'misleading_context'), distribution)),
        'display': {'label': category, 'color': colors[category]},
        'decision_reasons': reasons, 'reasons': reasons.copy(),
        'supporting_evidence': supporting, 'negative_evidence': negative,
        'warnings': list(dict.fromkeys(warnings)),
        'media_authenticity_assessment': media_assessment,
        'claim_context_assessment': context_assessment,
        'agent_summary': {name: {'available': available[name], **{key: _safe(output[key]) for key in fields[name] if key in output}} for name, output in agents.items()},
        'agent_outputs': _safe(agents),
        'decision_state': {**{name + '_available': value for name, value in available.items()},
                           'strong_provenance_evidence': strong_provenance,
                           'c2pa_tampered_evidence': c2pa_tampered,
                           'c2pa_verification_incomplete': c2pa_incomplete,
                           'strong_manipulation_evidence': strong_manipulation,
                           'contextual_contradiction': contradiction, 'contextual_conflict': context_conflict,
                           'verified_context': bool(verified), 'integrity_valid': integrity_ok,
                           'official_source_corroboration': strong_source_corroboration,
                           'source_image_match': source_image_match,
                           'baseline_trigger': baseline_trigger,
                           'used_inconclusive_baseline': baseline_trigger is not None,
                           'used_no_evidence_default': no_usable_evidence,
                           'evidence_contributions': evidence_trace},
        'is_calibrated_probability': False,
        'likelihood_note': 'Policy-based evidence distribution; values are not calibrated probabilities or percentages of image content.',
        'ready_for_display': True,
    }
