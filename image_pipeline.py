"""Unified TrueLens orchestration for local files and supported URLs."""

import json
import logging
import os
import sys
import tempfile
import time
from datetime import datetime, timezone

from decision_engine import run_decision_engine
from agent.crossReff import run_crossref_agent
from agent.forensic import run_forensics_agent
from agent.ingestor import run_ingestor_agent
from agent.provinence import run_provenance_agent


log = logging.getLogger(__name__)


def _timed_call(function, *args):
    started = time.perf_counter()
    result = function(*args)
    if not isinstance(result, dict):
        result = {
            "status": "error",
            "agent": getattr(function, "__name__", "unknown"),
            "error_message": "Agent returned a non-dictionary value",
            "ready_for_next_agent": False,
        }
    return result, round((time.perf_counter() - started) * 1000, 2)


def make_final_decision(provenance, forensics, cross_reference, ingestor=None):
    """Compatibility entry point; pass Ingestor for a complete decision."""
    return run_decision_engine(ingestor, provenance, forensics, cross_reference)


def _run_pipeline(
    user_input: str,
    cookies_file: str,
    canonical_output_dir: str,
) -> dict:
    """Run all agents while the request's canonical workspace exists."""
    pipeline_started = time.perf_counter()
    if not str(user_input).lower().startswith(("http://", "https://")):
        user_input = os.path.abspath(user_input)

    if cookies_file is None:
        ingestor, ingest_ms = _timed_call(
            run_ingestor_agent, user_input, None, canonical_output_dir
        )
    else:
        ingestor, ingest_ms = _timed_call(
            run_ingestor_agent, user_input, cookies_file, canonical_output_dir
        )
    log.info('pipeline input=%s ingestor_status=%s media_type=%s source_hash=%s canonical_path=%s',
             user_input, ingestor.get('status'), ingestor.get('media_type'),
             ingestor.get('source_file_hash'), ingestor.get('normalized_file_path'))
    if not ingestor.get("ready_for_next_agent") or ingestor.get("media_type") not in {
        "image", "video", "audio"
    }:
        return {
            "status": "error",
            "stage": "ingestor",
            "error_message": ingestor.get("error_message", "Input is not supported"),
            "agents": {"ingestor": ingestor},
            "final_decision": run_decision_engine(ingestor, None, None, None),
            "ready_for_display": True,
        }

    provenance, provenance_ms = _timed_call(run_provenance_agent, ingestor)
    if not provenance.get("ready_for_next_agent"):
        return {
            "status": "error",
            "stage": "provenance",
            "error_message": provenance.get("error_message", "Provenance failed"),
            "agents": {"ingestor": ingestor, "provenance": provenance},
            "final_decision": run_decision_engine(ingestor, provenance, None, None),
            "ready_for_display": True,
        }

    forensics, forensics_ms = _timed_call(run_forensics_agent, provenance)
    if not forensics.get("ready_for_next_agent"):
        cross_reference = {
            "status": "not_run",
            "crossref_score": None,
            "claims_checked": 0,
            "reason": "Forensics did not complete",
            "ready_for_next_agent": True,
        }
        crossref_ms = 0.0
    else:
        cross_reference, crossref_ms = _timed_call(
            run_crossref_agent, provenance, forensics)

    final_decision = run_decision_engine(
        ingestor, provenance, forensics, cross_reference)
    log.info('pipeline evidence provenance=%s c2pa=%s forensics=%s/%s claims=%s '
             'decision=%s baseline=%s', provenance.get('status'),
             provenance.get('c2pa', {}).get('c2pa_status'),
             forensics.get('media_authenticity'), forensics.get('confidence'),
             cross_reference.get('claims_checked'), final_decision.get('category'),
             final_decision.get('decision_state', {}).get('baseline_trigger'))
    return {
        "status": "completed",
        "processed_at": datetime.now(timezone.utc).isoformat(),
        "input_type": ingestor.get("input_type"),
        "media_type": ingestor.get("media_type"),
        "agents": {
            "ingestor": ingestor,
            "provenance": provenance,
            "forensics": forensics,
            "cross_reference": cross_reference,
        },
        "timing_ms": {
            "ingestor": ingest_ms,
            "provenance": provenance_ms,
            "forensics": forensics_ms,
            "cross_reference": crossref_ms,
            "total": round((time.perf_counter() - pipeline_started) * 1000, 2),
        },
        "final_decision": final_decision,
        "ready_for_display": True,
    }


def run_pipeline(user_input: str, cookies_file: str = None) -> dict:
    """Run the four agents in an isolated, automatically cleaned workspace."""
    request_workspace = tempfile.TemporaryDirectory(prefix="truelens-request-")
    try:
        return _run_pipeline(user_input, cookies_file, request_workspace.name)
    finally:
        request_workspace.cleanup()


def run_local_image_pipeline(local_image: str) -> dict:
    """Backward-compatible local-image entry point."""
    return run_pipeline(local_image)


if __name__ == "__main__":
    input_value = sys.argv[1] if len(sys.argv) > 1 else "imageprocessor.jpeg"
    cookies = sys.argv[2] if len(sys.argv) > 2 else None
    print(json.dumps(run_pipeline(input_value, cookies), indent=2))
