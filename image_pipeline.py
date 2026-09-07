"""Programmatic local-image orchestration for the existing TrueLens agents."""

import json
import os
import sys
import time
from datetime import datetime, timezone

from decision_engine import run_decision_engine
from agent.crossReff import run_crossref_agent
from agent.forensic import run_forensics_agent
from agent.ingestor import run_ingestor_agent
from agent.provinence import run_provenance_agent


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


def run_local_image_pipeline(local_image: str) -> dict:
    """Run Ingestor -> Provenance -> Forensics -> Cross-Reference without pasted JSON."""
    pipeline_started = time.perf_counter()
    local_image = os.path.abspath(local_image)

    ingestor, ingest_ms = _timed_call(run_ingestor_agent, local_image)
    if not ingestor.get("ready_for_next_agent") or ingestor.get("media_type") != "image":
        return {
            "status": "error",
            "stage": "ingestor",
            "error_message": ingestor.get("error_message", "Input is not a supported image"),
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
    return {
        "status": "completed",
        "processed_at": datetime.now(timezone.utc).isoformat(),
        "media_type": "image",
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


if __name__ == "__main__":
    image_path = sys.argv[1] if len(sys.argv) > 1 else "imageprocessor.jpeg"
    decision_output = run_local_image_pipeline(image_path)["final_decision"]
    print(json.dumps(decision_output, indent=2))
