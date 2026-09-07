"""Programmatic local-image orchestration for the existing TrueLens agents."""

import json
import os
import sys
import time
from datetime import datetime, timezone

from crossReff import run_crossref_agent
from forensic import run_forensics_agent
from ingestor import run_ingestor_agent
from provinence import run_provenance_agent


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


def make_final_decision(provenance: dict, forensics: dict, cross_reference: dict) -> dict:
    """Fuse only explicit evidence; heuristic scores are not treated as probabilities."""
    reasons = []
    c2pa_valid = provenance.get("c2pa", {}).get("c2pa_valid") is True
    hash_verdict = provenance.get("hash_lookup", {}).get("verdict")
    hash_mismatch = provenance.get("integrity_check", {}).get("hash_mismatch") is True
    forensic_verdict = str(forensics.get("media_authenticity", "UNCERTAIN")).upper()
    forensic_confidence = float(forensics.get("confidence") or 0)
    debunked_claims = int(cross_reference.get("debunked_count") or 0)

    if hash_mismatch:
        reasons.append("The source file no longer matches the hash recorded by Ingestor.")
    if hash_verdict == "DEBUNKED":
        reasons.append("The exact source hash matches a known debunked record.")
    if debunked_claims:
        reasons.append(
            f"{debunked_claims} extracted factual claim(s) were contradicted by evidence."
        )
    if forensic_verdict == "MANIPULATED" and forensic_confidence >= 85:
        reasons.append("Forensics found multiple strong, corroborated manipulation indicators.")

    strong_manipulation = (
        hash_mismatch
        or hash_verdict == "DEBUNKED"
        or debunked_claims > 0
        or (forensic_verdict == "MANIPULATED" and forensic_confidence >= 85)
    )
    if strong_manipulation:
        category = "Likely Manipulated"
        confidence = "High" if hash_verdict == "DEBUNKED" or hash_mismatch else "Medium"
    elif c2pa_valid and forensic_verdict != "MANIPULATED":
        category = "Likely Authentic"
        confidence = "High"
        reasons.append("A C2PA manifest was cryptographically validated.")
    else:
        category = "Needs Review"
        confidence = "Low"
        reasons.append(
            "Available provenance, forensic, and contextual evidence is not decisive; unknown is not fake."
        )

    return {
        "category": category,
        "confidence": confidence,
        "reasons": reasons,
        "is_calibrated_probability": False,
    }


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
            "ready_for_display": True,
        }

    provenance, provenance_ms = _timed_call(run_provenance_agent, ingestor)
    if not provenance.get("ready_for_next_agent"):
        return {
            "status": "error",
            "stage": "provenance",
            "error_message": provenance.get("error_message", "Provenance failed"),
            "agents": {"ingestor": ingestor, "provenance": provenance},
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
        cross_reference, crossref_ms = _timed_call(run_crossref_agent, provenance, forensics)

    final_decision = make_final_decision(provenance, forensics, cross_reference)
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
    print(json.dumps(run_local_image_pipeline(image_path), indent=2, default=str))
