"""Run a reproducible, evidence-preserving evaluation of repository image fixtures.

This is an observational benchmark. It never mocks search, assigns labels from
filenames, or changes the Decision Engine policy.

Run from the repository root:
    .venv/Scripts/python.exe -B tests/run_image_evaluation.py
"""
from __future__ import annotations

import argparse
import contextlib
from datetime import datetime, timezone
import hashlib
import io
import json
import logging
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from image_pipeline import run_local_image_pipeline
from agent import crossReff as crossref_module

MANIFEST_PATH = ROOT / "tests" / "image_evaluation_manifest.json"
DEFAULT_JSON_PATH = ROOT / "result" / "image_evaluation.json"
DEFAULT_MARKDOWN_PATH = ROOT / "result" / "image_evaluation.md"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _get(value: Any, *keys: str, default: Any = None) -> Any:
    current = value
    for key in keys:
        if not isinstance(current, dict):
            return default
        current = current.get(key)
    return default if current is None else current


def summarize_pipeline(result: dict[str, Any]) -> dict[str, Any]:
    agents = result.get("agents") or {}
    ingestor = agents.get("ingestor") or {}
    provenance = agents.get("provenance") or {}
    forensics = agents.get("forensics") or {}
    crossref = agents.get("cross_reference") or {}
    decision = result.get("final_decision") or {}
    claims = crossref.get("claim_results") or []
    return {
        "pipeline_status": result.get("status"),
        "ingestor": {
            "status": ingestor.get("status"),
            "source_sha256": ingestor.get("source_file_hash"),
            "normalized_sha256": ingestor.get("normalized_file_hash"),
            "media_type": ingestor.get("media_type"),
        },
        "ocr": crossref.get("ocr") or {
            "status": crossref.get("modules", {}).get("ocr"),
            "raw_text": crossref.get("ocr_text", ""),
        },
        "claims": claims,
        "provenance": {
            "status": provenance.get("status"),
            "assessment": provenance.get("provenance_assessment"),
            "c2pa": provenance.get("c2pa"),
            "metadata": provenance.get("metadata"),
            "hash_verification": provenance.get("hash_verification"),
            "flags": provenance.get("flags", []),
        },
        "forensics": {
            "status": forensics.get("status"),
            "media_authenticity": forensics.get("media_authenticity"),
            "confidence": forensics.get("confidence"),
            "findings": forensics.get("raw_findings"),
            "flags": forensics.get("flags", []),
        },
        "cross_reference": {
            "status": crossref.get("status"),
            "search_status": crossref.get("search_status"),
            "external_searches_performed": crossref.get("external_searches_performed", 0),
            "queries": [query for claim in claims for query in claim.get("search_queries", [])],
            "search_traces": [trace for claim in claims for trace in claim.get("search_trace", [])],
            "source_results": crossref.get("source_results", []),
            "sources_used": crossref.get("sources_used", []),
            "claims_checked": crossref.get("claims_checked", 0),
            "assessment": [
                {
                    "claim": claim.get("claim"),
                    "source_text": claim.get("source_text"),
                    "verdict": claim.get("verdict"),
                    "reason": claim.get("verification_reason"),
                    "evidence": claim.get("evidence_comparisons", []),
                }
                for claim in claims
            ],
            "flags": crossref.get("flags", []),
        },
        "decision": {
            "category": decision.get("category"),
            "confidence": decision.get("confidence"),
            "likelihood": decision.get("likelihood"),
            "media_authenticity_assessment": _get(decision, "agent_summary", "forensics", "media_authenticity"),
            "claim_context_assessment": [claim.get("verdict") for claim in claims],
            "reasons": decision.get("decision_reasons", []),
            "warnings": decision.get("warnings", []),
            "trigger": decision.get("decision_state", {}).get("baseline_trigger"),
            "state": decision.get("decision_state", {}),
            "evidence": {
                "supporting": decision.get("supporting_evidence", []),
                "negative": decision.get("negative_evidence", []),
            },
        },
    }


def assess_support(fixture: dict[str, Any], summary: dict[str, Any]) -> dict[str, Any]:
    expected = fixture["expected"]
    checks = {
        "ingestor_status": summary["ingestor"]["status"] == expected["ingestor_status"]
    }
    expected_c2pa = expected.get("c2pa_status")
    if expected_c2pa:
        checks["c2pa_status"] = _get(summary, "provenance", "c2pa", "c2pa_status") == expected_c2pa
    return {
        "supported_by_available_ground_truth": all(checks.values()),
        "checks": checks,
        "accuracy_scored": False,
        "accuracy_exclusion_reason": (
            "Control fixture tests provenance parsing, not an A-D content label."
            if fixture["label_status"] == "verified_control_only"
            else "No independently established category label or licensed source record."
        ),
    }


def _clip(value: Any, limit: int = 160) -> str:
    text = " ".join(str(value or "").split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# TrueLens image capability evaluation",
        "",
        f"Generated: `{report['generated_at']}`",
        "",
        "This is an observational run of the real pipeline. External search was not mocked. "
        "A successful software run is not counted as correct detection unless the fixture has independent ground truth.",
        "",
        "## Dataset validity and coverage",
        "",
        f"- Files evaluated: {report['dataset']['fixture_count']}",
        f"- Independently labeled A-D examples: {report['metrics']['labeled_example_count']}",
        f"- Accuracy estimate: {report['metrics']['accuracy']}",
        f"- Missing categories: {', '.join(report['dataset']['missing_categories'])}",
        "- The Adobe asset is a verified C2PA parser/provenance control, not a real-world A-D content label.",
        "- Repository filenames, logos, and test method names were not accepted as ground truth.",
        "",
        "## Fixture results",
        "",
        "| Fixture | SHA-256 | Label status | Pipeline | OCR quality | Searches / status | Media assessment | Context/final |",
        "| --- | --- | --- | --- | ---: | --- | --- | --- |",
    ]
    for item in report["fixtures"]:
        s = item["actual"]
        ocr_quality = _get(s, "ocr", "quality_score", default="n/a")
        search = s["cross_reference"]
        decision = s["decision"]
        lines.append(
            f"| `{item['filename']}` | `{item['sha256']}` | {item['label_status']} | "
            f"{s['pipeline_status']} | {ocr_quality} | {search['external_searches_performed']} / "
            f"{search['search_status']} | {_clip(s['forensics']['media_authenticity'])} | "
            f"{_clip(decision['claim_context_assessment'])}; **{decision['category']}** |"
        )
    for item in report["fixtures"]:
        s = item["actual"]
        search = s["cross_reference"]
        decision = s["decision"]
        lines.extend([
            "",
            f"### `{item['filename']}`",
            "",
            f"- Category: `{item['category']}`; role: `{item['role']}`; label: `{item['label_status']}`.",
            f"- Ground truth/source: {_clip(item['ground_truth'].get('claim') or 'None documented in this repository.')}",
            f"- Source URL: {item['ground_truth'].get('source_url') or 'not documented'}",
            f"- License: {item['ground_truth'].get('license_url') or 'not documented; do not redistribute until cleared'}",
            f"- Expected media behavior: {item['expected']['media_authenticity']}",
            f"- Expected context behavior: {item['expected']['claim_context']}",
            f"- Raw OCR: `{_clip(_get(s, 'ocr', 'raw_text', default=_get(s, 'ocr', 'text', default='')), 300)}`",
            f"- Reconstructed headline: `{_clip(_get(s, 'ocr', 'reconstructed_headline', default=''), 240)}`",
            f"- Claims: {len(s['claims'])}; searches: {search['external_searches_performed']}; provider status: `{search['search_status']}`.",
            f"- Final: `{decision['category']}` / `{decision['confidence']}`; trigger: `{decision['trigger']}`.",
            f"- Evidence-supported behavior: `{item['evaluation']['supported_by_available_ground_truth']}`. "
            f"Accuracy excluded because {item['evaluation']['accuracy_exclusion_reason']}",
            "- Full OCR passes, claims/source text, query records, provider results, provenance, forensic findings, warnings, and the complete pipeline envelope are retained in the JSON report.",
        ])
    lines.extend([
        "",
        "## Missing benchmark fixtures",
        "",
    ])
    for category, requirement in report["dataset"]["fixture_requirements"].items():
        lines.append(f"- **{category}:** {requirement}")
    lines.extend([
        "",
        "## Metrics",
        "",
        report["metrics"]["reason"],
        "",
        f"Operationally, {report['search_observations']['routed_query_attempts']} routed query attempts produced "
        f"{report['search_observations']['provider_requests_made']} actual provider requests and "
        f"{report['search_observations']['returned_result_count']} returned results. Provider: "
        f"{report['search_observations']['primary_provider']}.",
        "",
        "## Next engineering step",
        "",
        report["recommendation"],
        "",
        "## Reproduction",
        "",
        "```powershell",
        ".venv/Scripts/python.exe -B tests/run_image_evaluation.py",
        "```",
        "",
        "The run is intentionally live: provider availability and indexed results are timestamped evidence, not deterministic ground truth. Unit tests continue to mock external services.",
    ])
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", type=Path, default=DEFAULT_JSON_PATH)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MARKDOWN_PATH)
    parser.add_argument(
        "--reuse-existing",
        action="store_true",
        help="Regenerate report summaries from the existing JSON without rerunning external providers.",
    )
    args = parser.parse_args()
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    evaluated = []
    logging.disable(logging.CRITICAL)
    if args.reuse_existing:
        previous_report = json.loads(args.json.read_text(encoding="utf-8"))
        previous_by_name = {item["filename"]: item for item in previous_report["fixtures"]}
        for fixture in manifest["fixtures"]:
            previous_item = previous_by_name[fixture["filename"]]
            pipeline = previous_item["pipeline_result"]
            summary = summarize_pipeline(pipeline)
            item = {
                **fixture,
                "sha256": previous_item["sha256"],
                "actual": summary,
                "pipeline_result": pipeline,
            }
            item["evaluation"] = assess_support(fixture, summary)
            evaluated.append(item)
    else:
      with tempfile.TemporaryDirectory(prefix="truelens-evaluation-") as temporary:
        previous = Path.cwd()
        try:
            os.chdir(temporary)
            for fixture in manifest["fixtures"]:
                source = ROOT / fixture["filename"]
                if not source.is_file():
                    raise FileNotFoundError(f"Missing fixture: {source}")
                actual_hash = sha256_file(source)
                expected_hash = fixture.get("ground_truth", {}).get("expected_sha256")
                if expected_hash and actual_hash != expected_hash:
                    raise ValueError(f"Ground-truth hash mismatch for {source.name}: {actual_hash}")
                copied = Path(temporary) / source.name
                shutil.copy2(source, copied)
                with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                    pipeline = run_local_image_pipeline(str(copied))
                summary = summarize_pipeline(pipeline)
                item = {**fixture, "sha256": actual_hash, "actual": summary, "pipeline_result": pipeline}
                item["evaluation"] = assess_support(fixture, summary)
                evaluated.append(item)
                print(f"{source.name}: {summary['pipeline_status']} -> {summary['decision']['category']}", flush=True)
        finally:
            os.chdir(previous)

    requirements = {
        "A": "A legally reusable news graphic plus publisher/origin record and at least one independent authoritative corroboration of its claim and publication date.",
        "B": "A licensed genuine photograph with documented original event/date/location and a preserved false caption or repost, supported by an authoritative debunk or source comparison.",
        "C": "A licensed image with generator provenance (prompt/run record or trustworthy benchmark annotation), preferably including generator family and post-processing history.",
        "D": "A licensed original-and-edited pair with an edit mask or operation log for compositing/object removal; preserve export history and exact hashes.",
    }
    routed_attempts = sum(
        item["actual"]["cross_reference"]["external_searches_performed"] for item in evaluated
    )
    traces = [
        trace
        for item in evaluated
        for trace in item["actual"]["cross_reference"]["search_traces"]
    ]
    provider_requests = sum(int(trace.get("requests_made", 0) or 0) for trace in traces)
    returned_results = sum(len(trace.get("results", [])) for trace in traces)
    trace_statuses: dict[str, int] = {}
    for trace in traces:
        status = str(trace.get("status") or "UNKNOWN")
        trace_statuses[status] = trace_statuses.get(status, 0) + 1
    report = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "runner": "tests/run_image_evaluation.py",
        "search_mode": "live_existing_configuration_no_mocks",
        "dataset": {
            "fixture_count": len(evaluated),
            "categories": manifest["categories"],
            "labeled_category_counts": {key: 0 for key in manifest["categories"]},
            "missing_categories": list(manifest["categories"]),
            "fixture_requirements": requirements,
            "canonical_derivatives_excluded": "canonical_image_* files are generated pipeline artifacts or byte duplicates, not independent ground-truth examples.",
        },
        "fixtures": evaluated,
        "search_observations": {
            "primary_provider": "DuckDuckGo HTML search",
            "trusted_domain_filtering": "Configured TRUSTED_DOMAINS are grouped into site:-restricted queries; returned domains are filtered again by is_trusted_domain.",
            "google_factcheck_configured": bool(crossref_module.GOOGLE_FACTCHECK_API_KEY),
            "news_api_configured": bool(crossref_module.NEWS_API_KEY),
            "routed_query_attempts": routed_attempts,
            "provider_requests_made": provider_requests,
            "returned_result_count": returned_results,
            "trace_status_counts": trace_statuses,
            "limitation": "A routed attempt is counted even when the DuckDuckGo circuit breaker prevents an HTTP request; use provider_requests_made for actual network coverage.",
        },
        "metrics": {
            "labeled_example_count": 0,
            "accuracy": None,
            "reason": "Classification accuracy, precision, recall, and confusion matrices are not reported: there are zero independently labeled A-D examples. Reporting a percentage from filenames or mock search responses would be invalid.",
            "software_completion": {
                "completed": sum(item["actual"]["pipeline_status"] == "completed" for item in evaluated),
                "total": len(evaluated),
            },
        },
        "recommendation": "Improve source discovery and corroboration next: expose the provider on every trace, distinguish routed attempts from actual HTTP requests in the API, and make the free provider path resilient enough that one circuit-breaker event does not prevent later fixtures from being checked. This is the largest demonstrated functional gap (queries were constructed for four fixtures but returned no evidence, and later searches made zero provider requests). Validate that change against a small licensed, source-documented set before considering a detector model or paid API.",
    }
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    args.markdown.write_text(render_markdown(report), encoding="utf-8")
    print(f"JSON: {args.json}")
    print(f"Markdown: {args.markdown}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
