# Final decision engine

`decision_engine.run_decision_engine(ingestor_output, provenance_output,
forensics_output, crossref_output)` accepts Python dictionaries and returns a
strictly JSON-serializable dictionary. It has no agent imports, I/O, network,
frontend dependencies, or media-type-specific routing.

`image_pipeline.run_pipeline(input, cookies_file=None)` is the unified coordinator
for local images, local videos, direct image URLs, YouTube URLs, and public
Facebook post URLs. `run_local_image_pipeline` remains a compatibility wrapper.
Both return the pipeline envelope,
agent outputs, timings, and `final_decision`. Its CLI prints the decision using
`json.dumps(decision_output, indent=2)`. Existing agent diagnostic logs can still
precede that final JSON. Early pipeline failures also include a review decision
with all four agent summaries. The legacy `make_final_decision` wrapper accepts
an optional fourth `ingestor` argument; callers omitting it receive Needs Review
because the complete pipeline evidence is unavailable.

## Evidence policy

Evidence strength is represented by decision precedence and coarse distribution
profiles, not by averaging or summing agent scores. There are two dimensions:
file authenticity and factual context. The evidence groups are:

1. Ingestor completion and provenance hash integrity: failed ingestion or a hash
   mismatch requires review. Matching hashes establish custody, not authenticity.
2. Cryptographic provenance: a present, valid C2PA manifest with C2PA_VERIFIED
   and a matching integrity check is strong file provenance evidence. Metadata,
   C2PA flags, and provenance scores do not count as extra independent signals.
3. Forensics: an explicit manipulation classification with confidence >=85/100
   is strong evidence. Numeric 0-1 confidence and High labels are supported.
   Weak flags, aggregate forensic scores, missing modules, and layouts never
   independently trigger manipulation. Existing agent classifications remain
   responsible for corroborating technical findings.
4. Context: only meaningful checked claims with trusted-domain comparisons,
   source URLs, and adequate claim confidence qualify. Contradiction requires
   a DEBUNKED/CONTRADICTED verdict, at least one trusted contradicting source,
   no supporting sources, and confidence >=65. High confidence contextual
   classification requires two contradicting sources, confidence >=75, and
   positive media evidence. Verified context requires two trusted supporting
   sources and a VERIFIED verdict. Counts alone are insufficient.

Claims are grouped by normalized text and sources deduplicated. A group with
conflicting verdicts is not promoted into decisive evidence. Verified claims
cannot establish file authenticity. Strong technical manipulation takes
precedence over context, with reduced confidence when positive provenance or
contextual contradiction coexists. Conflicting context otherwise requires review.
The current demo hash database is retained for inspection, not treated as
production evidence of manipulation or authenticity.

| Evidence profile | Category | Confidence | Authentic / manipulated / misleading context |
| --- | --- | --- | --- |
| Unknown, weak evidence, or failed custody | Needs Review | Low | 35 / 35 / 30 |
| Strong manipulation | Likely Manipulated | High | 5 / 90 / 5 |
| Strong manipulation plus signed provenance | Likely Manipulated | Medium | 25 / 65 / 10 |
| Strong manipulation plus contradicted context | Likely Manipulated | Medium | 5 / 65 / 30 |
| Reliable contradiction | Misleading Context | Medium | 20 / 15 / 65 |
| Strong contradiction plus positive media evidence | Misleading Context | High | 10 / 5 / 85 |
| Signed provenance, uncertain/unavailable forensics | Likely Authentic | Medium | 75 / 10 / 15 |
| Signed provenance, clean forensics | Likely Authentic | High | 85 / 5 / 10 |
| Previous row plus verified context | Likely Authentic | High | 90 / 5 / 5 |

These integers always sum to 100. They are policy-based evidence distributions,
not calibrated probabilities. Needs Review is an abstention category, not a
fourth probability bucket. An authenticity category concerns the media; unresolved
captions and search failures remain explicit limitations even when media evidence
is strong. Valid C2PA does not establish that media was never edited or generated.

## Output schema

| Field | Type / meaning |
| --- | --- |
| `category` | Likely Authentic, Likely Manipulated, Misleading Context, Needs Review |
| `confidence` | High, Medium, Low |
| `likelihood` | Three integer fields: authentic, manipulated, misleading_context |
| `display` | label and semantic color (green/red/orange/yellow) |
| `decision_reasons` | List of reasons for the selected category |
| `reasons` | Compatibility copy of decision_reasons |
| `supporting_evidence`, `negative_evidence` | Lists of interpreted evidence groups |
| `warnings` | Missing coverage, conflicts, agent limitations, calibration caveat |
| `agent_summary` | ingestor, provenance, forensics, cross_reference; each includes availability and relevant original fields |
| `agent_outputs` | Copies of all four original outputs; non-JSON diagnostics converted to strings and nonfinite floats to null |
| `decision_state` | Boolean stage availability, strong_provenance_evidence, strong_manipulation_evidence, contextual_contradiction, contextual_conflict, verified_context, integrity_valid |
| `is_calibrated_probability` | Always false |
| `ready_for_display` | Always true, including review outcomes |

None, missing dictionaries, malformed confidence, and unavailable stages remain
neutral. Inputs are not mutated. No JSON is passed between agents.

## Validation

Run the deterministic policy tests:

```powershell
.venv/Scripts/python.exe -B -m unittest discover -s tests -v
```

Run the existing image agents on local fixtures:

```powershell
.venv/Scripts/python.exe -B tests/run_image_regressions.py
```

The image runner copies fixtures into a temporary directory so canonicalization
does not overwrite workspace images. Seven cases cover JPEG, PNG, social
screenshot, newspaper/document, metadata presence, validated Adobe C2PA, and
zero meaningful claims. JPEG and metadata cases deliberately share the same
fixture. Full outputs are saved to `result/decision_engine_regressions.json`.
No pre-existing automated test suite was present.

Multilingual OCR requires the external Tesseract `eng` and `ben` trained-data
packs. The agent enumerates installed languages, uses `ben+eng` when both exist,
and reports partial or unavailable coverage when a required pack is missing.

Remaining limits: upstream forensics and claim comparisons are heuristic;
source trust is supplied by Cross-Reference; missing external search or optional
dependencies limit coverage. Near-duplicate visual matches and cheapfake layout
flags alone do not establish misleading context. The engine accepts image,
video, and audio outputs, but the existing coordinator remains image-specific;
this change does not add URL/video orchestration or change the video agents.

Validation on 2026-09-08: 21 automated policy/coordinator tests passed; all seven
image cases passed. The signed Adobe JPEG returned Likely Authentic / High /
85-5-10 with C2PA_VERIFIED and matching integrity. Other image cases returned
Needs Review / Low / 35-35-30. Full video pipeline execution was not tested.
