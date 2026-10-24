# TrueLens image capability evaluation

Generated: `2026-10-20T17:36:12.121604+00:00`

This is an observational run of the real pipeline. External search was not mocked. A successful software run is not counted as correct detection unless the fixture has independent ground truth.

## Dataset validity and coverage

- Files evaluated: 8
- Independently labeled A-D examples: 0
- Accuracy estimate: None
- Missing categories: A, B, C, D
- The Adobe asset is a verified C2PA parser/provenance control, not a real-world A-D content label.
- Repository filenames, logos, and test method names were not accepted as ground truth.

## Fixture results

| Fixture | SHA-256 | Label status | Pipeline | OCR quality | Searches / status | Media assessment | Context/final |
| --- | --- | --- | --- | ---: | --- | --- | --- |
| `adobe-20220124-C.jpg` | `75a8da33f6eaf1e16bf3b42cd78913b22b2e6a671fda217a508b1ba4230ce864` | verified_control_only | completed | 0.0 | 0 / NOT_CHECKED | AUTHENTIC | ; **Likely Authentic** |
| `c2pa_adobe_positive.jpg` | `9d33d48863ac4f94711e289bebc43e849d45be1819ee16c479bd9a8385f1ae08` | unverified | completed | 74.0 | 9 / INSUFFICIENT_EVIDENCE | UNCERTAIN | ['INSUFFICIENT_EVIDENCE', 'INSUFFICIENT_EVIDENCE', 'INSUFFICIENT_EVIDENCE']; **Needs Review** |
| `carson-screenshot8_0.png` | `dd6351e0d6b59732cd9fb988c1db78926d66ade05673ffcf2c9a727356e2f903` | unverified | completed | 75.9 | 0 / NOT_CHECKED | UNCERTAIN | ; **Needs Review** |
| `imageprocessor.jpeg` | `9a2ec38ac3569da8f94e637800d656ef15842e810f9133c5b7c2efbc6d771c3c` | unverified | completed | 72.6 | 4 / SEARCH_FAILED | UNCERTAIN | ['SEARCH_FAILED', 'SEARCH_FAILED', 'SEARCH_FAILED', 'SEARCH_FAILED']; **Needs Review** |
| `chek.jpeg` | `590e1c060284ca933c0b4299dc5e41fbfcc45d1a969d2ee0704296b74d7705ff` | unverified | completed | 66.1 | 2 / SEARCH_FAILED | UNCERTAIN | ['SEARCH_FAILED']; **Needs Review** |
| `images.jpeg` | `22b6b4530bb2d698f0fd73f698dcf0efdb8ab4484cb73035ce4489baebb2743b` | unverified | completed | 63.6 | 0 / NOT_CHECKED | UNCERTAIN | ; **Needs Review** |
| `newsntvfake.jpg` | `e2e70f4af6677f20deaa284fab9907e35d4d70f21d8c6ec6eaa36ba4b05e4c5b` | unverified | completed | 67.5 | 2 / SEARCH_FAILED | UNCERTAIN | ['SEARCH_FAILED']; **Needs Review** |
| `fakenews.webp` | `8b5e20f4e60a03c42b88a26f3636b06145980a1656039112232010e1e2b743b3` | unverified | completed | 72.3 | 0 / NOT_CHECKED | UNCERTAIN | ; **Needs Review** |

### `adobe-20220124-C.jpg`

- Category: `None`; role: `provenance_control`; label: `verified_control_only`.
- Ground truth/source: Official C2PA public test asset with a valid simple claim; this establishes the expected C2PA validation behavior, not whether the depicted pixels are news, AI…
- Source URL: https://github.com/c2pa-org/public-testfiles/blob/main/legacy/1.4/image/jpeg/adobe-20220124-C.jpg
- License: https://github.com/c2pa-org/public-testfiles/blob/main/LICENSE
- Expected media behavior: Valid C2PA and asset integrity should be reported; no content-class label is justified.
- Expected context behavior: No factual news-context conclusion is expected.
- Raw OCR: ``
- Reconstructed headline: ``
- Claims: 0; searches: 0; provider status: `NOT_CHECKED`.
- Final: `Likely Authentic` / `High`; trigger: `None`.
- Evidence-supported behavior: `True`. Accuracy excluded because Control fixture tests provenance parsing, not an A-D content label.
- Full OCR passes, claims/source text, query records, provider results, provenance, forensic findings, warnings, and the complete pipeline envelope are retained in the JSON report.

### `c2pa_adobe_positive.jpg`

- Category: `None`; role: `unlabeled_repository_fixture`; label: `unverified`.
- Ground truth/source: None documented in this repository.
- Source URL: not documented
- License: not documented; do not redistribute until cleared
- Expected media behavior: Abstain without independent ground truth.
- Expected context behavior: Abstain without independently verified context.
- Raw OCR: `২572 ple নাক ৯ 82231272411 7 TST চিত ER হর 25534754 KER Se Ae ADS ERA gOS 71755 পরা 1227779৮47 Be কি Ee Pees pe দত গ্রে Roe তা nen টি ই ES, রি তি ae ats a aa eg Serres 75525 চি te on সাও ee OLA AE ett ক ge 2 হন এ মগ ৮০৯ ERAS উন সি ০ লা ০ বুলি OE AT or iy il রর সী ইতি, ১৩ ভিত টি ee 1 Bop ee : Sela. …`
- Reconstructed headline: `“AR Sa Sa 1.) yrs LOD a OE Cet he Fees শি আস`
- Claims: 3; searches: 9; provider status: `INSUFFICIENT_EVIDENCE`.
- Final: `Needs Review` / `Low`; trigger: `NO_DECISIVE_PROVENANCE_FORENSIC_OR_CONTEXTUAL_EVIDENCE`.
- Evidence-supported behavior: `True`. Accuracy excluded because No independently established category label or licensed source record.
- Full OCR passes, claims/source text, query records, provider results, provenance, forensic findings, warnings, and the complete pipeline envelope are retained in the JSON report.

### `carson-screenshot8_0.png`

- Category: `None`; role: `unlabeled_repository_fixture`; label: `unverified`.
- Ground truth/source: None documented in this repository.
- Source URL: not documented
- License: not documented; do not redistribute until cleared
- Expected media behavior: Abstain without independent ground truth.
- Expected context behavior: Abstain without independently verified context.
- Raw OCR: `VAL THOR @ @CMDRVALTHOR Please keep North Carolina and Tennessee in your prayers. Don't act like you didn't see this. | said the world needs God now, more than ever! Amen. Matthew 6:33 “But seek ye first the kingdom of God, and his righteousness; and all these things shall be added unto you.” 1:58 …`
- Reconstructed headline: `Please keep North Carolina and Tennessee in your prayers. Don't act like you didn't see this. | said the world needs God now, more than ever! Matthew 6:33 “But seek ye first the kingdom of God, and his`
- Claims: 0; searches: 0; provider status: `NOT_CHECKED`.
- Final: `Needs Review` / `Low`; trigger: `NO_DECISIVE_PROVENANCE_FORENSIC_OR_CONTEXTUAL_EVIDENCE`.
- Evidence-supported behavior: `True`. Accuracy excluded because No independently established category label or licensed source record.
- Full OCR passes, claims/source text, query records, provider results, provenance, forensic findings, warnings, and the complete pipeline envelope are retained in the JSON report.

### `imageprocessor.jpeg`

- Category: `None`; role: `unlabeled_repository_fixture`; label: `unverified`.
- Ground truth/source: None documented in this repository.
- Source URL: not documented
- License: not documented; do not redistribute until cleared
- Expected media behavior: Abstain without independent ground truth.
- Expected context behavior: Abstain without independently verified context.
- Raw OCR: `Bangladesh Updat Monday, January 15, 2024 SHOWBIZ Magh 01, 1430, Rajab 03, 1445 পলা, Ferdous, Jyoti pay team up for নন Julie Pe Showbiz Desk ২৬ fie he newly elected member of the parlamert Fevdous 12)? শব ৩0890 সা গানও লে 8 NOW fim ক “ae film wil also 13 Jy ৪165 actress ws rector of Publi Hubba' to…`
- Reconstructed headline: `ioe Dotan, eee be eee i 114 is ] married Bere eos tore im rane seme nioradress nice gira rat une Sis pe, Mon nh ey Keaiunemervnpercerncees Imes sequels of 2`
- Claims: 4; searches: 4; provider status: `SEARCH_FAILED`.
- Final: `Needs Review` / `Low`; trigger: `NO_DECISIVE_PROVENANCE_FORENSIC_OR_CONTEXTUAL_EVIDENCE`.
- Evidence-supported behavior: `True`. Accuracy excluded because No independently established category label or licensed source record.
- Full OCR passes, claims/source text, query records, provider results, provenance, forensic findings, warnings, and the complete pipeline envelope are retained in the JSON report.

### `chek.jpeg`

- Category: `None`; role: `candidate_news_graphic`; label: `unverified`.
- Ground truth/source: None documented in this repository.
- Source URL: not documented
- License: not documented; do not redistribute until cleared
- Expected media behavior: A logo or document layout must not establish authenticity.
- Expected context behavior: Keep Needs Review unless external evidence supports or contradicts a coherent claim.
- Raw OCR: `পন ; হাসিন ©) eae Ni -. a ~ SUG [ie ¢ ইসি ৬৮] টি ্ © 627, রী 4 Pes age, ক, 12 1 টি mS < for} a সহ তা \ %, DHAKA eH i eo ২ PTD gt ta, = C ১ ২5 সার, ০:৮১1% wet. | A টিটি হর ৯২২ 1 DEO oo EERE’ 741 Mag Woe 1 foie লট রি NY 2 gees LN Ges) (a ed Zs Th ws (em 4 ic 5 নিট ৯ ্ Lapel on . ‘ 3 ae ine Beige AOS …`
- Reconstructed headline: `ঢাকার দুই সিটিতে ত্রিমুখী লড়াই দেখার অপেক্ষায় রাজ দেখার অপেক্ষায় রাজধানীবাসী`
- Claims: 1; searches: 2; provider status: `SEARCH_FAILED`.
- Final: `Needs Review` / `Low`; trigger: `NO_DECISIVE_PROVENANCE_FORENSIC_OR_CONTEXTUAL_EVIDENCE`.
- Evidence-supported behavior: `True`. Accuracy excluded because No independently established category label or licensed source record.
- Full OCR passes, claims/source text, query records, provider results, provenance, forensic findings, warnings, and the complete pipeline envelope are retained in the JSON report.

### `images.jpeg`

- Category: `None`; role: `candidate_ai_image`; label: `unverified`.
- Ground truth/source: A test method calls this an AI flood image, but supplies no generator record, original source, or dataset citation; the filename/test name is not ground truth.
- Source URL: not documented
- License: not documented; do not redistribute until cleared
- Expected media behavior: Do not score as category C without independently established ground truth.
- Expected context behavior: No context conclusion is expected without a supported claim.
- Raw OCR: `ies { 1017 OE ১ —— > - yo t . au দে ূ - . ae =a" ৮.৮... (কে 2,155 a “ah - তি ৬ আত এ ক SSE Bo ee == io me Le 2 LI = sae স্‌ a . ২ তে? | ৩ আপার ০ গুটি 1১০০০: EGE ies te a ad o ক am by Cr 7, স্ব -ললপাশ্লিটি। ০০০ ১০০০৯ eatin`
- Reconstructed headline: `a “ah - তি ৬ আত এ ৩ আপার ০ গুটি`
- Claims: 0; searches: 0; provider status: `NOT_CHECKED`.
- Final: `Needs Review` / `Low`; trigger: `INCONCLUSIVE_SUSPICIOUS_FORENSICS`.
- Evidence-supported behavior: `True`. Accuracy excluded because No independently established category label or licensed source record.
- Full OCR passes, claims/source text, query records, provider results, provenance, forensic findings, warnings, and the complete pipeline envelope are retained in the JSON report.

### `newsntvfake.jpg`

- Category: `None`; role: `candidate_news_graphic`; label: `unverified`.
- Ground truth/source: The filename is not evidence that the image or its election claim is false.
- Source URL: not documented
- License: not documented; do not redistribute until cleared
- Expected media behavior: Abstain unless provenance or strong technical evidence supports a conclusion.
- Expected context behavior: Keep Needs Review when trusted-source corroboration is absent.
- Raw OCR: `aww ie Bs as os PY at he \ জারা সহ eS আপনার বিশ্বস্ত সময় নির্বাচনের সময় : সন্ব্যা ৭টা ১৩ ফেব্রুয়ারি ২০২৬ co ENT EDU উঠ 1) eee |). টিসি ১ | Ret | IN Py gag জিভ |`
- Reconstructed headline: `\ জারা আপনার বিশ্বস্ত নির্বাচনের সময় : সন্ব্যা ৭টা`
- Claims: 1; searches: 2; provider status: `SEARCH_FAILED`.
- Final: `Needs Review` / `Low`; trigger: `NO_DECISIVE_PROVENANCE_FORENSIC_OR_CONTEXTUAL_EVIDENCE`.
- Evidence-supported behavior: `True`. Accuracy excluded because No independently established category label or licensed source record.
- Full OCR passes, claims/source text, query records, provider results, provenance, forensic findings, warnings, and the complete pipeline envelope are retained in the JSON report.

### `fakenews.webp`

- Category: `None`; role: `candidate_fact_check_graphic`; label: `unverified`.
- Ground truth/source: The filename and visible branding are not independent ground truth.
- Source URL: not documented
- License: not documented; do not redistribute until cleared
- Expected media behavior: A publisher logo must not establish authenticity.
- Expected context behavior: Keep Needs Review unless external evidence supports or contradicts a coherent claim.
- Raw OCR: `Qe 25 ESTES LR SS ne মা 228 রি বকে হি রর কি হা ABB ees, Ud Ae ই, ee ieee Barca 2 পি £ ot 8 OP NINE UG ANE he OEE 8288278১৬২9 vate 3 Wee মতা 2 ১৮২১] চা পট poke eo ই তত BE ot 7 a ১০২৮1 বরা ‘Ves, ৮৩ জপ চান weit পা টি " [215১18217৮০ av eee 18 TaN 5 Ete 1225 -8868-1417031 ১8৮ নি যা IP aba 2 কি 138 (ও ৫ …`
- Reconstructed headline: `Qe 25 ESTES LR SS ne মা 228 রি বকে হি £ peoplesamid.the ongoing Israeli, genocide,in,Gazay`
- Claims: 0; searches: 0; provider status: `NOT_CHECKED`.
- Final: `Needs Review` / `Low`; trigger: `NO_DECISIVE_PROVENANCE_FORENSIC_OR_CONTEXTUAL_EVIDENCE`.
- Evidence-supported behavior: `True`. Accuracy excluded because No independently established category label or licensed source record.
- Full OCR passes, claims/source text, query records, provider results, provenance, forensic findings, warnings, and the complete pipeline envelope are retained in the JSON report.

## Missing benchmark fixtures

- **A:** A legally reusable news graphic plus publisher/origin record and at least one independent authoritative corroboration of its claim and publication date.
- **B:** A licensed genuine photograph with documented original event/date/location and a preserved false caption or repost, supported by an authoritative debunk or source comparison.
- **C:** A licensed image with generator provenance (prompt/run record or trustworthy benchmark annotation), preferably including generator family and post-processing history.
- **D:** A licensed original-and-edited pair with an edit mask or operation log for compositing/object removal; preserve export history and exact hashes.

## Metrics

Classification accuracy, precision, recall, and confusion matrices are not reported: there are zero independently labeled A-D examples. Reporting a percentage from filenames or mock search responses would be invalid.

Operationally, 17 routed query attempts produced 10 actual provider requests and 0 returned results. Provider: DuckDuckGo HTML search.

## Next engineering step

Improve source discovery and corroboration next: expose the provider on every trace, distinguish routed attempts from actual HTTP requests in the API, and make the free provider path resilient enough that one circuit-breaker event does not prevent later fixtures from being checked. This is the largest demonstrated functional gap (queries were constructed for four fixtures but returned no evidence, and later searches made zero provider requests). Validate that change against a small licensed, source-documented set before considering a detector model or paid API.

## Reproduction

```powershell
.venv/Scripts/python.exe -B tests/run_image_evaluation.py
```

The run is intentionally live: provider availability and indexed results are timestamped evidence, not deterministic ground truth. Unit tests continue to mock external services.
