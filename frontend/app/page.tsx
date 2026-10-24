"use client";

import { ChangeEvent, DragEvent, useEffect, useRef, useState } from "react";

type Json = Record<string, unknown>;
const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://127.0.0.1:8000";
const IMAGE_TYPES = ["image/jpeg", "image/png", "image/webp", "image/bmp", "image/gif"];
const AGENTS = [
  ["ingestor", "Ingestor"], ["provenance", "Provenance Agent"],
  ["forensics", "Forensics Agent"], ["cross_reference", "Cross-Reference Agent"],
] as const;

function asRecord(value: unknown): Json { return value && typeof value === "object" && !Array.isArray(value) ? value as Json : {}; }
function asList(value: unknown): unknown[] { return Array.isArray(value) ? value : []; }
function text(value: unknown): string { return typeof value === "string" || typeof value === "number" ? String(value) : "Not available"; }
function pretty(value: string): string { return value.replaceAll("_", " ").replace(/\b\w/g, (c) => c.toUpperCase()); }
function colorFor(category: unknown): string {
  const v = String(category).toLowerCase();
  if (v.includes("authentic")) return "good";
  if (v.includes("manipulated")) return "risk";
  if (v.includes("misleading")) return "context";
  return "review";
}

export default function Home() {
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [result, setResult] = useState<Json | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const input = useRef<HTMLInputElement>(null);

  useEffect(() => () => { if (preview) URL.revokeObjectURL(preview); }, [preview]);
  function select(candidate?: File) {
    setError(null); setResult(null);
    if (!candidate) return;
    if (!IMAGE_TYPES.includes(candidate.type)) { setFile(null); setPreview(null); setError("Choose a JPEG, PNG, WebP, BMP, or GIF image."); return; }
    if (candidate.size > 50 * 1024 * 1024) { setFile(null); setPreview(null); setError("Images must be 50 MB or smaller."); return; }
    if (preview) URL.revokeObjectURL(preview);
    setFile(candidate); setPreview(URL.createObjectURL(candidate));
  }
  function reset() { if (preview) URL.revokeObjectURL(preview); setFile(null); setPreview(null); setResult(null); setError(null); if (input.current) input.current.value = ""; }
  async function analyze() {
    if (!file) return;
    setLoading(true); setError(null); setResult(null);
    try {
      const form = new FormData(); form.append("file", file);
      const response = await fetch(`${API_URL}/api/analyze-image`, { method: "POST", body: form });
      const body: unknown = await response.json().catch(() => null);
      if (!response.ok) throw new Error(text(asRecord(body).detail) === "Not available" ? `Request failed (${response.status})` : text(asRecord(body).detail));
      if (!body || typeof body !== "object") throw new Error("The API returned an invalid response.");
      setResult(body as Json);
    } catch (cause) { setError(cause instanceof Error ? cause.message : "Could not reach the TrueLens API."); }
    finally { setLoading(false); }
  }
  const decision = asRecord(result?.final_decision);
  const likelihood = asRecord(decision.likelihood);
  const mediaAssessment = asRecord(decision.media_authenticity_assessment);
  const contextAssessment = asRecord(decision.claim_context_assessment);
  const agents = asRecord(result?.agents);
  const crossref = asRecord(agents.cross_reference);
  const ocr = asRecord(crossref.ocr);
  const claims = asList(crossref.claim_results).map(asRecord);
  const sources = asList(crossref.sources_used).map(asRecord);
  const evidence = (name: string) => asList(decision[name]).filter((item): item is string => typeof item === "string");
  return <main>
    <header><a className="brand" href="#top"><span>◈</span> TrueLens</a><p>AI-powered media verification</p><div className="local">● Local analysis</div></header>
    <section id="top" className="hero"><div><p className="eyebrow">EVIDENCE, NOT ASSUMPTIONS</p><h1>See beyond<br />the image.</h1><p className="lede">TrueLens examines image evidence across ingestion, provenance, forensics, and cross-reference stages—then presents the combined decision transparently.</p></div><aside><span className="orbit">◌</span><p>Four evidence stages</p><strong>One reviewable decision</strong></aside></section>
    <section className="upload-card" onDragOver={(e: DragEvent) => e.preventDefault()} onDrop={(e: DragEvent) => { e.preventDefault(); select(e.dataTransfer.files[0]); }}>
      <div className="section-title"><div><p className="eyebrow">IMAGE ANALYSIS</p><h2>Upload an image to review</h2></div>{file && <button className="quiet" onClick={reset}>Reset</button>}</div>
      {!file ? <button className="dropzone" onClick={() => input.current?.click()}><span className="upload-mark">↑</span><strong>Drop an image here, or browse files</strong><small>JPEG, PNG, WebP, BMP, or GIF · maximum 50 MB</small></button> : <div className="selected"><img src={preview ?? ""} alt="Selected image preview" /><div><strong>{file.name}</strong><p>{(file.size / 1024 / 1024).toFixed(2)} MB · ready for analysis</p><button className="quiet" onClick={() => input.current?.click()}>Choose another</button></div></div>}
      <input ref={input} type="file" accept={IMAGE_TYPES.join(",")} hidden onChange={(e: ChangeEvent<HTMLInputElement>) => select(e.target.files?.[0])} />
      {error && <p className="error" role="alert">{error}</p>}
      <button className="analyze" disabled={!file || loading} onClick={analyze}>{loading ? "Analyzing evidence…" : "Analyze image"}<span>→</span></button>
      {loading && <p className="processing">Running the existing verification stages. This can take a moment.</p>}
    </section>
    {result && <section className="results" aria-live="polite"><div className="section-title"><div><p className="eyebrow">ANALYSIS RESULT</p><h2>Decision Engine output</h2></div><span className="timestamp">{text(result.processed_at)}</span></div>
      <article className={`verdict ${colorFor(decision.category)}`}><div><p className="eyebrow">FINAL CATEGORY</p><h2>{text(decision.category)}</h2><p className="confidence">Confidence: <strong>{text(decision.confidence)}</strong></p><small>{text(decision.likelihood_note)}</small></div><div className="distribution">{Object.entries(likelihood).map(([name, value]) => <div key={name}><div><span>{pretty(name)}</span><strong>{text(value)}%</strong></div><i><b style={{ width: `${Number(value) || 0}%` }} /></i></div>)}</div></article>
      <div className="evidence-grid"><Assessment title="Media authenticity" value={mediaAssessment} /><Assessment title="Claim & context" value={contextAssessment} /></div>
      <div className="evidence-grid"><Evidence title="Reasons" items={evidence("decision_reasons")} /><Evidence title="Supporting evidence" items={evidence("supporting_evidence")} /><Evidence title="Negative evidence" items={evidence("negative_evidence")} /><Evidence title="Warnings & limits" items={evidence("warnings")} /></div>
      {(typeof ocr.reconstructed_headline === "string" || typeof ocr.text === "string") && <article className="evidence"><h3>Extracted text</h3>{typeof ocr.reconstructed_headline === "string" && ocr.reconstructed_headline.length > 0 && <p><strong>Headline:</strong> {ocr.reconstructed_headline}</p>}<p>{text(ocr.text)}</p><small>OCR status: {text(ocr.status)} · quality: {text(ocr.quality_score)}</small></article>}
      {!!claims.length && <article className="evidence"><h3>Checked claims</h3><ul>{claims.map((claim, index) => <li key={index}><strong>{text(claim.verdict)}</strong>: {text(claim.claim ?? claim.text)}</li>)}</ul></article>}
      {!!sources.length && <article className="evidence"><h3>Sources</h3><ul>{sources.map((source, index) => <li key={index}>{typeof source.url === "string" ? <a href={source.url} target="_blank" rel="noreferrer">{text(source.title ?? source.url)}</a> : text(source.title)}</li>)}</ul></article>}
      <div className="agent-section"><div className="section-title"><div><p className="eyebrow">EVIDENCE STAGES</p><h2>Agent outputs</h2></div><p className="note">The Decision Engine applies evidence rules; it does not average agent scores.</p></div><div className="agents">{AGENTS.map(([key, label]) => <AgentCard key={key} name={label} output={asRecord(agents[key])} />)}</div></div>
    </section>}
  </main>;
}
function Evidence({ title, items }: { title: string; items: string[] }) { return <article className="evidence"><h3>{title}</h3>{items.length ? <ul>{items.map((item, index) => <li key={index}>{item}</li>)}</ul> : <p>No {title.toLowerCase()} were returned.</p>}</article>; }
function Assessment({ title, value }: { title: string; value: Json }) { return <article className="evidence"><h3>{title}</h3><p><strong>{text(value.status)}</strong> · {text(value.confidence)} confidence</p><p>{text(value.reason)}</p></article>; }
function AgentCard({ name, output }: { name: string; output: Json }) { const entries = Object.entries(output).filter(([key, value]) => !["raw_findings", "explainability", "claim_results"].includes(key) && (typeof value === "string" || typeof value === "number" || typeof value === "boolean")); return <article className="agent"><div><h3>{name}</h3><span className={String(output.status).toLowerCase() === "error" ? "bad-status" : "status"}>{text(output.status)}</span></div>{entries.length ? <dl>{entries.slice(0, 5).map(([key, value]) => <div key={key}><dt>{pretty(key)}</dt><dd>{text(value)}</dd></div>)}</dl> : <p>No summary fields returned.</p>}<details><summary>View raw output</summary><pre>{JSON.stringify(output, null, 2)}</pre></details></article>; }
