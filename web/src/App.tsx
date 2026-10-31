import { useEffect, useState } from 'react';
import { loadAll, analyze, selfCheck } from './ai';
import type { Result, Example, Golden, CheckRow } from './ai';
import './App.css';

interface Review { id: number; query: string; intent: string; decision: string; reply: string; time: string; }
const STORE = 'brandassist-reviews';

function loadReviews(): Review[] {
  try { return JSON.parse(localStorage.getItem(STORE) ?? '[]') as Review[]; } catch { return []; }
}

export default function App() {
  const [status, setStatus] = useState('Starting...');
  const [ready, setReady] = useState(false);
  const [examples, setExamples] = useState<Example[]>([]);
  const [golden, setGolden] = useState<Golden[]>([]);
  const [query, setQuery] = useState('');
  const [busy, setBusy] = useState(false);
  const [result, setResult] = useState<Result | null>(null);
  const [draft, setDraft] = useState('');
  const [reviews, setReviews] = useState<Review[]>(loadReviews);
  const [check, setCheck] = useState<CheckRow[] | null>(null);

  useEffect(() => {
    loadAll(setStatus)
      .then(d => { setExamples(d.examples); setGolden(d.golden); setReady(true); })
      .catch(e => setStatus('Could not load the AI: ' + String(e)));
  }, []);

  async function run(q: string) {
    const text = q.trim();
    if (!text) return;
    setBusy(true);
    try {
      const r = await analyze(text);
      setResult(r);
      setDraft(r.reply);
    } catch (e) {
      setStatus('Error: ' + String(e));
    }
    setBusy(false);
  }

  function decide(decision: string) {
    if (!result) return;
    const item: Review = {
      id: Date.now(), query: result.query, intent: result.intent, decision,
      reply: decision === 'Escalated' ? '' : draft, time: new Date().toLocaleString(),
    };
    const next = [item, ...reviews].slice(0, 50);
    setReviews(next);
    try { localStorage.setItem(STORE, JSON.stringify(next)); } catch { /* ignore */ }
    setResult(null);
    setQuery('');
  }

  const ollamaReply = result
    ? examples.find(e => e.question.toLowerCase() === result.query.toLowerCase())?.reply
    : '';

  return (
    <div className="page">
      <header>
        <h1>BrandAssist AI</h1>
        <p>Customer-support assistant: intent classification, semantic search, grounded reply drafts and human review.</p>
        <p className="note">
          Live demo runs fully in your browser. Reply drafts here are built from the retrieved policy text;
          the full version (see GitHub) writes replies with a local LLM using RAG.
        </p>
      </header>

      {!ready && <div className="loading">{status}</div>}

      <section className="card">
        <h2>Customer query</h2>
        <div className="row">
          <input value={query} placeholder="e.g. I want my money back"
            onChange={e => setQuery(e.target.value)}
            onKeyDown={e => { if (e.key === 'Enter') run(query); }} />
          <button disabled={!ready || busy} onClick={() => run(query)}>{busy ? 'Working...' : 'Analyze'}</button>
        </div>
        <div className="chips">
          {examples.filter((_, i) => i % 6 === 0).map(ex => (
            <button key={ex.question} className="chip" disabled={!ready || busy}
              onClick={() => { setQuery(ex.question); run(ex.question); }}>{ex.question}</button>
          ))}
        </div>
      </section>

      {result && (
        <section className="card">
          <div className="row between">
            <h2>Result</h2>
            <span className={'badge ' + (result.status === 'ESCALATE' ? 'esc' : 'ok')}>
              {result.status === 'ESCALATE' ? 'Needs human review' : 'Draft ready'}
            </span>
          </div>
          <p><b>Predicted intent:</b> {result.intent} ({Math.round(result.confidence * 100)}% confidence)</p>
          <div className="bar"><div style={{ width: result.confidence * 100 + '%' }} /></div>
          {result.reasons.length > 0 && <p className="warn">Escalation reason: {result.reasons.join('; ')}</p>}

          <h3>Evidence from the knowledge base</h3>
          {result.hits.map(h => (
            <div className="evidence" key={h.entry.id}>
              <b>{h.entry.title}</b> <span className="score">match {h.score.toFixed(2)}</span>
              <div>{h.entry.answer}</div>
            </div>
          ))}

          {result.status === 'DRAFT_READY' && (
            <>
              <h3>Draft reply (editable)</h3>
              <textarea rows={4} value={draft} onChange={e => setDraft(e.target.value)} />
            </>
          )}
          {ollamaReply && (
            <div className="llm"><b>Reply written by the local LLM (Ollama) for this example:</b><div>{ollamaReply}</div></div>
          )}

          <div className="row">
            {result.status === 'DRAFT_READY' && <button onClick={() => decide('Approved')}>Approve</button>}
            {result.status === 'DRAFT_READY' && <button className="secondary" onClick={() => decide('Rejected')}>Reject</button>}
            <button className="secondary" onClick={() => decide('Escalated')}>Escalate to human</button>
          </div>
        </section>
      )}

      <section className="card">
        <h2>Review log</h2>
        {reviews.length === 0 && <p className="muted">No reviews yet. Approve, reject or escalate a result to see it here.</p>}
        {reviews.map(r => (
          <div className="evidence" key={r.id}>
            <b>{r.decision}</b> <span className="score">{r.intent} - {r.time}</span>
            <div>{r.query}</div>
          </div>
        ))}
      </section>

      <section className="card">
        <h2>Self-check against the Python model</h2>
        <p className="muted">Compares this browser version with the results saved from Python.</p>
        <button className="secondary" disabled={!ready} onClick={() => selfCheck(golden).then(setCheck)}>Run self-check</button>
        {check && (
          <table>
            <thead><tr><th>Query</th><th>Python intent</th><th>Browser intent</th><th>Python score</th><th>Browser score</th></tr></thead>
            <tbody>
              {check.map(c => (
                <tr key={c.query} className={c.intentOk ? '' : 'bad'}>
                  <td>{c.query}</td><td>{c.pyIntent}</td><td>{c.jsIntent}</td>
                  <td>{c.pyScore.toFixed(3)}</td><td>{c.jsScore.toFixed(3)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </section>
    </div>
  );
}