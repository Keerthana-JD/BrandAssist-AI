import { pipeline, env } from '@huggingface/transformers';

env.allowRemoteModels = false;
env.allowLocalModels = true;
env.localModelPath = import.meta.env.BASE_URL + 'models/';

export interface KbEntry { id: number; intent: string; title: string; answer: string; vec: number[]; }
interface ModelJson { classes: string[]; vocab: Record<string, number>; idf: number[]; coef: number[][]; intercept: number[]; }
interface Config { min_score: number; sensitive: string[]; }
export interface Example { question: string; status: string; predicted_intent: string; best_score: number; reasons: string; reply: string; }
export interface Golden { query: string; intent: string; confidence: number; top_title: string; top_score: number; }
export interface Hit { entry: KbEntry; score: number; }
export interface Result {
  query: string; intent: string; confidence: number; hits: Hit[]; bestScore: number;
  status: 'DRAFT_READY' | 'ESCALATE'; reasons: string[]; reply: string;
}
export interface CheckRow { query: string; pyIntent: string; jsIntent: string; intentOk: boolean; pyScore: number; jsScore: number; }

type Embedder = (text: string, opts: object) => Promise<{ data: Float32Array }>;

const base = import.meta.env.BASE_URL;
let kb: KbEntry[] = [];
let model: ModelJson;
let config: Config;
let sensitiveRe: RegExp;
let embedder: Embedder;
let loadPromise: Promise<{ examples: Example[]; golden: Golden[] }> | null = null;

async function getJson<T>(file: string): Promise<T> {
  const res = await fetch(base + 'data/' + file);
  if (!res.ok) throw new Error('Could not load ' + file);
  return (await res.json()) as T;
}

export function loadAll(onStatus: (m: string) => void) {
  if (!loadPromise) loadPromise = doLoad(onStatus);
  return loadPromise;
}

async function doLoad(onStatus: (m: string) => void) {
  onStatus('Loading knowledge base and classifier...');
  kb = await getJson<KbEntry[]>('kb.json');
  model = await getJson<ModelJson>('intent_model.json');
  config = await getJson<Config>('config.json');
  const examples = await getJson<Example[]>('examples.json');
  const golden = await getJson<Golden[]>('golden.json');
  sensitiveRe = new RegExp('\\b(' + config.sensitive.join('|') + ')\\b');

  onStatus('Downloading the search model (first visit only, about 25 MB)...');
  const make = pipeline as unknown as (task: string, name: string, opts: object) => Promise<Embedder>;
  embedder = await make('feature-extraction', 'Xenova/all-MiniLM-L6-v2', { dtype: 'q8' });
  onStatus('Ready');
  return { examples, golden };
}

function features(text: string): Map<number, number> {
  const tokens = text.toLowerCase().match(/\b\w\w+\b/g) ?? [];
  const terms: string[] = [...tokens];
  for (let i = 0; i < tokens.length - 1; i++) terms.push(tokens[i] + ' ' + tokens[i + 1]);
  const f = new Map<number, number>();
  for (const t of terms) {
    if (Object.prototype.hasOwnProperty.call(model.vocab, t)) {
      const idx = model.vocab[t];
      f.set(idx, (f.get(idx) ?? 0) + 1);
    }
  }
  let norm = 0;
  for (const [idx, c] of f) {
    const v = c * model.idf[idx];
    f.set(idx, v);
    norm += v * v;
  }
  norm = Math.sqrt(norm) || 1;
  for (const [idx, v] of f) f.set(idx, v / norm);
  return f;
}

export function classify(q: string) {
  const f = features(q);
  const logits = model.classes.map((_, k) => {
    let s = model.intercept[k];
    for (const [idx, v] of f) s += model.coef[k][idx] * v;
    return s;
  });
  const max = Math.max(...logits);
  const exps = logits.map(l => Math.exp(l - max));
  const sum = exps.reduce((a, b) => a + b, 0);
  const probs = exps.map(e => e / sum);
  const best = probs.indexOf(Math.max(...probs));
  return { intent: model.classes[best], confidence: probs[best] };
}

async function search(q: string, k = 3): Promise<Hit[]> {
  const out = await embedder(q, { pooling: 'mean', normalize: true });
  const v = out.data;
  const hits = kb.map(entry => {
    let dot = 0;
    for (let i = 0; i < v.length; i++) dot += v[i] * entry.vec[i];
    return { entry, score: dot };
  });
  return hits.sort((a, b) => b.score - a.score).slice(0, k);
}

export async function analyze(query: string): Promise<Result> {
  const q = query.toLowerCase().trim();
  const { intent, confidence } = classify(q);
  const hits = await search(query, 3);
  const bestScore = hits[0].score;

  const reasons: string[] = [];
  if (bestScore < config.min_score) reasons.push('no relevant policy found');
  if (sensitiveRe.test(q)) reasons.push('sensitive topic');

  const status = reasons.length ? 'ESCALATE' : 'DRAFT_READY';
  const reply = reasons.length
    ? ''
    : 'Hello, thank you for reaching out. ' + hits[0].entry.answer +
      ' Please let us know if there is anything else we can help you with.';
  return { query, intent, confidence, hits, bestScore, status, reasons, reply };
}

export async function selfCheck(golden: Golden[]): Promise<CheckRow[]> {
  const rows: CheckRow[] = [];
  for (const g of golden) {
    const c = classify(g.query);
    const h = await search(g.query, 1);
    rows.push({
      query: g.query, pyIntent: g.intent, jsIntent: c.intent,
      intentOk: g.intent === c.intent, pyScore: g.top_score, jsScore: h[0].score,
    });
  }
  return rows;
}