# BrandAssist AI

An AI-assisted customer-support tool. It classifies a customer query by intent, retrieves the relevant company policy with semantic search, drafts a reply grounded in that policy, and sends uncertain or sensitive queries to a human agent. A web dashboard lets an agent review each result.

**Live demo:** https://keerthana-jd.github.io/BrandAssist-AI/

> The company, its policies and all figures in the knowledge base are fictional. The project demonstrates the technique, not a real business.

## What it does

1. **Classifies** each query into one of 5 intents: payment, refund, order, delivery, account.
2. **Retrieves** the most relevant policies from a 43-entry knowledge base using sentence embeddings.
3. **Drafts a reply** using only the retrieved policy text (Retrieval-Augmented Generation).
4. **Escalates to a human** when no policy is relevant enough or the query is sensitive (legal threats, fraud, police, complaints).
5. **Dashboard** showing the predicted intent, evidence cards, an editable draft, Approve / Reject / Escalate buttons and a review log.

## Two versions

| | Full version (`ai/`) | Live demo (`web/`) |
|---|---|---|
| Runs on | Laptop (Python) | Visitor's browser |
| Reply writer | Local LLM (Llama 3.2 3B via Ollama) with a "use only these facts" prompt | The top retrieved policy text placed in a reply template |
| Search | FAISS | Cosine similarity in JavaScript (43 entries) |
| Purpose | Real RAG pipeline and evaluation | A free link with no server that can go down |

The live demo also shows the real LLM replies from the full version as clickable examples.

## How it works

```
Customer query
   |-- Intent classifier (TF-IDF + Logistic Regression) --> intent + confidence
   |-- Query embedding (MiniLM) --> nearest policies in the knowledge base --> evidence + match score
   |-- Escalation rules: match score below 0.45, or a sensitive word --> human review
   `-- Otherwise: draft reply from the evidence --> agent approves, edits or rejects
```

## Tech stack

- **Data and ML:** Python, Pandas, scikit-learn (TF-IDF, Logistic Regression)
- **Retrieval:** Sentence Transformers (all-MiniLM-L6-v2), FAISS
- **LLM:** Ollama with Llama 3.2 3B (local and free)
- **Web app:** React, TypeScript, Vite, transformers.js (runs the embedding model in the browser)
- **Hosting:** GitHub Pages (free)

## Results

**Dataset:** Bitext customer-support dataset, 26,872 rows. After cleaning (removing duplicates and placeholders, keeping 5 intents): 19,042 rows.

**Intent classifier on the held-out dataset rows (3,809):** 99% accuracy. The browser version of the classifier (15,000-word vocabulary, 0.22 MB) scored 99.5%. The dataset is synthetic and template-based, so these scores are optimistic.

**Hand-written test set:** 46 questions written by me and worded differently from the knowledge base: 32 answerable from the knowledge base and 14 out-of-scope or risky.

| Measure | Result |
|---|---|
| Intent accuracy, classifier (32 answerable questions) | 59.4% |
| Intent accuracy, taken from the top search result | 90.6% |
| Out-of-scope or risky queries correctly escalated | 14 of 14 |
| Answerable queries that received a draft reply | 32 of 32 |

**Overall decision accuracy (answer vs. escalate) during tuning:**

| Step | Accuracy |
|---|---|
| First run | 89.1% |
| Stopped the weak classifier from overruling strong search evidence | 91.3% |
| Added knowledge-base entries for phrasings that had failed | 100% (46 of 46) |

The last figure is **not an independent measure**: the knowledge-base entries were written after I saw which test questions failed, so the system was tuned on this test set.

## Findings

- A classifier trained on synthetic data scored 99% on similar text but only 59.4% on messy hand-written questions.
- Semantic search identified the intent of those same questions far more reliably (90.6%) than the classifier did.
- Combining both, with match scores deciding when to escalate, kept the system safe: every out-of-scope or risky test query went to a human.
- Reading the generated replies caught a problem the decision metric missed: in one test the local LLM said fast shipping was unavailable although the knowledge base lists express delivery. Metrics that only count answer-versus-escalate decisions cannot detect this, which is why the design keeps a human reviewer in the loop.

## Limitations

- The dataset is synthetic and the company is fictional. A real deployment needs real support history and policies.
- The test set is small (46 questions) and was used for tuning, so the final score overstates real-world performance.
- LLM replies can add small details or, rarely, contradict the knowledge base. I checked replies by reading them but did not compute a formal groundedness score.
- The live demo has no LLM. Its draft replies are built from policy text, so they are less natural than the full version's.
- The review log is saved in each visitor's browser (localStorage) and is not shared.


## Run it yourself

**AI (Python):**
```bash
cd ai
python -m venv venv
venv\Scripts\activate
pip install pandas numpy scikit-learn joblib datasets ipykernel sentence-transformers faiss-cpu ollama
ollama pull llama3.2:3b
```
Download the Bitext dataset CSV from Hugging Face, save it as `ai/raw.csv`, then run the notebooks in order: `01_data.ipynb` (cleaning), `02_classifier.ipynb` (intent classifier), `03_rag.ipynb` (search, RAG, evaluation and export for the web app).

**Web app:**
```bash
cd web
npm install
npm run dev
```

## Project structure

```
ai/    notebooks, knowledge base (kb.csv), test questions, evaluation results
web/   React + TypeScript dashboard, exported model files in public/data
```

## Author

Keerthana J D
