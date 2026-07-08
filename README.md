# GitHub Repository RAG Agent

Paste in a GitHub repo URL, and chat with it. The agent clones the repo,
splits it into meaningful chunks, embeds them locally, stores them in a
local vector database, and answers your questions using hybrid (semantic +
keyword) retrieval over the actual code -- with citations you can click to
preview. It can also auto-generate documentation, a module dependency
diagram, and a dependency/language breakdown.

Built for: Windows, Python 3.12, CPU-only, $0 budget.

## Features

- **Chat with a repo**, with conversational memory per session.
- **Language-aware chunking**: Python files are split by function/class via
  the `ast` module; common curly-brace languages (JS/TS/Java/Go/etc.) use a
  lightweight heuristic; everything else falls back to a sliding window.
  Every chunk carries line numbers.
- **Hybrid retrieval**: combines semantic search (embeddings via Chroma)
  with keyword search (BM25), merged with Reciprocal Rank Fusion -- catches
  exact function/variable names that pure semantic search can miss.
- **Click-to-preview citations**: every answer lists its source files with
  line ranges; click one to see the exact code snippet used.
- **Auto-generated documentation**: a Markdown project summary (Overview,
  Key Modules, Architecture, Dependencies), downloadable as `README.md`.
- **Architecture diagram**: a Mermaid graph of which local Python modules
  import which other local modules (falls back to a folder map for
  non-Python repos).
- **Dependency & language analysis**: language breakdown by file count,
  plus parsed `requirements.txt` / `pyproject.toml` / `package.json`.
- **Background indexing with a progress bar**: indexing runs in a
  background thread on the server; the UI polls and shows live stage
  progress (cloning -> parsing -> embedding -> storing) instead of one
  long blocking spinner.

## How it works

1. You give it a GitHub URL. Indexing starts in the background immediately.
2. The repo is cloned and split into chunks (function/class-aware where
   possible), each tagged with its file and line range.
3. Chunks are embedded locally (CPU-friendly model, no API cost) and stored
   in Chroma; a BM25 keyword index is built alongside it in memory.
4. You ask a question. The agent retrieves the most relevant chunks via
   hybrid search and sends them, plus your recent chat history, to Groq's
   free/fast LLM API to generate an answer.
5. The answer comes back with clickable citations to the exact source
   files and lines it used.

Only step 4 leaves your machine -- everything else runs locally for free.

## Prerequisites

- **Python 3.12** -- you already have this.
- **Git** installed and on your PATH (needed to clone repos). Check with `git --version`.
- **A free Groq API key** -- no credit card required. Get one at
  https://console.groq.com/keys

## Setup

1. Open a terminal in the project folder.

2. Create and activate a virtual environment:
   ```
   python -m venv venv
   venv\Scripts\activate
   ```

3. Install dependencies (this also downloads the ~80MB embedding model the
   first time it runs):
   ```
   pip install -r requirements.txt
   ```

4. Copy the environment file and add your Groq key:
   ```
   copy .env.example backend\.env
   ```
   Open `backend\.env` and paste your key into `GROQ_API_KEY=`.

## Running it

You need **two terminals** open at the same time (both with the venv activated).

**Terminal 1 -- backend:**
```
cd backend
uvicorn main:app --reload --port 8000
```
Leave this running. You should see `Uvicorn running on http://127.0.0.1:8000`.

**Terminal 2 -- frontend:**
```
cd frontend
streamlit run app.py
```
This opens the chat UI in your browser (usually http://localhost:8501). It
needs an internet connection to load the Mermaid diagram library from a
CDN when you use the Architecture tab.

## Using it

1. In the sidebar, paste a GitHub URL (e.g. `https://github.com/pallets/click`)
   and click **Index repository**. You'll see a live progress bar as it
   clones, parses, embeds, and stores the repo.
2. Once indexing finishes, four tabs become available:
   - **Chat** -- ask questions; expand any citation to see the exact snippet.
   - **Documentation** -- generate and download a Markdown project summary.
   - **Architecture** -- generate a Mermaid diagram of module relationships.
   - **Dependencies** -- see a language breakdown and parsed dependency list.
3. Good questions to try: "What does this repo do?", "Explain the main
   entry point.", "Where is X handled?", "What are the main dependencies?"

Indexing a **private** repo: generate a token at
https://github.com/settings/tokens (repo scope), and put it in
`GITHUB_TOKEN=` in your `.env` file.

## Troubleshooting

- **"Can't reach the backend"** in Streamlit -> make sure Terminal 1 (uvicorn)
  is still running and didn't crash. Check its logs for errors.
- **`GROQ_API_KEY is not set`** -> double check `backend\.env` exists (not
  just `.env.example`) and has your real key with no quotes around it.
- **`git clone failed`** -> make sure `git` is installed and the repo URL is
  correct and public (or you've set `GITHUB_TOKEN` for private repos).
- **`pip install` fails on `chromadb`** -> this is usually a missing C++
  build tool issue on Windows. Installing "Desktop development with C++"
  from the Visual Studio Build Tools installer resolves it, or try
  upgrading pip first (`pip install --upgrade pip`).
- **"This repository is already being indexed"** -> wait for the current
  indexing job to finish before re-indexing the same URL.
- **Architecture tab shows a plain folder map, not an import graph** -> this
  is the fallback for non-Python-heavy repos, or Python repos with very few
  internal imports between files (e.g. small scripts).
- **Answers seem to ignore the repo / say "not enough information"** -> the
  repo may use file types outside the indexed list (see `ALLOWED_EXTENSIONS`
  in `backend/ingest.py`) -- add extensions there if needed.
- **Port already in use** -> change `--port 8000` to something else, and
  update `BACKEND_URL` in `frontend/app.py` to match.

## Known limitations (by design, for simplicity)

- Conversation history and the BM25 keyword index reset when the backend
  restarts (both are kept in memory, not a database).
- Chunking is language-aware for Python and common curly-brace languages,
  but line numbers for the plain-text fallback (docs/config files) are an
  approximation, not exact.
- The architecture diagram only traces imports between the repo's own
  Python files (not external libraries), and caps at the 40 most-connected
  modules so large repos stay readable.
- One backend process serves all indexed repos, but the Streamlit UI tracks
  one "active" repo per browser session.

## Next steps (optional, ask any time)

- Add a `Dockerfile` / `docker-compose.yml` to containerize both services.
- Add automated tests (`pytest`).
- Deployment guides for Render, Railway, or a VPS.
- Swap the Groq call for a local model (Ollama) if you get a GPU later --
  the LLM call is isolated in `backend/rag.py`, so this is a small change.
- Persist conversation history and the BM25 index to disk/SQLite so they
  survive a backend restart.
