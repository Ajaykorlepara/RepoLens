"""
FastAPI backend for the GitHub RAG Agent.

Run with:  uvicorn main:app --reload --port 8000
"""
import logging
import threading
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import docgen
import ingest
from config import settings
from rag import ask_question, get_history

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("github_rag_agent")

app = FastAPI(title="GitHub RAG Agent", version="2.0.0")

# Wide open CORS since this is a local dev tool talking to a local Streamlit app.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory registries. Simple by design -- reset if the backend restarts.
_indexed_repos: dict = {}
_indexing_progress: dict = {}


class IndexRequest(BaseModel):
    url: str


class ChatRequest(BaseModel):
    repo_id: str
    question: str


def _run_indexing(url: str, repo_id: str) -> None:
    """Runs in a background thread so /index can return immediately."""
    def on_progress(stage: str, processed: int, total: int) -> None:
        _indexing_progress[repo_id] = {
            "stage": stage, "processed": processed, "total": total,
            "done": False, "error": None,
        }

    try:
        logger.info(f"Indexing repository: {url}")
        result = ingest.index_repository(url, progress_callback=on_progress)
        _indexed_repos[repo_id] = result
        _indexing_progress[repo_id] = {
            "stage": "done", "processed": result["chunks_indexed"],
            "total": result["chunks_indexed"], "done": True, "error": None,
        }
        logger.info(f"Indexed {result['files_indexed']} files, {result['chunks_indexed']} chunks.")
    except Exception as e:
        logger.exception("Background indexing failed")
        _indexing_progress[repo_id] = {
            "stage": "error", "processed": 0, "total": 0, "done": True, "error": str(e),
        }


@app.get("/status")
def status():
    return {"status": "ok", "indexed_repos": list(_indexed_repos.values())}


@app.post("/index")
def index_repo(req: IndexRequest):
    if not req.url.strip():
        raise HTTPException(status_code=400, detail="Repository URL is required.")

    repo_id = ingest.repo_id_from_url(req.url)
    if repo_id in _indexing_progress and not _indexing_progress[repo_id].get("done", True):
        raise HTTPException(status_code=409, detail="This repository is already being indexed.")

    # Set this synchronously, before the thread starts, so the very first
    # progress poll never races against thread startup.
    _indexing_progress[repo_id] = {"stage": "queued", "processed": 0, "total": 0, "done": False, "error": None}
    thread = threading.Thread(target=_run_indexing, args=(req.url, repo_id), daemon=True)
    thread.start()
    return {"repo_id": repo_id, "repo_name": ingest.repo_name_from_url(req.url), "status": "started"}


@app.get("/index/progress/{repo_id}")
def index_progress(repo_id: str):
    if repo_id not in _indexing_progress:
        raise HTTPException(status_code=404, detail="No indexing job found for this repo id.")
    return _indexing_progress[repo_id]


@app.post("/chat")
def chat(req: ChatRequest):
    if req.repo_id not in _indexed_repos:
        raise HTTPException(status_code=404, detail="Repository not indexed. Call /index first.")
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question is required.")
    try:
        return ask_question(req.repo_id, req.question)
    except Exception as e:
        logger.exception("Failed to answer question")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/history/{repo_id}")
def history(repo_id: str):
    return {"history": get_history(repo_id)}


@app.get("/document/{repo_id}")
def document(repo_id: str):
    if repo_id not in _indexed_repos:
        raise HTTPException(status_code=404, detail="Repository not indexed. Call /index first.")
    try:
        return {"markdown": docgen.generate_documentation(repo_id)}
    except Exception as e:
        logger.exception("Failed to generate documentation")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/diagram/{repo_id}")
def diagram(repo_id: str):
    if repo_id not in _indexed_repos:
        raise HTTPException(status_code=404, detail="Repository not indexed. Call /index first.")
    try:
        repo_path = Path(settings.clone_dir) / repo_id
        return {"mermaid": docgen.generate_dependency_diagram(repo_path)}
    except Exception as e:
        logger.exception("Failed to generate diagram")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/analyze/{repo_id}")
def analyze(repo_id: str):
    if repo_id not in _indexed_repos:
        raise HTTPException(status_code=404, detail="Repository not indexed. Call /index first.")
    try:
        repo_path = Path(settings.clone_dir) / repo_id
        indexed_files = _indexed_repos[repo_id].get("indexed_files", [])
        return docgen.analyze_dependencies(repo_path, indexed_files)
    except Exception as e:
        logger.exception("Failed to analyze dependencies")
        raise HTTPException(status_code=500, detail=str(e))
