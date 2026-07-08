"""
Handles everything needed to turn a GitHub repo URL into searchable,
embedded chunks stored in a local Chroma collection.
"""
import hashlib
import os
import shutil
import subprocess
import stat
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional

import chromadb
from sentence_transformers import SentenceTransformer

from chunker import chunk_file
from config import settings

# Called as progress_callback(stage, processed, total). Optional -- lets the
# API layer report indexing progress without ingest.py knowing about FastAPI.
ProgressCallback = Optional[Callable[[str, int, int], None]]

# --- Lazy singletons (loaded once, reused across requests) ---
_embedder = None
_chroma_client = None


def get_embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(settings.embedding_model)
    return _embedder


def get_chroma_client() -> chromadb.ClientAPI:
    global _chroma_client
    if _chroma_client is None:
        _chroma_client = chromadb.PersistentClient(path=settings.chroma_persist_dir)
    return _chroma_client


# Directories we never want to index.
SKIP_DIRS = {
    ".git", "node_modules", "venv", ".venv", "env", "__pycache__",
    "dist", "build", ".idea", ".vscode", "target", ".mypy_cache",
    ".pytest_cache", "site-packages", ".next", "coverage",
}

# File types worth indexing (code + docs + config).
ALLOWED_EXTENSIONS = {
    ".py", ".js", ".jsx", ".ts", ".tsx", ".java", ".go", ".rb", ".rs",
    ".c", ".cpp", ".h", ".hpp", ".cs", ".php", ".swift", ".kt", ".scala",
    ".sh", ".sql", ".html", ".css", ".md", ".txt", ".yaml", ".yml",
    ".json", ".toml", ".ini", ".cfg",
}


def repo_id_from_url(url: str) -> str:
    """Short, stable identifier for a repo URL. Used as the Chroma collection name."""
    return hashlib.sha1(url.strip().encode()).hexdigest()[:12]


def repo_name_from_url(url: str) -> str:
    """Best-effort 'owner/repo' extraction for display purposes."""
    cleaned = url.rstrip("/").removesuffix(".git")
    parts = cleaned.split("/")
    if len(parts) >= 2:
        return f"{parts[-2]}/{parts[-1]}"
    return cleaned


def _remove_path_on_windows(path: Path) -> None:
    """Remove a directory tree, retrying on Windows file lock/read-only issues."""

    def onerror(func, path_str, exc_info):
        try:
            os.chmod(path_str, stat.S_IWRITE)
        except OSError:
            pass
        try:
            func(path_str)
        except OSError:
            pass

    last_error: Optional[BaseException] = None
    for attempt in range(3):
        try:
            shutil.rmtree(path, onerror=onerror)
            return
        except FileNotFoundError:
            return
        except OSError as exc:
            last_error = exc
            if attempt < 2:
                time.sleep(0.2 * (attempt + 1))

    if last_error is not None:
        raise last_error


def clone_repo(url: str) -> Path:
    """Shallow-clone the repo into a per-repo folder under data/repos."""
    rid = repo_id_from_url(url)
    dest = Path(settings.clone_dir) / rid

    if dest.exists():
        _remove_path_on_windows(dest)

    clone_url = url
    if settings.github_token and "github.com" in url and url.startswith("https://"):
        clone_url = url.replace("https://", f"https://{settings.github_token}@", 1)

    result = subprocess.run(
        ["git", "clone", "--depth", "1", clone_url, str(dest)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"git clone failed: {result.stderr.strip()}")

    return dest


def walk_files(root: Path) -> List[Path]:
    """Return every file worth indexing under root, skipping junk directories and huge files."""
    files = []
    for path in root.rglob("*"):
        if path.is_dir():
            continue
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        if path.suffix.lower() not in ALLOWED_EXTENSIONS:
            continue
        try:
            if path.stat().st_size > settings.max_file_size_kb * 1024:
                continue
        except OSError:
            continue
        files.append(path)
    return files


def index_repository(url: str, progress_callback: ProgressCallback = None) -> Dict:
    """Clone, parse, chunk, embed, and store a repository. Safe to call again to re-index.

    progress_callback(stage, processed, total) is called at each stage so
    the API layer can report live progress (stages: cloning, parsing,
    embedding, storing)."""
    rid = repo_id_from_url(url)

    if progress_callback:
        progress_callback("cloning", 0, 0)
    repo_path = clone_repo(url)
    files = walk_files(repo_path)

    if progress_callback:
        progress_callback("parsing", 0, len(files))

    client = get_chroma_client()
    try:
        client.delete_collection(rid)
    except Exception:
        pass
    collection = client.create_collection(rid)

    documents: List[str] = []
    metadatas: List[Dict] = []
    ids: List[str] = []
    indexed_files: List[str] = []
    chunk_counter = 0

    for idx, file_path in enumerate(files):
        try:
            text = file_path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue

        rel_path = str(file_path.relative_to(repo_path)).replace("\\", "/")
        indexed_files.append(rel_path)

        pieces = chunk_file(text, file_path.suffix.lower(), settings.chunk_size, settings.chunk_overlap)
        for i, piece in enumerate(pieces):
            documents.append(piece["text"])
            metadatas.append({
                "file": rel_path,
                "chunk_index": i,
                "line_start": piece["line_start"],
                "line_end": piece["line_end"],
                "kind": piece.get("kind", "text"),
            })
            ids.append(f"{rid}_{chunk_counter}")
            chunk_counter += 1

        if progress_callback and idx % 5 == 0:
            progress_callback("parsing", idx + 1, len(files))

    embedder = get_embedder()
    if documents:
        embed_batch = 64
        all_embeddings: List[List[float]] = []
        for i in range(0, len(documents), embed_batch):
            batch = documents[i:i + embed_batch]
            all_embeddings.extend(embedder.encode(batch, show_progress_bar=False).tolist())
            if progress_callback:
                progress_callback("embedding", min(i + embed_batch, len(documents)), len(documents))

        store_batch = 500
        for i in range(0, len(documents), store_batch):
            collection.add(
                documents=documents[i:i + store_batch],
                embeddings=all_embeddings[i:i + store_batch],
                metadatas=metadatas[i:i + store_batch],
                ids=ids[i:i + store_batch],
            )
        if progress_callback:
            progress_callback("storing", len(documents), len(documents))

    # Build the keyword (BM25) index alongside the semantic one. Deferred
    # import avoids a circular import (search.py imports helpers from here).
    from search import build_bm25_index
    build_bm25_index(rid, documents, metadatas)

    return {
        "repo_id": rid,
        "repo_name": repo_name_from_url(url),
        "url": url,
        "files_indexed": len(files),
        "chunks_indexed": chunk_counter,
        "indexed_files": indexed_files,
    }
