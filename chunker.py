"""
Language-aware chunking.

Instead of one generic sliding window over raw text, this tries to split
code at meaningful boundaries (functions/classes) so each chunk is a
coherent unit -- and tracks line numbers so citations can point at exact
lines, not just a filename.

Strategy:
- .py files: parsed with the built-in `ast` module, split at top-level
  function/class boundaries.
- Common curly-brace / keyword languages (JS, TS, Java, Go, etc.): a
  lightweight regex heuristic looks for lines that start a function/class.
- Anything else (or anything that fails to parse): falls back to a plain
  sliding window, same as before, with an approximate line count.

No extra dependencies (no tree-sitter) -- keeps install simple.
"""
import ast
import re
from typing import Dict, List, Optional

GENERIC_LANGUAGE_EXTENSIONS = {
    ".js", ".jsx", ".ts", ".tsx", ".java", ".go", ".rs", ".c", ".cpp",
    ".h", ".hpp", ".cs", ".php", ".swift", ".kt", ".scala",
}

_BOUNDARY_PATTERN = re.compile(
    r"^\s*(export\s+)?(default\s+)?(async\s+)?"
    r"(function\b|class\b|interface\b|public\s|private\s|protected\s|static\s|func\b|impl\b)"
)


def _slide(text: str, size: int, overlap: int) -> List[str]:
    """Plain sliding window over a block of text."""
    text = text.strip()
    if not text:
        return []
    if len(text) <= size:
        return [text]
    chunks = []
    start = 0
    step = max(size - overlap, 1)
    while start < len(text):
        chunks.append(text[start:start + size])
        start += step
    return chunks


def _split_oversized(text: str, start_line: int, size: int, overlap: int, kind: str) -> List[Dict]:
    """A single function/class was bigger than our chunk budget -- break it
    up further, while still giving each piece an approximate line range."""
    pieces = []
    offset = start_line
    for sub in _slide(text, size, overlap):
        n_lines = sub.count("\n") + 1
        pieces.append({"text": sub, "line_start": offset, "line_end": offset + n_lines - 1, "kind": kind})
        offset += max(n_lines - 1, 1)
    return pieces


def chunk_python(source: str, chunk_size: int, chunk_overlap: int) -> Optional[List[Dict]]:
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError):
        return None

    lines = source.splitlines(keepends=True)
    n_lines = len(lines)
    covered = [False] * (n_lines + 1)  # 1-indexed
    chunks: List[Dict] = []

    for node in ast.iter_child_nodes(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue

        start = node.lineno
        if getattr(node, "decorator_list", None):
            start = min(d.lineno for d in node.decorator_list)
        end = getattr(node, "end_lineno", node.lineno)
        end = min(end, n_lines)

        for ln in range(start, end + 1):
            if 0 < ln <= n_lines:
                covered[ln] = True

        text = "".join(lines[start - 1:end]).strip()
        if not text:
            continue

        kind = "class" if isinstance(node, ast.ClassDef) else "function"
        if len(text) > chunk_size * 1.5:
            chunks.extend(_split_oversized(text, start, chunk_size, chunk_overlap, kind))
        else:
            chunks.append({"text": text, "line_start": start, "line_end": end, "kind": kind})

    # Leftover: module docstring, imports, top-level constants, __main__ guard, etc.
    leftover_line_nums = [i for i in range(1, n_lines + 1) if not covered[i]]
    if leftover_line_nums:
        leftover_text = "".join(lines[i - 1] for i in leftover_line_nums).strip()
        if leftover_text:
            lo, hi = leftover_line_nums[0], leftover_line_nums[-1]
            for sub in _slide(leftover_text, chunk_size, chunk_overlap):
                chunks.append({"text": sub, "line_start": lo, "line_end": hi, "kind": "module"})

    return chunks if chunks else None


def chunk_generic(source: str, chunk_size: int, chunk_overlap: int) -> Optional[List[Dict]]:
    lines = source.splitlines(keepends=True)
    boundaries = [i for i, line in enumerate(lines) if _BOUNDARY_PATTERN.match(line)]
    if not boundaries:
        return None

    chunks: List[Dict] = []

    # Leading content before the first recognized boundary (imports, etc.)
    if boundaries[0] > 0:
        header = "".join(lines[:boundaries[0]]).strip()
        if header:
            chunks.append({"text": header, "line_start": 1, "line_end": boundaries[0], "kind": "module"})

    boundaries.append(len(lines))
    for idx in range(len(boundaries) - 1):
        start_idx, end_idx = boundaries[idx], boundaries[idx + 1]
        text = "".join(lines[start_idx:end_idx]).strip()
        if not text:
            continue
        if len(text) > chunk_size * 1.5:
            chunks.extend(_split_oversized(text, start_idx + 1, chunk_size, chunk_overlap, "block"))
        else:
            chunks.append({"text": text, "line_start": start_idx + 1, "line_end": end_idx, "kind": "block"})

    return chunks if chunks else None


def chunk_file(text: str, extension: str, chunk_size: int, chunk_overlap: int) -> List[Dict]:
    """Entry point used by the ingestion pipeline. Always returns a list of
    {text, line_start, line_end, kind} dicts, even for plain-text files."""
    extension = extension.lower()

    if extension == ".py":
        result = chunk_python(text, chunk_size, chunk_overlap)
        if result:
            return result

    if extension in GENERIC_LANGUAGE_EXTENSIONS:
        result = chunk_generic(text, chunk_size, chunk_overlap)
        if result:
            return result

    # Fallback: plain sliding window. Line numbers here are an approximation
    # (based on counting newlines), not exact -- fine for docs/config files
    # where precise citations matter less than for source code.
    pieces = []
    running_line = 1
    for chunk in _slide(text, chunk_size, chunk_overlap):
        n_lines = chunk.count("\n") + 1
        pieces.append({
            "text": chunk,
            "line_start": running_line,
            "line_end": running_line + n_lines - 1,
            "kind": "text",
        })
        running_line += max(n_lines - 1, 1)
    return pieces
