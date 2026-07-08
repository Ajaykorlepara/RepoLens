"""
Three "meta" features built on top of the same ingestion/RAG pipeline:

1. generate_documentation  -- LLM-written Markdown project summary,
   grounded in several targeted retrieval queries.
2. generate_dependency_diagram -- a Mermaid graph of which local Python
   modules import which other local modules (falls back to a simple
   folder map for non-Python repos).
3. analyze_dependencies -- language breakdown + parsed manifest files
   (requirements.txt, pyproject.toml, package.json).
"""
import ast
import json
import tomllib
from collections import Counter
from pathlib import Path
from typing import Dict, List

from config import settings
from rag import get_groq_client
from search import hybrid_search

DOC_QUERIES = [
    "What is the overall purpose and functionality of this project?",
    "What are the main modules, classes, or components and what do they do?",
    "How is the project structured -- what are the key entry points?",
    "What are the main dependencies and requirements of this project?",
]


def generate_documentation(repo_id: str) -> str:
    """Ask the LLM for a structured Markdown summary, grounded in context
    retrieved from a few different angles (overview, modules, structure, deps)."""
    context_sections = []
    for query in DOC_QUERIES:
        contexts = hybrid_search(repo_id, query, top_k=4)
        block = "\n".join(f"[{c['file']}]\n{c['text'][:800]}" for c in contexts)
        context_sections.append(f"### Regarding: {query}\n{block}")

    combined_context = "\n\n".join(context_sections)

    system_message = (
        "You are a technical writer. Using ONLY the provided source excerpts, "
        "write a clear Markdown project summary with exactly these sections: "
        "## Overview, ## Key Modules, ## Architecture, ## Dependencies. "
        "Be concise and factual. If something isn't covered by the excerpts, "
        "say it isn't clear from the available code rather than guessing."
    )

    client = get_groq_client()
    response = client.chat.completions.create(
        model=settings.groq_model,
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": combined_context},
        ],
        temperature=0.2,
    )
    return response.choices[0].message.content


def _local_module_name(rel_path: Path) -> tuple[str, bool]:
    """Returns (dotted_module_name, is_package_init). Strips a leading
    src/ or lib/ layout folder, since that's not part of the real import
    path (e.g. src/click/core.py -> "click.core", not "src.click.core")."""
    parts = list(rel_path.with_suffix("").parts)
    if parts and parts[0] in ("src", "lib"):
        parts = parts[1:]
    is_init = bool(parts) and parts[-1] == "__init__"
    if is_init:
        parts = parts[:-1]
    return ".".join(parts), is_init


def _resolve_relative_import(mod_parts: List[str], is_init: bool, level: int, module: str) -> str:
    """Resolves a relative import (`from . import x`, `from .foo import y`,
    `from ..foo import y`) to a dotted absolute-style module path, relative
    to the importing file's own module. Accounts for the fact that a
    package's __init__.py is one level "shallower" than a regular
    submodule for relative-import purposes."""
    effective_strip = level - 1 if is_init else level
    if effective_strip > 0:
        base = mod_parts[:-effective_strip] if effective_strip <= len(mod_parts) else []
    else:
        base = mod_parts
    parts = base + ([module] if module else [])
    return ".".join(p for p in parts if p)


def _folder_diagram(repo_path: Path) -> str:
    """Fallback: a simple top-level folder/file map as a Mermaid diagram."""
    lines = ["graph TD", '    root["repository root"]']
    try:
        top_level = sorted(p for p in repo_path.iterdir() if p.name != ".git")
    except OSError:
        top_level = []
    for i, p in enumerate(top_level[:20]):
        node_id = f"n{i}"
        label = (p.name + "/") if p.is_dir() else p.name
        label = label.replace('"', "'")
        lines.append(f'    {node_id}["{label}"]')
        lines.append(f"    root --> {node_id}")
    return "\n".join(lines)


def generate_dependency_diagram(repo_path: Path) -> str:
    """Mermaid graph of local-module import relationships for Python repos.
    Falls back to a folder map when there's no meaningful Python graph."""
    py_files = [
        p for p in repo_path.rglob("*.py")
        if ".git" not in p.parts and "node_modules" not in p.parts and "venv" not in p.parts
    ]
    if not py_files:
        return _folder_diagram(repo_path)

    module_names = set()
    file_to_module = {}  # file -> (module_name, is_init)
    for f in py_files:
        rel = f.relative_to(repo_path)
        mod, is_init = _local_module_name(rel)
        if mod:
            module_names.add(mod)
            file_to_module[f] = (mod, is_init)

    edges = set()
    for f, (mod, is_init) in file_to_module.items():
        try:
            tree = ast.parse(f.read_text(encoding="utf-8", errors="ignore"))
        except (SyntaxError, ValueError):
            continue

        mod_parts = mod.split(".")
        for node in ast.walk(tree):
            targets: List[str] = []
            if isinstance(node, ast.Import):
                targets = [n.name for n in node.names]
            elif isinstance(node, ast.ImportFrom):
                if node.level and node.level > 0:
                    # Relative import: from . import x / from .foo import y
                    resolved = _resolve_relative_import(mod_parts, is_init, node.level, node.module or "")
                    if resolved:
                        targets = [resolved]
                elif node.module:
                    targets = [node.module]

            for t in targets:
                for candidate in module_names:
                    if candidate != mod and (candidate == t or t.startswith(candidate + ".")):
                        edges.add((mod, candidate))

    if not edges:
        return _folder_diagram(repo_path)

    # Keep large repos legible: cap to the most-connected modules (by degree)
    # rather than truncating lines, which could show mostly nodes and cut
    # off nearly every edge.
    max_nodes = 40
    involved = {m for pair in edges for m in pair}
    if len(involved) > max_nodes:
        degree = Counter()
        for a, b in edges:
            degree[a] += 1
            degree[b] += 1
        top_modules = {m for m, _ in degree.most_common(max_nodes)}
        edges = {(a, b) for a, b in edges if a in top_modules and b in top_modules}
        involved = {m for pair in edges for m in pair}

    if not edges:
        return _folder_diagram(repo_path)

    safe_id = {name: f"m{i}" for i, name in enumerate(sorted(involved))}
    lines = ["graph TD"]
    for name, node_id in safe_id.items():
        label = name.replace('"', "'")
        lines.append(f'    {node_id}["{label}"]')
    for src, dst in sorted(edges):
        lines.append(f"    {safe_id[src]} --> {safe_id[dst]}")

    return "\n".join(lines)


LANGUAGE_BY_EXTENSION = {
    ".py": "Python", ".js": "JavaScript", ".jsx": "JavaScript", ".ts": "TypeScript",
    ".tsx": "TypeScript", ".java": "Java", ".go": "Go", ".rb": "Ruby", ".rs": "Rust",
    ".c": "C", ".cpp": "C++", ".h": "C/C++ Header", ".hpp": "C++ Header", ".cs": "C#",
    ".php": "PHP", ".swift": "Swift", ".kt": "Kotlin", ".scala": "Scala", ".sh": "Shell",
    ".sql": "SQL", ".html": "HTML", ".css": "CSS",
}


def analyze_dependencies(repo_path: Path, indexed_files: List[str]) -> Dict:
    """Language breakdown by indexed file count, plus parsed dependency
    manifests (requirements.txt, pyproject.toml, package.json)."""
    language_counts = Counter()
    for f in indexed_files:
        ext = Path(f).suffix.lower()
        language_counts[LANGUAGE_BY_EXTENSION.get(ext, ext or "other")] += 1

    dependencies: List[Dict] = []

    req_file = repo_path / "requirements.txt"
    if req_file.exists():
        for line in req_file.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                dependencies.append({"name": line, "source": "requirements.txt"})

    pyproject = repo_path / "pyproject.toml"
    if pyproject.exists():
        try:
            data = tomllib.loads(pyproject.read_text(encoding="utf-8", errors="ignore"))

            for d in data.get("project", {}).get("dependencies", []):
                dependencies.append({"name": d, "source": "pyproject.toml"})

            for extra, deps in data.get("project", {}).get("optional-dependencies", {}).items():
                for d in deps:
                    dependencies.append({"name": d, "source": f"pyproject.toml (optional: {extra})"})

            # PEP 735 dependency groups (e.g. dev, test) -- items are either
            # plain requirement strings or {"include-group": "..."} refs.
            for group, deps in data.get("dependency-groups", {}).items():
                for d in deps:
                    if isinstance(d, str):
                        dependencies.append({"name": d, "source": f"pyproject.toml (group: {group})"})

            poetry_deps = data.get("tool", {}).get("poetry", {}).get("dependencies", {})
            for name, version in poetry_deps.items():
                if name.lower() == "python":
                    continue
                dependencies.append({"name": f"{name} {version}", "source": "pyproject.toml"})
        except (tomllib.TOMLDecodeError, OSError):
            pass

    package_json = repo_path / "package.json"
    if package_json.exists():
        try:
            data = json.loads(package_json.read_text(encoding="utf-8", errors="ignore"))
            for section in ("dependencies", "devDependencies"):
                for name, version in data.get(section, {}).items():
                    dependencies.append({"name": f"{name} {version}", "source": f"package.json ({section})"})
        except (json.JSONDecodeError, OSError):
            pass

    return {
        "languages": dict(language_counts.most_common()),
        "dependencies": dependencies,
    }
