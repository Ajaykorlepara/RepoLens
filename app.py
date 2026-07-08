"""
Streamlit frontend for the GitHub RAG Agent.

Run with:  streamlit run app.py
(Make sure the FastAPI backend is already running on http://localhost:8000)
"""
import time

import pandas as pd
import requests
import streamlit as st
import streamlit.components.v1 as components

BACKEND_URL = "http://localhost:8000"
MERMAID_CDN = "https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.min.js"

st.set_page_config(page_title="RepoLens", page_icon=":books:", layout="wide")
st.title("RepoLens")

# --- Session state defaults ---
for key, value in {
    "repo_id": None, "repo_name": None, "messages": [],
    "documentation": None, "diagram": None, "analysis": None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = value


def render_mermaid(code: str, height: int = 500) -> None:
    html = f"""
    <div class="mermaid">
    {code}
    </div>
    <script src="{MERMAID_CDN}"></script>
    <script>mermaid.initialize({{ startOnLoad: true, theme: "neutral" }});</script>
    """
    components.html(html, height=height, scrolling=True)


_LANGUAGE_BY_EXT = {
    "py": "python", "js": "javascript", "jsx": "javascript", "ts": "typescript",
    "tsx": "typescript", "java": "java", "go": "go", "rb": "ruby", "rs": "rust",
    "c": "c", "cpp": "cpp", "h": "c", "hpp": "cpp", "cs": "csharp", "php": "php",
    "swift": "swift", "kt": "kotlin", "sh": "bash", "sql": "sql", "html": "html",
    "css": "css", "json": "json", "yaml": "yaml", "yml": "yaml", "md": "markdown",
}


def language_for(filename: str) -> str:
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    return _LANGUAGE_BY_EXT.get(ext, "text")


def render_citations(citations: list) -> None:
    for c in citations:
        label = c["file"]
        if c.get("line_start") and c.get("line_end"):
            label += f" (lines {c['line_start']}-{c['line_end']})"
        with st.expander(f"📄 {label}"):
            st.code(c["text"], language=language_for(c["file"]))


STAGE_LABELS = {
    "queued": "Queued...",
    "cloning": "Cloning repository...",
    "parsing": "Parsing and chunking files...",
    "embedding": "Embedding chunks locally...",
    "storing": "Saving to vector store...",
    "done": "Done!",
}

# --- Sidebar: index a repository ---
with st.sidebar:
    st.header("Index a repository")
    repo_url = st.text_input("GitHub repository URL", placeholder="https://github.com/owner/repo")

    if st.button("Index repository", type="primary"):
        if repo_url.strip():
            try:
                resp = requests.post(f"{BACKEND_URL}/index", json={"url": repo_url}, timeout=30)
                resp.raise_for_status()
                data = resp.json()
                repo_id = data["repo_id"]

                progress_bar = st.progress(0.0, text="Starting...")
                prog = {}
                while True:
                    prog_resp = requests.get(f"{BACKEND_URL}/index/progress/{repo_id}", timeout=10)
                    prog_resp.raise_for_status()
                    prog = prog_resp.json()

                    if prog.get("error"):
                        st.error(f"Indexing failed: {prog['error']}")
                        break

                    stage = prog.get("stage", "queued")
                    total = prog.get("total", 0) or 1
                    processed = prog.get("processed", 0)
                    fraction = min(processed / total, 1.0) if total else 0.0
                    progress_bar.progress(fraction, text=STAGE_LABELS.get(stage, stage))

                    if prog.get("done"):
                        break
                    time.sleep(1)

                if not prog.get("error"):
                    st.session_state.repo_id = repo_id
                    st.session_state.repo_name = data.get("repo_name", repo_id)
                    st.session_state.messages = []
                    st.session_state.documentation = None
                    st.session_state.diagram = None
                    st.session_state.analysis = None
                    progress_bar.progress(1.0, text="Indexed!")
                    st.success("Repository indexed. Explore the tabs below.")

            except requests.exceptions.ConnectionError:
                st.error("Can't reach the backend. Is it running at http://localhost:8000?")
            except requests.exceptions.HTTPError as e:
                if e.response is not None and e.response.status_code == 409:
                    st.warning("This repository is already being indexed -- please wait for it to finish.")
                else:
                    st.error(f"Indexing failed: {e}")
            except Exception as e:
                st.error(f"Indexing failed: {e}")
        else:
            st.warning("Please enter a GitHub URL first.")

    if st.session_state.repo_id:
        st.divider()
        st.caption(f"Active repository: **{st.session_state.repo_name}**")

st.divider()

if not st.session_state.repo_id:
    st.info("Index a repository from the sidebar to get started.")
else:
    tab_chat, tab_docs, tab_diagram, tab_deps = st.tabs(
        ["Chat", "Documentation", "Architecture", "Dependencies"]
    )

    # --- Chat tab ---
    with tab_chat:
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                render_citations(msg.get("citations", []))

        question = st.chat_input("Ask a question about the repository")
        if question:
            st.session_state.messages.append({"role": "user", "content": question, "citations": []})
            with st.chat_message("user"):
                st.markdown(question)

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        resp = requests.post(
                            f"{BACKEND_URL}/chat",
                            json={"repo_id": st.session_state.repo_id, "question": question},
                            timeout=120,
                        )
                        resp.raise_for_status()
                        data = resp.json()
                        st.markdown(data["answer"])
                        citations = data.get("citations", [])
                        render_citations(citations)
                        st.session_state.messages.append({
                            "role": "assistant", "content": data["answer"], "citations": citations,
                        })
                    except requests.exceptions.ConnectionError:
                        st.error("Can't reach the backend. Is it running at http://localhost:8000?")
                    except Exception as e:
                        st.error(f"Error: {e}")

    # --- Documentation tab ---
    with tab_docs:
        st.caption("Generates a Markdown project summary grounded in the indexed repo.")
        if st.button("Generate documentation"):
            with st.spinner("Writing documentation..."):
                try:
                    resp = requests.get(f"{BACKEND_URL}/document/{st.session_state.repo_id}", timeout=120)
                    resp.raise_for_status()
                    st.session_state.documentation = resp.json()["markdown"]
                except Exception as e:
                    st.error(f"Failed to generate documentation: {e}")

        if st.session_state.documentation:
            st.markdown(st.session_state.documentation)
            st.download_button(
                "Download as README.md",
                data=st.session_state.documentation,
                file_name="README_generated.md",
                mime="text/markdown",
            )

    # --- Architecture diagram tab ---
    with tab_diagram:
        st.caption("Maps local Python module imports (falls back to a folder overview for other languages).")
        if st.button("Generate architecture diagram"):
            with st.spinner("Analyzing imports..."):
                try:
                    resp = requests.get(f"{BACKEND_URL}/diagram/{st.session_state.repo_id}", timeout=60)
                    resp.raise_for_status()
                    st.session_state.diagram = resp.json()["mermaid"]
                except Exception as e:
                    st.error(f"Failed to generate diagram: {e}")

        if st.session_state.diagram:
            render_mermaid(st.session_state.diagram)
            with st.expander("View raw Mermaid code"):
                st.code(st.session_state.diagram, language="text")

    # --- Dependencies tab ---
    with tab_deps:
        st.caption("Language breakdown and parsed dependency manifests (requirements.txt, pyproject.toml, package.json).")
        if st.button("Analyze dependencies"):
            with st.spinner("Analyzing..."):
                try:
                    resp = requests.get(f"{BACKEND_URL}/analyze/{st.session_state.repo_id}", timeout=30)
                    resp.raise_for_status()
                    st.session_state.analysis = resp.json()
                except Exception as e:
                    st.error(f"Failed to analyze dependencies: {e}")

        if st.session_state.analysis:
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Languages")
                languages = st.session_state.analysis.get("languages", {})
                if languages:
                    st.bar_chart(pd.DataFrame({"files": languages}))
                else:
                    st.caption("No recognized languages found among indexed files.")
            with col2:
                st.subheader("Dependencies")
                deps = st.session_state.analysis.get("dependencies", [])
                if deps:
                    st.dataframe(pd.DataFrame(deps), use_container_width=True, hide_index=True)
                else:
                    st.caption("No requirements.txt, pyproject.toml, or package.json found.")
