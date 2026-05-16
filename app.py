"""
RAG Sales Assistant — Streamlit Frontend
Modern, production-ready UI for the local RAG pipeline.
"""

import os
import sys
import time
import logging
from pathlib import Path
from typing import Optional

import streamlit as st

# ── Path setup ──────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from document_processor import DocumentProcessor
from vector_store import VectorStoreManager
from rag_chain import RAGChain, SUPPORTED_MODELS

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
)
logger = logging.getLogger(__name__)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Sales Knowledge Assistant",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');

/* ── Global Reset ── */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* ── Background ── */
.stApp {
    background: #0a0d14;
    color: #e8eaf0;
}

.main .block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1100px;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: #0f1420;
    border-right: 1px solid #1e2435;
}

section[data-testid="stSidebar"] * {
    color: #c8cdd8 !important;
}

/* ── Header ── */
.rag-header {
    display: flex;
    align-items: center;
    gap: 14px;
    margin-bottom: 0.5rem;
}

.rag-logo {
    font-family: 'Space Mono', monospace;
    font-size: 2.6rem;
    font-weight: 700;
    background: linear-gradient(135deg, #00d4ff 0%, #7c3aed 50%, #f59e0b 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    letter-spacing: -1px;
    line-height: 1;
}

.rag-tagline {
    font-size: 0.85rem;
    color: #6b7280;
    font-weight: 400;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    margin-top: 0.25rem;
}

.divider {
    border: none;
    border-top: 1px solid #1e2435;
    margin: 1.2rem 0;
}

/* ── Status badges ── */
.status-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 4px 10px;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 600;
    font-family: 'Space Mono', monospace;
    letter-spacing: 0.03em;
}

.status-ok {
    background: rgba(16, 185, 129, 0.12);
    color: #10b981;
    border: 1px solid rgba(16, 185, 129, 0.25);
}

.status-warn {
    background: rgba(245, 158, 11, 0.12);
    color: #f59e0b;
    border: 1px solid rgba(245, 158, 11, 0.25);
}

.status-err {
    background: rgba(239, 68, 68, 0.12);
    color: #ef4444;
    border: 1px solid rgba(239, 68, 68, 0.25);
}

/* ── Chat messages ── */
.chat-wrap {
    display: flex;
    flex-direction: column;
    gap: 1.2rem;
    margin-bottom: 1.5rem;
}

.msg-user {
    display: flex;
    justify-content: flex-end;
}

.msg-assistant {
    display: flex;
    justify-content: flex-start;
}

.bubble {
    max-width: 82%;
    padding: 14px 18px;
    border-radius: 16px;
    line-height: 1.7;
    font-size: 0.92rem;
}

.bubble-user {
    background: linear-gradient(135deg, #1e3a5f, #2a1f6e);
    border: 1px solid rgba(124, 58, 237, 0.3);
    color: #e2e8f0;
    border-bottom-right-radius: 4px;
}

.bubble-assistant {
    background: #12182a;
    border: 1px solid #1e2d45;
    color: #d1d8e8;
    border-bottom-left-radius: 4px;
}

.bubble-role {
    font-family: 'Space Mono', monospace;
    font-size: 0.68rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-bottom: 6px;
    opacity: 0.55;
}

/* ── Source chips ── */
.sources-wrap {
    margin-top: 10px;
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
}

.source-chip {
    background: rgba(0, 212, 255, 0.08);
    border: 1px solid rgba(0, 212, 255, 0.2);
    color: #00d4ff;
    padding: 3px 10px;
    border-radius: 12px;
    font-size: 0.72rem;
    font-family: 'Space Mono', monospace;
}

/* ── Input area ── */
.stTextInput > div > div > input,
.stTextArea > div > div > textarea {
    background: #0f1420 !important;
    border: 1px solid #1e2435 !important;
    border-radius: 10px !important;
    color: #e8eaf0 !important;
    font-family: 'DM Sans', sans-serif !important;
}

.stTextInput > div > div > input:focus,
.stTextArea > div > div > textarea:focus {
    border-color: #7c3aed !important;
    box-shadow: 0 0 0 2px rgba(124, 58, 237, 0.15) !important;
}

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #7c3aed, #4f46e5) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 600 !important;
    transition: opacity 0.2s !important;
}

.stButton > button:hover {
    opacity: 0.88 !important;
}

/* ── File uploader ── */
[data-testid="stFileUploader"] {
    background: #0f1420;
    border: 1.5px dashed #2a3348;
    border-radius: 12px;
    padding: 1rem;
    transition: border-color 0.2s;
}

[data-testid="stFileUploader"]:hover {
    border-color: #7c3aed;
}

/* ── Metrics ── */
[data-testid="metric-container"] {
    background: #0f1420;
    border: 1px solid #1e2435;
    border-radius: 10px;
    padding: 12px 16px;
}

/* ── Expander ── */
.streamlit-expanderHeader {
    background: #0f1420 !important;
    border: 1px solid #1e2435 !important;
    border-radius: 8px !important;
    color: #8892a4 !important;
    font-size: 0.8rem !important;
}

/* ── Selectbox ── */
.stSelectbox > div > div {
    background: #0f1420 !important;
    border: 1px solid #1e2435 !important;
    color: #e8eaf0 !important;
    border-radius: 8px !important;
}

/* ── Slider ── */
.stSlider [data-baseweb="slider"] {
    margin-top: 0.5rem;
}

/* ── Sample Q chips ── */
.sample-q-wrap {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    margin: 0.6rem 0 1.2rem 0;
}

.sample-q {
    background: #0f1420;
    border: 1px solid #1e2435;
    color: #8892a4;
    padding: 5px 12px;
    border-radius: 20px;
    font-size: 0.8rem;
    cursor: pointer;
    transition: all 0.2s;
}

.sample-q:hover {
    border-color: #7c3aed;
    color: #c4b5fd;
}

/* ── Welcome card ── */
.welcome-card {
    background: #0f1420;
    border: 1px solid #1e2435;
    border-radius: 16px;
    padding: 2.5rem 2rem;
    text-align: center;
    margin: 2rem auto;
    max-width: 580px;
}

.welcome-icon {
    font-size: 3rem;
    margin-bottom: 1rem;
}

.welcome-title {
    font-size: 1.4rem;
    font-weight: 600;
    color: #e8eaf0;
    margin-bottom: 0.5rem;
}

.welcome-sub {
    color: #6b7280;
    font-size: 0.9rem;
    line-height: 1.6;
}

/* ── Step indicators ── */
.step-list {
    text-align: left;
    margin-top: 1.5rem;
    list-style: none;
    padding: 0;
}

.step-list li {
    display: flex;
    align-items: flex-start;
    gap: 10px;
    padding: 8px 0;
    border-bottom: 1px solid #1a2030;
    font-size: 0.88rem;
    color: #9ca3af;
}

.step-list li:last-child { border-bottom: none; }

.step-num {
    background: linear-gradient(135deg, #7c3aed, #4f46e5);
    color: white;
    width: 22px;
    height: 22px;
    border-radius: 50%;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    font-size: 0.7rem;
    font-weight: 700;
    flex-shrink: 0;
    font-family: 'Space Mono', monospace;
}
</style>
""",
    unsafe_allow_html=True,
)


# ── Session state ──────────────────────────────────────────────────────────────
def init_state():
    defaults = {
        "chat_history": [],
        "vs_manager": None,
        "rag_chain": None,
        "selected_model": "llama3-8b",
        "groq_api_key": "",
        "top_k": 5,
        "temperature": 0.1,
        "show_sources": True,
        "show_context": False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


init_state()


# ── Helper: get or build VectorStoreManager ───────────────────────────────────
@st.cache_resource
def get_vector_store() -> VectorStoreManager:
    vs = VectorStoreManager(
        vectorstore_dir=str(ROOT / "vectorstore"),
        index_name="sales_assistant",
    )
    vs.load_existing()
    return vs


@st.cache_resource
def get_doc_processor() -> DocumentProcessor:
    return DocumentProcessor(chunk_size=800, chunk_overlap=150)


def get_rag_chain(model_key: str, temperature: float) -> RAGChain:
    key = f"rag_{model_key}_{temperature}"
    if key not in st.session_state:
        st.session_state[key] = RAGChain(
            model_key=model_key,
            temperature=temperature,
            groq_api_key=st.session_state.get("groq_api_key", ""),
        )
    return st.session_state[key]


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    # Header
    st.markdown(
        """
        <div style="padding:1.2rem 0 0.5rem 0">
          <div style="font-family:'Space Mono',monospace;font-size:1.1rem;font-weight:700;
                      background:linear-gradient(135deg,#00d4ff,#7c3aed);
                      -webkit-background-clip:text;-webkit-text-fill-color:transparent;
                      background-clip:text;">⚡ RAG SALES AI</div>
          <div style="font-size:0.7rem;color:#4b5563;margin-top:2px;letter-spacing:.08em">
            OPEN SOURCE · ZERO COST
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.divider()

    # ── Model selection ──
    st.markdown("**🤖 Language Model**")

    groq_keys = list(RAGChain.groq_models().keys())
    ollama_keys = list(RAGChain.ollama_models().keys())
    all_keys = groq_keys + ollama_keys

    model_options = {k: v["display"] for k, v in SUPPORTED_MODELS.items()}
    default_idx = all_keys.index(st.session_state.selected_model) if st.session_state.selected_model in all_keys else 0

    selected_model = st.selectbox(
        "Choose LLM",
        options=all_keys,
        format_func=lambda k: model_options[k],
        index=default_idx,
        label_visibility="collapsed",
    )
    st.session_state.selected_model = selected_model

    model_info = SUPPORTED_MODELS[selected_model]
    st.caption(f"_{model_info['description']}_")

    backend = model_info["backend"]

    # ── Groq API key input (only for Groq models) ──
    if backend == "groq":
        st.markdown("**🔑 Groq API Key**")

        # Try to read from secrets first
        try:
            import streamlit as _st
            secret_key = _st.secrets.get("GROQ_API_KEY", "")
        except Exception:
            secret_key = ""

        if secret_key:
            st.session_state.groq_api_key = secret_key
            st.markdown('<span class="status-badge status-ok">● KEY FROM SECRETS</span>', unsafe_allow_html=True)
        else:
            entered_key = st.text_input(
                "Groq API Key",
                type="password",
                value=st.session_state.get("groq_api_key", ""),
                placeholder="gsk_...",
                label_visibility="collapsed",
            )
            if entered_key:
                st.session_state.groq_api_key = entered_key
                # Invalidate cached chain so it rebuilds with new key
                cache_key = f"rag_{selected_model}_{st.session_state.temperature}"
                st.session_state.pop(cache_key, None)

            if not st.session_state.get("groq_api_key"):
                st.markdown('<span class="status-badge status-warn">● KEY REQUIRED</span>', unsafe_allow_html=True)
                st.caption("Free key → [console.groq.com](https://console.groq.com)")
            else:
                st.markdown('<span class="status-badge status-ok">● KEY SET</span>', unsafe_allow_html=True)

    # ── Ollama status (only for Ollama models) ──
    if backend == "ollama":
        chain = get_rag_chain(selected_model, st.session_state.temperature)
        ok, msg = chain.check_backend_available()
        if ok:
            st.markdown('<span class="status-badge status-ok">● OLLAMA READY</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="status-badge status-err">● OLLAMA OFFLINE</span>', unsafe_allow_html=True)
            with st.expander("Setup instructions"):
                st.code(
                    "# 1. Install Ollama\nbrew install ollama  # macOS\n"
                    "# or visit https://ollama.com/download\n\n"
                    "# 2. Start server\nollama serve\n\n"
                    f"# 3. Pull model\nollama pull {model_info.get('ollama_name', '')}",
                    language="bash",
                )
            st.warning("⚠️ Ollama models only work locally, not on Streamlit Cloud. Use a Groq model for cloud.")

    st.divider()

    # ── Retrieval settings ──
    st.markdown("**🔍 Retrieval Settings**")
    st.session_state.top_k = st.slider(
        "Top-K chunks", min_value=1, max_value=10, value=st.session_state.top_k
    )
    st.session_state.temperature = st.slider(
        "Temperature", min_value=0.0, max_value=1.0,
        value=st.session_state.temperature, step=0.05
    )
    st.session_state.show_sources = st.toggle(
        "Show source references", value=st.session_state.show_sources
    )
    st.session_state.show_context = st.toggle(
        "Show raw context", value=st.session_state.show_context
    )

    st.divider()

    # ── Knowledge Base stats ──
    vs = get_vector_store()
    stats = vs.get_stats()
    st.markdown("**📚 Knowledge Base**")

    col1, col2 = st.columns(2)
    col1.metric("Files", stats["num_files"])
    col2.metric("Vectors", stats.get("num_vectors", 0))

    if stats["ingested_files"]:
        st.markdown("**Indexed files:**")
        for f in stats["ingested_files"]:
            st.markdown(f"• `{f}`")

    if st.button("🗑 Reset Knowledge Base", use_container_width=True):
        vs.reset()
        st.session_state.chat_history = []
        st.cache_resource.clear()
        st.success("Knowledge base cleared!")
        st.rerun()

    st.divider()

    # ── Clear chat ──
    if st.button("🔄 Clear Chat", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()


# ── Main area ────────────────────────────────────────────────────────────────
# Header
st.markdown(
    """
    <div class="rag-header">
      <div>
        <div class="rag-logo">⚡ Sales AI</div>
        <div class="rag-tagline">Local RAG · Open Source · Zero API Cost</div>
      </div>
    </div>
    <hr class="divider">
    """,
    unsafe_allow_html=True,
)

# Tabs
tab_chat, tab_upload, tab_setup = st.tabs(["💬 Chat", "📁 Documents", "⚙️ Setup Guide"])


# ────────────────────────────────────────────────────────────────────────────
# TAB 1 – CHAT
# ────────────────────────────────────────────────────────────────────────────
with tab_chat:
    vs = get_vector_store()
    stats = vs.get_stats()

    if not stats["is_ready"]:
        # Welcome / empty state
        st.markdown(
            """
            <div class="welcome-card">
              <div class="welcome-icon">🗂</div>
              <div class="welcome-title">No Knowledge Base Yet</div>
              <div class="welcome-sub">
                Upload your sales documents to get started.<br>
                The assistant will answer questions from your files.
              </div>
              <ul class="step-list">
                <li><span class="step-num">1</span>Go to the <strong>Documents</strong> tab and upload PDFs or Excel/CSV files</li>
                <li><span class="step-num">2</span>Wait for indexing to complete</li>
                <li><span class="step-num">3</span>Return here and ask your question</li>
              </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        # Sample questions
        SAMPLE_QUESTIONS = [
            "What are the top-selling products?",
            "Summarise the pricing tiers.",
            "What is our Q3 revenue target?",
            "List key sales territories.",
            "Which deals are in the pipeline?",
        ]

        st.markdown("**Quick questions:**")
        sample_cols = st.columns(len(SAMPLE_QUESTIONS))
        for i, q in enumerate(SAMPLE_QUESTIONS):
            if sample_cols[i].button(q, key=f"sq_{i}", use_container_width=True):
                st.session_state["_pending_q"] = q

        st.markdown('<hr class="divider">', unsafe_allow_html=True)

        # ── Render chat history ──
        if st.session_state.chat_history:
            st.markdown('<div class="chat-wrap">', unsafe_allow_html=True)
            for turn in st.session_state.chat_history:
                role = turn["role"]
                content = turn["content"]

                if role == "user":
                    st.markdown(
                        f"""
                        <div class="msg-user">
                          <div class="bubble bubble-user">
                            <div class="bubble-role">You</div>
                            {content}
                          </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                else:
                    sources_html = ""
                    if st.session_state.show_sources and turn.get("sources"):
                        chips = "".join(
                            f'<span class="source-chip">📄 {s.get("source_file","?")}'
                            + (f' p.{s["page"]}' if "page" in s else "")
                            + f' [{s.get("relevance_score","")}]</span>'
                            for s in turn["sources"]
                        )
                        sources_html = f'<div class="sources-wrap">{chips}</div>'

                    st.markdown(
                        f"""
                        <div class="msg-assistant">
                          <div class="bubble bubble-assistant">
                            <div class="bubble-role">Sales AI</div>
                            {content}
                            {sources_html}
                          </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    if st.session_state.show_context and turn.get("context"):
                        with st.expander("View raw context passed to LLM"):
                            st.text(turn["context"])

            st.markdown("</div>", unsafe_allow_html=True)

        # ── Input ──
        pending = st.session_state.pop("_pending_q", None)
        user_input = st.chat_input(
            "Ask anything about your sales documents…",
        )

        question = pending or user_input

        if question:
            # Append user message
            st.session_state.chat_history.append({"role": "user", "content": question})

            with st.spinner("Searching knowledge base…"):
                try:
                    results = vs.similarity_search(
                        question, k=st.session_state.top_k
                    )
                except RuntimeError as e:
                    st.error(str(e))
                    results = []

            if results is not None:
                rag = get_rag_chain(
                    st.session_state.selected_model,
                    st.session_state.temperature,
                )

                with st.spinner(f"Generating answer with {SUPPORTED_MODELS[st.session_state.selected_model]['display']}…"):
                    try:
                        response = rag.query(question, results)
                        st.session_state.chat_history.append(
                            {
                                "role": "assistant",
                                "content": response["answer"],
                                "sources": response["sources"],
                                "context": response["context_used"],
                            }
                        )
                    except Exception as e:
                        err_msg = (
                            f"LLM error: {e}\n\n"
                            f"Model: {SUPPORTED_MODELS[st.session_state.selected_model]['display']}\n"
                            f"Backend: {SUPPORTED_MODELS[st.session_state.selected_model]['backend']}"
                        )
                        st.session_state.chat_history.append(
                            {"role": "assistant", "content": f"⚠️ {err_msg}"}
                        )

            st.rerun()


# ────────────────────────────────────────────────────────────────────────────
# TAB 2 – DOCUMENT UPLOAD
# ────────────────────────────────────────────────────────────────────────────
with tab_upload:
    st.markdown("### Upload Sales Documents")
    st.markdown(
        "Supported formats: **PDF**, **Excel** (`.xlsx`, `.xls`), **CSV**  \n"
        "Documents are chunked, embedded locally, and stored in a FAISS index."
    )

    uploaded_files = st.file_uploader(
        "Drop files here",
        type=["pdf", "xlsx", "xls", "csv"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    if uploaded_files:
        if st.button("⚡ Index Documents", use_container_width=True):
            vs = get_vector_store()
            processor = get_doc_processor()

            progress_bar = st.progress(0)
            status_text = st.empty()

            for i, f in enumerate(uploaded_files):
                status_text.markdown(f"Processing **{f.name}**…")

                if f.name in vs.ingested_files:
                    st.warning(f"⚠️ `{f.name}` already indexed – skipped.")
                    progress_bar.progress((i + 1) / len(uploaded_files))
                    continue

                try:
                    docs = processor.load_from_bytes(f.getvalue(), f.name)
                    n = vs.add_documents(docs, f.name)

                    if n > 0:
                        st.success(f"✅ `{f.name}` — {n} chunks indexed")
                    else:
                        st.info(f"ℹ️ `{f.name}` — already in index")

                except Exception as e:
                    st.error(f"❌ Failed to process `{f.name}`: {e}")

                progress_bar.progress((i + 1) / len(uploaded_files))

            status_text.markdown("**Done!** Switch to the Chat tab to ask questions.")
            time.sleep(0.5)
            st.cache_resource.clear()

    # Current knowledge base summary
    st.divider()
    st.markdown("### 📊 Knowledge Base Status")
    vs = get_vector_store()
    stats = vs.get_stats()

    c1, c2, c3 = st.columns(3)
    c1.metric("Status", "Ready ✅" if stats["is_ready"] else "Empty")
    c2.metric("Documents", stats["num_files"])
    c3.metric("Total Vectors", stats.get("num_vectors", 0))

    if stats["ingested_files"]:
        st.markdown("**Indexed files:**")
        for fname in sorted(stats["ingested_files"]):
            st.markdown(f"- 📄 `{fname}`")
    else:
        st.info("No documents indexed yet. Upload files above.")


# ────────────────────────────────────────────────────────────────────────────
# TAB 3 – SETUP GUIDE
# ────────────────────────────────────────────────────────────────────────────
with tab_setup:
    st.markdown("### ⚙️ Setup Guide — Zero Cost, Fully Local")

    st.markdown(
        """
        This application runs **100% locally** — no API keys, no cloud services,
        no login required. Follow these steps to get started.
        """
    )

    with st.expander("**Step 1 — Install Python dependencies**", expanded=True):
        st.code(
            "# Create virtual environment (recommended)\npython -m venv .venv\nsource .venv/bin/activate  # Windows: .venv\\Scripts\\activate\n\n# Install packages\npip install -r requirements.txt",
            language="bash",
        )

    with st.expander("**Step 2 — Install Ollama**"):
        st.markdown(
            """
            Ollama runs LLMs locally on your machine.

            - **macOS**: `brew install ollama`
            - **Linux**: `curl -fsSL https://ollama.com/install.sh | sh`
            - **Windows**: Download from [ollama.com/download](https://ollama.com/download)
            """
        )
        st.code("# Verify installation\nollama --version", language="bash")

    with st.expander("**Step 3 — Pull a language model**"):
        st.markdown("Choose one of these free models:")
        for key, info in SUPPORTED_MODELS.items():
            st.code(f"ollama pull {info['ollama_name']}  # {info['display']} ({info['pull_size']})", language="bash")

    with st.expander("**Step 4 — Start Ollama server**"):
        st.code("ollama serve", language="bash")
        st.caption("Keep this running in a separate terminal window.")

    with st.expander("**Step 5 — Run the application**"):
        st.code("streamlit run app.py", language="bash")

    st.divider()
    st.markdown("### 🏗 Architecture")
    st.markdown(
        """
        | Component | Technology | Cost |
        |-----------|-----------|------|
        | LLM | Ollama (Mistral / Llama 3 / Phi-3) | Free |
        | Embeddings | `all-MiniLM-L6-v2` (HuggingFace) | Free |
        | Vector DB | FAISS (Facebook AI) | Free |
        | Framework | LangChain | Free |
        | Frontend | Streamlit | Free |
        | PDF parsing | pdfplumber + pypdf | Free |
        | Excel/CSV | pandas + openpyxl | Free |
        """
    )

    st.markdown("### 📁 Project Structure")
    st.code(
        """
rag-sales-assistant/
├── app.py                  # Streamlit frontend
├── requirements.txt        # Python dependencies
├── src/
│   ├── document_processor.py   # PDF, Excel, CSV ingestion
│   ├── vector_store.py          # FAISS + embeddings
│   └── rag_chain.py             # LLM + RAG pipeline
├── vectorstore/            # Persisted FAISS index (auto-created)
└── uploads/                # Temp upload staging
        """,
        language="text",
    )
