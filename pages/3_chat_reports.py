"""
pages/3_chat_reports.py

Ask natural language questions about past analysis reports.
ChromaDB retrieves relevant chunks; the chat LLM (routed through
litellm + OPENAI_API_KEY, OPENAI-specific — separate from
the Gemini routing used by agents.py / data_cleaner.py / domain_detector.py
/ dashboard_agent.py) answers from them.
"""

import streamlit as st
import os
import time
import litellm
from dotenv import load_dotenv
from auth import render_login_page, check_authentication, render_logout
from vector_store import query_reports, list_stored_reports, delete_report

load_dotenv()

# ── Retry settings (mirrors agents.py) ────────────────────────────────────────
MAX_RETRIES  = 5
INITIAL_WAIT = 10  # seconds

# ── set_page_config MUST be the very first st call ────────────────────────────
st.set_page_config(page_title="Chat with Reports", page_icon="💬", layout="wide")

# ── Auth ──────────────────────────────────────────────────────────────────────
authenticator, config = render_login_page()
is_authenticated, name, username = check_authentication(authenticator, config)
if not is_authenticated:
    st.stop()

render_logout(authenticator, name)

# ── Global styles — light blue theme ──────────────────────────────────────────
st.markdown("""
<style>
.stApp { background-color: #00060f; color: #cae6ff; }
h1 {
    background: linear-gradient(90deg, #38bdf8, #0ea5e9, #7dd3fc);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    font-size: 2.2rem !important; font-weight: 800 !important;
}
h2, h3 { color: #38bdf8 !important; }
hr { border-color: rgba(56,189,248,0.18) !important; }

[data-testid="stChatMessage"] {
    background: rgba(14,165,233,0.06) !important;
    border: 1px solid rgba(56,189,248,0.2) !important;
    border-radius: 12px !important;
    margin-bottom: 8px !important;
}
[data-testid="stChatInput"] textarea {
    background: rgba(0,15,30,0.75) !important;
    border: 1px solid rgba(56,189,248,0.35) !important;
    color: #e0f2fe !important;
    border-radius: 12px !important;
}
div[data-testid="stExpander"] {
    border: 1px solid rgba(56,189,248,0.2) !important;
    border-radius: 12px !important;
}

.question-pill {
    display: inline-block;
    background: rgba(14,165,233,0.1);
    border: 1px solid rgba(56,189,248,0.25);
    border-radius: 999px;
    padding: 6px 14px;
    margin: 4px;
    font-size: 0.82rem;
    color: #7dd3fc;
    cursor: pointer;
}
.question-pill:hover { border-color: #38bdf8; color: #38bdf8; }
</style>
""", unsafe_allow_html=True)


# ── LLM completion with retry (mirrors agents.py's backoff logic) ─────────────
def _llm_completion_with_retry(model, messages, max_tokens=1200, temperature=0.3):
    wait = INITIAL_WAIT
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return litellm.completion(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                api_key=os.getenv("OPENAI_API_KEY"),  
            )
        except Exception as e:
            err = str(e).lower()
            is_retryable = any(x in err for x in [
                "503", "unavailable", "rate limit", "rate_limit",
                "429", "overloaded", "capacity", "quota", "resource exhausted",
            ])
            if is_retryable and attempt < MAX_RETRIES:
                time.sleep(wait)
                wait *= 2
            else:
                raise


# ── Chat answer — routed through OPENAI via litellm, NOT Gemini ──────────
def ask_chat_model(question: str, context_chunks: list[dict]) -> str:
    context = "\n\n---\n\n".join(
        f"[From: {c['dataset_name']} | {c['timestamp'][:10]} | Relevance: {c['score']}]\n{c['text']}"
        for c in context_chunks
    )
    prompt = f"""You are a business intelligence assistant.
Answer the user's question using ONLY the report excerpts below.
If the answer isn't in the excerpts, say "I couldn't find that in your stored reports."
Never invent numbers or facts.

REPORT EXCERPTS:
{context}

USER QUESTION:
{question}

Answer in plain business English. Be specific and cite numbers where available.
Format your answer clearly — use bullet points or short paragraphs as appropriate."""

    
    model   = os.getenv("MODEL")
    api_key = os.getenv("OPENAI_API_KEY")

    if not model:
        return (
            "⚠️ No chat model configured. Set the `MODEL` environment variable "
            "in your `.env` file (e.g. `MODEL=gpt-4o-mini`)."
        )
    if not api_key:
        return (
            "⚠️ `OPENAI_API_KEY` is not set in your `.env` file. "
            "Add `OPENAI_API_KEY=sk-...` and restart the app."
        )

    response = _llm_completion_with_retry(
        model=model,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    col_home, _ = st.columns([1, 9])
    with col_home:
        if st.button("🏠 Home", use_container_width=True):
            st.switch_page("home_page.py")

    st.markdown("""
    <div style="padding: 1.5rem 0 0.5rem 0;">
        <h1 style="margin-bottom:0;">💬 Chat with Past Reports</h1>
        <p style="color:#7dd3fc; font-size:1.05rem; margin-top:0.25rem;">
            Ask questions about any analysis you've run — ChromaDB finds the relevant sections,
            your AI model answers from them.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    username = st.session_state.get("auth_username")
    stored   = list_stored_reports(username)

    if not stored:
        st.markdown("""
        <div style="background:rgba(14,165,233,0.06);border:1px solid rgba(56,189,248,0.2);
        border-left:4px solid #38bdf8;border-radius:10px;padding:20px 24px;margin-bottom:20px;">
            <div style="color:#38bdf8;font-weight:700;font-size:1rem;margin-bottom:6px;">
                📭 No reports stored yet
            </div>
            <div style="color:#7dd3fc;font-size:0.9rem;">
                Run an analysis from the main page — it will be automatically saved here
                so you can ask questions about it later.
            </div>
        </div>
        """, unsafe_allow_html=True)

        if st.button("⬅ Go to Main Analyzer", type="primary"):
            st.switch_page("home_page.py")
        return

    # ── Report controls (inline, no sidebar) ─────────────────────────────────
    with st.expander("⚙️ Report Settings", expanded=False):
        col_filter, col_delete = st.columns(2)
        with col_filter:
            st.markdown("**📁 Filter by dataset**")
            selected_dataset = st.selectbox(
                "Dataset",
                options=["All reports", "None"] + stored,
                label_visibility="collapsed",
            )
        with col_delete:
            st.markdown("**🗑️ Delete a report**")
            to_delete = st.selectbox(
                "Report to delete", options=["None"] + stored, key="del",
                label_visibility="collapsed",
            )
            if st.button("Delete report", type="secondary", disabled=(to_delete == "None")):
                n = delete_report(username, to_delete)
                st.success(f"Deleted {n} chunks for '{to_delete}'.")
                st.rerun()

    dataset_filter = None if selected_dataset in ("All reports", "None") else selected_dataset

    # ── Dataset context card ──────────────────────────────────────────────────
    filter_label = selected_dataset if selected_dataset != "All reports" else f"{len(stored)} report(s)"
    st.markdown(
        f"""
        <div style="background:rgba(14,165,233,0.06);border:1px solid rgba(56,189,248,0.2);
        border-radius:10px;padding:12px 18px;margin-bottom:16px;display:flex;align-items:center;gap:12px;">
            <span style="font-size:1.4rem;">📂</span>
            <div>
                <div style="color:#7dd3fc;font-size:0.75rem;text-transform:uppercase;letter-spacing:1px;">
                    Searching across</div>
                <div style="color:#e0f2fe;font-weight:600;">{filter_label}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Suggested questions ───────────────────────────────────────────────────
    if "chat_history" not in st.session_state or not st.session_state["chat_history"]:
        st.markdown("**💡 Try asking:**")
        suggestions = [
            "What was the total revenue?",
            "Which category performed best?",
            "What were the top 3 insights?",
            "What should the business focus on?",
            "Were there any anomalies or risks flagged?",
            "What was the month-on-month trend?",
        ]
        cols = st.columns(3)
        for i, suggestion in enumerate(suggestions):
            with cols[i % 3]:
                if st.button(suggestion, key=f"sugg_{i}", use_container_width=True):
                    st.session_state["_inject_question"] = suggestion
                    st.rerun()

    # ── Chat history ──────────────────────────────────────────────────────────
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

    for msg in st.session_state["chat_history"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # ── Handle injected question from suggestion buttons ──────────────────────
    injected = st.session_state.pop("_inject_question", None)

    # ── Chat input ────────────────────────────────────────────────────────────
    question = st.chat_input("Ask anything about your past analyses…") or injected

    if question:
        st.session_state["chat_history"].append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Searching reports and generating answer..."):
                chunks = query_reports(
                    query=question,
                    username=username,
                    n_results=5,
                    dataset_name=dataset_filter,
                )

                if not chunks:
                    answer = "No relevant content found in your stored reports for that question."
                else:
                    try:
                        answer = ask_chat_model(question, chunks)
                    except Exception as e:
                        answer = f"⚠️ Could not get a response from the AI model: {e}"

                st.markdown(answer)

                if chunks:
                    with st.expander("📎 Source chunks used", expanded=False):
                        for c in chunks:
                            st.caption(
                                f"📄 **{c['dataset_name']}** | {c['timestamp'][:10]} | "
                                f"Score: {c['score']}"
                            )
                            st.text(c['text'][:300] + "...")

        st.session_state["chat_history"].append({"role": "assistant", "content": answer})

    # ── Clear chat button ─────────────────────────────────────────────────────
    if st.session_state.get("chat_history"):
        st.divider()
        if st.button("🗑️ Clear Chat History", use_container_width=False):
            st.session_state["chat_history"] = []
            st.rerun()


if __name__ == "__main__":
    main()