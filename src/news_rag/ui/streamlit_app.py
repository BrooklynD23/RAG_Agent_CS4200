import os
from typing import Optional

import httpx
import streamlit as st

try:
    from src.news_rag.config import settings
except ModuleNotFoundError:  # pragma: no cover - import fallback
    try:
        from news_rag.config import settings
    except ModuleNotFoundError:  # pragma: no cover - import fallback
        settings = None  # type: ignore[assignment]

try:
    from src.news_rag.ui.components import render_summary, render_sources
except ModuleNotFoundError:
    from components import render_summary, render_sources


API_BASE_URL = os.getenv("NEWS_RAG_API_BASE_URL", "http://localhost:8000")

# Whether to use the new RAG API (True) or legacy summarize API (False)
USE_RAG_API = os.getenv("USE_RAG_API", "true").lower() == "true"


def main() -> None:
    st.set_page_config(
        page_title="Briefly – AI News RAG Agent",
        page_icon="📰",
        layout="wide",
    )

    st.markdown(
        """
        <style>
        .stApp {
            background: radial-gradient(circle at top left, #192438, #050816 55%, #02030a 100%);
        }
        .nr-app-title {
            font-size: 2rem;
            font-weight: 600;
            margin-bottom: 0.1rem;
        }
        .nr-app-subtitle {
            font-size: 0.9rem;
            opacity: 0.8;
            margin-bottom: 1.5rem;
        }
        .nr-source-card {
            background: rgba(15, 15, 25, 0.96);
            border-radius: 0.9rem;
            padding: 0.75rem 0.9rem;
            margin-bottom: 0.75rem;
            border: 1px solid rgba(250, 250, 255, 0.06);
            box-shadow: 0 18px 45px rgba(0, 0, 0, 0.65);
        }
        .nr-source-header {
            display: flex;
            gap: 0.75rem;
            align-items: center;
            margin-bottom: 0.4rem;
        }
        .nr-source-favicon {
            width: 24px;
            height: 24px;
            border-radius: 6px;
        }
        .nr-source-header-text {
            display: flex;
            flex-direction: column;
            gap: 0.1rem;
        }
        .nr-source-domain {
            font-size: 0.7rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            opacity: 0.7;
        }
        .nr-source-title {
            font-size: 0.9rem;
            font-weight: 600;
        }
        .nr-source-footer {
            display: flex;
            justify-content: space-between;
            align-items: center;
            font-size: 0.75rem;
            opacity: 0.85;
            margin-top: 0.35rem;
        }
        .nr-source-link {
            color: #4C8DF5;
            text-decoration: none;
            font-weight: 500;
        }
        .nr-source-link:hover {
            text-decoration: underline;
        }
        .nr-footer {
            margin-top: 2rem;
            font-size: 0.75rem;
            opacity: 0.6;
            text-align: center;
        }
        .stButton > button {
            border-radius: 999px;
            background: #141b2b;
            color: #f5f5ff;
            border: 1px solid rgba(255, 255, 255, 0.14);
            padding: 0.35rem 1.1rem;
            font-size: 0.8rem;
        }
        .stButton > button:hover {
            background: #1e2740;
            border-color: rgba(255, 255, 255, 0.26);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    if "messages" not in st.session_state:
        st.session_state["messages"] = []
    if "last_context" not in st.session_state:
        st.session_state["last_context"] = None
    if "conversation_id" not in st.session_state:
        st.session_state["conversation_id"] = None

    with st.sidebar:
        st.markdown("### Settings")
        time_range = st.selectbox("Time range", ["24h", "7d", "30d", "all"], index=1)
        verification = st.checkbox(
            "Enable verification (slower, more accurate)", value=True
        )
        max_articles = st.slider("Max articles", min_value=3, max_value=15, value=10)

        if st.button("Reset conversation"):
            # Clear conversation from backend if using RAG API
            if USE_RAG_API and st.session_state.get("conversation_id"):
                try:
                    httpx.delete(
                        f"{API_BASE_URL}/rag/conversation/{st.session_state['conversation_id']}",
                        timeout=10.0,
                    )
                except Exception:
                    pass  # Ignore errors on cleanup
            st.session_state["messages"] = []
            st.session_state["last_context"] = None
            st.session_state["conversation_id"] = None
            if hasattr(st, "rerun"):
                st.rerun()
            else:
                st.experimental_rerun()
        
        # Show conversation ID in sidebar for debugging
        if st.session_state.get("conversation_id"):
            st.markdown(f"**Conversation ID:** `{st.session_state['conversation_id'][:8]}...`")

    st.markdown('<div class="nr-app-title">Briefly</div>', unsafe_allow_html=True)
    st.markdown(
        "<div class=\"nr-app-subtitle\">Briefly summarizes and verifies the latest news for you.</div>",
        unsafe_allow_html=True,
    )

    def call_rag_api(message: str, is_followup: bool = False) -> dict:
        """Call the RAG API for both initial queries and follow-ups."""
        conversation_id = st.session_state.get("conversation_id")
        
        try:
            response = httpx.post(
                f"{API_BASE_URL}/rag/query",
                json={
                    "message": message,
                    "conversation_id": conversation_id,
                    "time_range": time_range,
                    "max_articles": max_articles,
                    "include_debug": True,
                },
                timeout=120.0,
            )
            response.raise_for_status()
            return response.json()
        except Exception as exc:
            st.error(f"Error calling RAG API: {exc}")
            return {
                "answer_text": f"Error: {exc}",
                "answer_type": "error",
                "sources": [],
                "conversation_id": conversation_id or "",
            }

    def chat_about_context_legacy(followup: str) -> str:
        """Legacy chat function using direct Gemini calls (fallback)."""
        import google.generativeai as genai

        context = st.session_state.get("last_context") or {}
        summary_text = context.get("summary_text") or ""
        sources = context.get("sources") or []

        lines = []
        for idx, source in enumerate(sources[:8], start=1):
            title = source.get("title") or ""
            source_name = source.get("source") or ""
            url = source.get("url") or ""
            parts = [f"{idx}."]
            if title:
                parts.append(title)
            if source_name:
                parts.append(f"({source_name})")
            if url:
                parts.append(f"- {url}")
            lines.append(" ".join(parts))

        sources_text = "\n".join(lines)

        system_content = (
            "You are a helpful assistant discussing a previously generated news summary. "
            "Answer follow-up questions using only the summary and sources provided. "
            "If the user asks about information that is not covered by this context, say you do not know."
        )

        history_messages = []
        for msg in st.session_state.get("messages", []):
            if msg.get("role") == "user":
                content = msg.get("content") or msg.get("query") or ""
                if content:
                    history_messages.append({"role": "user", "content": content})
            elif msg.get("role") == "assistant" and msg.get("type") == "chat":
                content = msg.get("content") or ""
                if content:
                    history_messages.append({"role": "assistant", "content": content})

        user_context = (
            "Here is the existing news summary and list of sources:\n\n"
            f"SUMMARY:\n{summary_text}\n\n"
            f"SOURCES:\n{sources_text}\n\n"
            f"User follow-up question: {followup}"
        )

        messages = [{"role": "system", "content": system_content}]
        if history_messages:
            messages.extend(history_messages[-6:])
        messages.append({"role": "user", "content": user_context})

        try:
            if settings is not None and getattr(settings, "google_api_key", None):
                genai.configure(api_key=settings.google_api_key)
            model_name = (
                getattr(settings, "news_rag_model_name", None)
                or getattr(settings, "google_chat_model", "gemini-1.5-flash")
            )
            model = genai.GenerativeModel(model_name)
            prompt_lines = [
                f"{msg.get('role', 'user').upper()}: {msg.get('content', '')}"
                for msg in messages
            ]
            prompt = "\n\n".join(prompt_lines)
            response = model.generate_content(prompt)
            return response.text or ""
        except Exception as exc:
            st.error(f"Error calling chat model: {exc}")
            return "There was an error while answering your question about the summary."

    def handle_prompt_rag(prompt: str) -> None:
        """Handle prompt using the new RAG API."""
        st.session_state["messages"].append({"role": "user", "content": prompt})

        with st.chat_message("user", avatar="🧑"):
            st.markdown(prompt)

        is_initial = not st.session_state.get("conversation_id")
        spinner_text = "Fetching news and generating summary..." if is_initial else "Retrieving from sources and generating answer..."

        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner(spinner_text):
                data = call_rag_api(prompt, is_followup=not is_initial)

            # Update conversation ID from response
            if data.get("conversation_id"):
                st.session_state["conversation_id"] = data["conversation_id"]

            answer_text = data.get("answer_text", "")
            answer_type = data.get("answer_type", "summary")
            sources = data.get("sources", [])
            debug = data.get("debug")

            # Render based on answer type
            if answer_type == "summary":
                render_summary(answer_text)
                render_sources(sources)
            elif answer_type in ("followup_answer", "web_augmented_answer"):
                st.markdown(answer_text)
                if answer_type == "web_augmented_answer":
                    st.info("🔍 Additional web search was performed to answer this question.")
                if sources:
                    with st.expander(f"Sources used ({len(sources)})"):
                        render_sources(sources)
            else:
                st.markdown(answer_text)

            if debug:
                with st.expander("Debug info"):
                    st.json(debug)

        # Update session state
        st.session_state["last_context"] = {
            "query": prompt,
            "summary_text": answer_text,
            "sources": sources,
            "answer_type": answer_type,
        }
        st.session_state["messages"].append(
            {
                "role": "assistant",
                "type": "chat" if answer_type != "summary" else "summary",
                "content": answer_text,
                "summary_text": answer_text if answer_type == "summary" else None,
                "sources": sources,
                "meta": debug,
            }
        )

    def handle_prompt_legacy(prompt: str) -> None:
        """Handle prompt using the legacy summarize API."""
        st.session_state["messages"].append({"role": "user", "content": prompt})

        with st.chat_message("user", avatar="🧑"):
            st.markdown(prompt)

        if not st.session_state.get("last_context"):
            with st.chat_message("assistant", avatar="🤖"):
                with st.spinner("Contacting backend and generating summary..."):
                    try:
                        response = httpx.post(
                            f"{API_BASE_URL}/summarize",
                            json={
                                "query": prompt,
                                "time_range": time_range,
                                "verification": verification,
                                "max_articles": max_articles,
                            },
                            timeout=60.0,
                        )
                        response.raise_for_status()
                        data = response.json()
                    except Exception as exc:  # pragma: no cover - network dependent
                        st.error(f"Error calling backend: {exc}")
                        return

                error = (
                    (data.get("meta") or {}).get("error") if isinstance(data, dict) else None
                )
                if error:
                    st.warning(f"Backend returned an error: {error}")

                summary_text = data.get("summary_text", "") if isinstance(data, dict) else ""
                sources = data.get("sources") or [] if isinstance(data, dict) else []
                render_summary(summary_text)
                render_sources(sources)
                meta = data.get("meta") if isinstance(data, dict) else None
                if meta:
                    with st.expander("Debug info"):
                        st.json(meta)

            st.session_state["last_context"] = {
                "query": prompt,
                "summary_text": summary_text,
                "sources": sources,
                "meta": meta,
            }
            st.session_state["messages"].append(
                {
                    "role": "assistant",
                    "type": "summary",
                    "summary_text": summary_text,
                    "sources": sources,
                    "meta": meta,
                }
            )
        else:
            with st.chat_message("assistant", avatar="🤖"):
                with st.spinner("Talking about the retrieved news..."):
                    reply_text = chat_about_context_legacy(prompt)
                    st.markdown(reply_text)
            st.session_state["messages"].append(
                {
                    "role": "assistant",
                    "type": "chat",
                    "content": reply_text,
                }
            )

    # Choose which handler to use based on USE_RAG_API flag
    handle_prompt = handle_prompt_rag if USE_RAG_API else handle_prompt_legacy

    if not st.session_state["messages"]:
        st.markdown("#### Try an example question")
        example_prompts = [
            "Latest developments in solid-state batteries",
            "What is happening with global interest rates this week?",
            "Recent news about climate change policy in the EU",
            "What are the latest AI regulation headlines?",
        ]
        cols = st.columns(2)
        for idx, example in enumerate(example_prompts):
            col = cols[idx % 2]
            with col:
                if st.button(example, key=f"example-{idx}"):
                    handle_prompt(example)
                    if hasattr(st, "rerun"):
                        st.rerun()
                    else:
                        st.experimental_rerun()

    for msg in st.session_state["messages"]:
        role = msg.get("role", "assistant")
        avatar = "🧑" if role == "user" else "🤖"
        with st.chat_message("user" if role == "user" else "assistant", avatar=avatar):
            if role == "user":
                st.markdown(msg.get("content", msg.get("query", "")))
            else:
                msg_type = msg.get("type", "summary")
                if msg_type == "chat":
                    st.markdown(msg.get("content", ""))
                else:
                    render_summary(msg.get("summary_text", ""))
                    render_sources(msg.get("sources") or [])
                    meta = msg.get("meta")
                    if meta:
                        with st.expander("Debug info"):
                            st.json(meta)

    placeholder = (
        "Ask a follow-up question about this summary..."
        if st.session_state.get("last_context")
        else "Ask about current news..."
    )
    prompt = st.chat_input(placeholder)
    if prompt:
        handle_prompt(prompt)

    st.markdown(
        '<div class="nr-footer"> 2025 Briefly · AI News Intelligence</div>',
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
