import os

import httpx
import streamlit as st

try:
    from src.news_rag.ui.components import render_summary, render_sources
except ModuleNotFoundError:
    from components import render_summary, render_sources


API_BASE_URL = os.getenv("NEWS_RAG_API_BASE_URL", "http://localhost:8000")


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

    with st.sidebar:
        st.markdown("### Settings")
        time_range = st.selectbox("Time range", ["24h", "7d", "30d", "all"], index=1)
        verification = st.checkbox(
            "Enable verification (slower, more accurate)", value=True
        )
        max_articles = st.slider("Max articles", min_value=3, max_value=15, value=10)

    st.markdown('<div class="nr-app-title">Briefly</div>', unsafe_allow_html=True)
    st.markdown(
        "<div class=\"nr-app-subtitle\">Briefly summarizes and verifies the latest news for you.</div>",
        unsafe_allow_html=True,
    )

    def handle_prompt(prompt: str) -> None:
        st.session_state["messages"].append({"role": "user", "query": prompt})

        with st.chat_message("user", avatar="🧑"):
            st.markdown(prompt)

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

        st.session_state["messages"].append(
            {
                "role": "assistant",
                "query": prompt,
                "summary_text": summary_text,
                "sources": sources,
                "meta": meta,
            }
        )

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
                    st.stop()

    for msg in st.session_state["messages"]:
        role = msg.get("role", "assistant")
        avatar = "🧑" if role == "user" else "🤖"
        with st.chat_message("user" if role == "user" else "assistant", avatar=avatar):
            if role == "user":
                st.markdown(msg.get("query", ""))
            else:
                render_summary(msg.get("summary_text", ""))
                render_sources(msg.get("sources") or [])
                meta = msg.get("meta")
                if meta:
                    with st.expander("Debug info"):
                        st.json(meta)

    prompt = st.chat_input("Ask about current news...")
    if prompt:
        handle_prompt(prompt)

    st.markdown(
        '<div class="nr-footer"> 2025 Briefly · AI News Intelligence</div>',
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
