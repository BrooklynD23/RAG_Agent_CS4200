from typing import Iterable, Mapping, Any
from datetime import datetime
from urllib.parse import urlparse

import streamlit as st


def render_summary(summary_text: str) -> None:
    st.subheader("Summary")
    if not summary_text:
        st.write("No summary available yet.")
        return

    text = summary_text.strip()

    # If the model returned bullet points using the "•" character, format them nicely
    if "•" in text:
        parts = [part.strip() for part in text.split("•") if part.strip()]
        if parts:
            bullet_lines = [f"- {part}" for part in parts]
            bullets_md = "\n".join(bullet_lines)
            st.markdown(bullets_md)
            return

    # Fallback: render as plain text
    st.write(summary_text)


def render_sources(sources: Iterable[Mapping[str, Any]]) -> None:
    st.subheader("Sources")
    sources_list = list(sources)
    if not sources_list:
        st.write("No sources available yet.")
        return

    for source in sources_list:
        title = source.get("title") or "Source"
        url = source.get("url") or ""
        raw_source = source.get("source") or ""
        published_at = source.get("published_at")

        domain = ""
        if url:
            try:
                domain = urlparse(url).netloc
            except Exception:
                domain = ""
        if not domain:
            domain = raw_source

        favicon_url = ""
        if domain:
            favicon_url = f"https://www.google.com/s2/favicons?sz=64&domain={domain}"

        published_str = ""
        if isinstance(published_at, str):
            try:
                dt = datetime.fromisoformat(published_at.replace("Z", ""))
                published_str = dt.strftime("%b %d, %Y")
            except Exception:
                published_str = published_at
        elif isinstance(published_at, datetime):
            published_str = published_at.strftime("%b %d, %Y")

        card_html = f"""
        <div class="nr-source-card">
            <div class="nr-source-header">
                {('<img src="' + favicon_url + '" class="nr-source-favicon"/>') if favicon_url else ''}
                <div class="nr-source-header-text">
                    <div class="nr-source-domain">{domain}</div>
                    <div class="nr-source-title">{title}</div>
                </div>
            </div>
            <div class="nr-source-footer">
                <span class="nr-source-date">{published_str}</span>
                {('<a href="' + url + '" target="_blank" class="nr-source-link">Open article</a>') if url else ''}
            </div>
        </div>
        """
        st.markdown(card_html, unsafe_allow_html=True)
