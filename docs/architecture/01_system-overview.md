# AI-Powered News Summarizer with Source Attribution – System Overview

This document summarizes the project goals, high-level architecture, runtime flow, constraints, and sprint plan for the AI News RAG Agent, adapted from the `Test` specification.

## 1. Goal

Build a Retrieval-Augmented Generation (RAG) agent that:

- Accepts topical queries about current events.
- Fetches the most recent news from the open web.
- Produces concise, multi-source summaries.
- Attaches explicit source citations to every factual claim.
- Optionally runs a verification loop to minimize hallucinations.

Educational focus:

- Agentic RAG (Thought → Action → Observation loop).
- Real-time, ephemeral data ingestion instead of static corpora.
- Hallucination mitigation through strict source grounding.

## 2. High-level architecture

Logical layers:

1. **Input Router**
   - Classifies user input into:
     - `news_query` (time-sensitive, needs fresh data).
     - `general_knowledge` (could be handled by model parametric knowledge or a separate knowledge base).
   - For this project, we primarily optimize for `news_query`.

2. **Retrieval Layer (Router–Retriever)**
   - Uses Tavily as the primary search tool to fetch:
     - Titles
     - URLs
     - Cleaned article content or extended snippets.
   - Deduplicates and filters articles.
   - Falls back to GNews if Tavily credits are exhausted.

3. **Synthesis Layer (Summarizer)**
   - LLM-based summarization over the retrieved article set.
   - Ensures every factual statement is followed by one or more citations.
   - Handles conflicting information across sources.

4. **Optional Verification Layer (Critic Agent)**
   - Reads the draft summary.
   - Checks whether each claim is supported by provided article texts and citations.
   - Requests regeneration or revision if unsupported claims are detected.

5. **Agentic Control (LangGraph)**
   - Encapsulates a loop:
     - Determine query → Search → Grade relevance → Re-search if needed → Summarize → (Optionally verify).
   - Moves beyond linear chains to a stateful agent graph.

6. **Interface Layer**
   - Streamlit UI (for class/demo) or API + web frontend.
   - Displays:
     - User query.
     - Final summary.
     - Structured source list with expandable article excerpts.

## 3. Technology stack

**Language**

- Python 3.11+

**Core libraries**

- **LangChain** – base abstractions, prompts, tools, chains.
- **LangGraph** – agent graph and state machine.
- **httpx** – HTTP client for Tavily/GNews.
- **Pydantic** – schema validation for requests/responses.
- **Streamlit** – simple UI.
- **FastAPI** – HTTP API for the agent core.

**LLM providers**

- OpenAI (e.g., GPT-4 family / `gpt-4o-mini`) for main agent.
- Optional: local models via other providers for experimentation.

**Storage**

- Ephemeral retrieval by default (no persistent vector store required).
- Optional:
  - Local cache (in-memory or small store) for recent queries.

**Observability**

- Logging via stdlib logging or `structlog`.
- Optional: LangSmith traces for agent thought/action logs.

## 4. Runtime flow

1. User enters query in UI.
2. Router decides this is a `news_query`.
3. Agent executes **Search** node:
   - Calls Tavily with the query.
   - Receives list of candidate articles.
4. Agent executes **Grade** node:
   - Heuristically (or via LLM) evaluates relevance and coverage.
   - Optionally refines the query and loops back to `search_news`.
5. Agent executes **Write** node:
   - LLM summarizes articles.
   - Attaches citations per sentence.
   - Produces structured output: `summary`, `sources`, `metadata`.
6. Optional **Verify** node:
   - LLM critic checks support for claims against article texts.
   - If issues found: requests revised summary.
7. API/UI returns:
   - Final summary.
   - Ordered, deduplicated source list.
   - Debug metadata (e.g., number of search calls).

## 5. Constraints and trade-offs

- **Ephemeral, real-time data**
  - No stable ground-truth corpus; correctness is time-dependent.
- **Free-tier API limits**
  - Tavily credits are limited; requires fallback and caching.
- **Context window management**
  - News articles are long; content must be chunked and ranked.
- **Hallucination risk**
  - Mitigated by:
    - Aggressive source grounding.
    - Verification agent.
    - Citation requirements.

## 6. Sprint alignment (example 14-day plan)

- **Days 1–3**: Tool setup, Tavily integration, `fetch_news_*`.
- **Days 4–7**: Simple summarization chain (no full agent yet).
- **Days 8–11**: LangGraph agentification with search–grade–write loop.
- **Days 12–14**: Streamlit UI + polish, testing, and docs.
