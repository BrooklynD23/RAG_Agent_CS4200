# Future Work and Extensions

This document collects ideas for how the AI News RAG Agent can be
extended beyond the initial CS4200 project. These are not requirements
for the base implementation but directions for future iterations or
research projects.

## 1. Retrieval and data sources

- **Richer news APIs and sources**  
  Add additional providers (e.g., NewsAPI, RSS feeds, custom scrapers)
  and ensemble their results with Tavily and GNews.

- **Better ranking and de-duplication**  
  Use semantic similarity or cross-encoder rerankers to remove near-
  duplicate articles and prioritize diverse, high-quality sources.

- **Historical archives**  
  Support queries over longer time ranges by indexing historical
  articles in a vector store (e.g., Chroma, FAISS) and combining them
  with fresh news.

## 2. Agent behavior and reasoning

- **Richer LangGraph workflows**  
  Expand the current search–grade–summarize–verify loop with branches
  for:
  - Comparing narratives across outlets ("contrastive" summaries).
  - Following up on unresolved questions with targeted queries.
  - Drilling down into specific entities (companies, politicians,
    regions).

- **Multi-step user interactions**  
  Turn the agent into a conversational assistant that can handle
  follow-up questions, refine time ranges, and "zoom in" or "zoom out"
  on topics while reusing prior context.

- **Tool selection and cost control**  
  Add a policy that decides when to use verification, how many articles
  to fetch, and which model to call based on user preferences
  (e.g. "fast" vs "thorough" modes).

## 3. Verification and safety

- **Richer critic outputs**  
  Extend the verification schema to:
  - Tag each summary sentence with a confidence score.
  - Provide short natural-language justifications for verdicts.
  - Highlight potentially misleading or speculative claims.

- **Cross-source consistency checks**  
  Detect when two sources strongly disagree and surface that explicitly
  (e.g., "Outlets A and B report conflicting timelines").

- **Safety filters**  
  Integrate content filters to detect and mitigate problematic outputs
  (hate speech, self-harm content, etc.), either via OpenAI safety
  models or a custom classifier.

## 4. UX and delivery channels

- **Richer Streamlit UX**  
  Improve the UI to include:
  - Highlighted sentences with clickable citations.
  - Side-by-side views of summary vs. source excerpts.
  - Simple controls for exporting summaries as PDF/Markdown.

- **Alternative frontends**  
  Expose the backend via:
  - A simple web dashboard built with React or similar.
  - A CLI tool for power users.
  - Chat integrations (e.g., Slack/Discord bots) that call the same
    `/summarize` endpoint.

## 5. Performance and scalability

- **Caching and rate limiting**  
  Persist the current in-memory cache or back it with Redis; add rate
  limiting per IP or per API key to protect external providers.

- **Batching and streaming**  
  Experiment with streaming LLM responses to the UI and/or batching
  multiple queries for classroom-scale demos.

- **Horizontal scaling**  
  Package the service for deployment on container platforms (e.g.,
  Kubernetes) with autoscaling based on CPU or request rate.

## 6. Academic and research directions

- **Evaluation datasets**  
  Build a small labeled dataset of news questions, reference summaries,
  and human ratings for faithfulness/coverage. Use this to compare
  different prompts, models, or retrieval strategies.

- **Explainability**  
  Study how well users understand the connection between summaries and
  sources; experiment with different citation and visualization styles.

- **Multi-lingual news summarization**  
  Extend the system to retrieve and summarize news in languages other
  than English, with mixed-language sources, and evaluate cross-lingual
  faithfulness.

These ideas are intentionally broad; future project teams can select a
subset that matches their interests and available time.
