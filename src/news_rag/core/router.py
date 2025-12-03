from typing import Literal


def classify_query(query: str) -> Literal["news", "general"]:
    """Very simple heuristic router for queries.

    Queries mentioning time-related phrases are treated as news; others
    fall back to general knowledge.
    """
    lowered = query.lower()
    time_markers = [
        "today",
        "latest",
        "breaking",
        "this week",
        "this month",
        "yesterday",
        "2025",
        "2024",
        "2023",
    ]
    if any(marker in lowered for marker in time_markers):
        return "news"
    return "general"
