# persistent concept graph, stability, review schedule
# memory/belief_graph.py
"""
Persistent concept graph. Grows across sessions.
Stores what the student actually believes, not just mastery scores.
"""
import json
import logging
from pathlib import Path
from datetime import datetime, timedelta
from config import SR_INITIAL_STABILITY, SR_STABILITY_GROWTH, SR_REVIEW_THRESHOLD

logger = logging.getLogger("SYRA.BeliefGraph")


def load(student_id: str) -> dict:
    path = Path(f"sessions/{student_id}/belief_graph.json")
    if not path.exists():
        return {"concepts": {}, "created": datetime.now().isoformat()}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {"concepts": {}}


def save(student_id: str, graph: dict):
    path = Path(f"sessions/{student_id}/belief_graph.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    graph["last_updated"] = datetime.now().isoformat()
    path.write_text(
        json.dumps(graph, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )


def get_due_reviews(graph: dict) -> list[str]:
    """
    Returns concepts whose retention has decayed below threshold.
    Used by spaced_repetition.py to schedule interleaved review.
    """
    due   = []
    now   = datetime.now()
    for concept, node in graph.get("concepts", {}).items():
        last_str  = node.get("last_reviewed", now.isoformat())
        stability = node.get("stability", SR_INITIAL_STABILITY)
        try:
            last = datetime.fromisoformat(last_str)
            days_since = (now - last).total_seconds() / 86400
            retention  = 2 ** (-days_since / stability)
            if retention < SR_REVIEW_THRESHOLD:
                due.append(concept)
        except Exception:
            pass
    return due


def update_stability_after_recall(
        graph: dict, concept: str, successful: bool
) -> dict:
    """
    Update spaced repetition stability after a recall attempt.
    Successful recall increases stability (review interval grows).
    Failed recall resets stability.
    """
    node = graph.get("concepts", {}).get(concept)
    if not node:
        return graph

    if successful:
        node["stability"] = node.get("stability", SR_INITIAL_STABILITY) \
                            * SR_STABILITY_GROWTH
    else:
        node["stability"] = SR_INITIAL_STABILITY

    node["last_reviewed"]   = datetime.now().isoformat()
    node["next_review_due"] = (
        datetime.now() + timedelta(days=node["stability"])
    ).isoformat()
    return graph


def get_concept_summary(graph: dict, concept: str) -> str:
    """Short string for prompt assembler."""
    node = graph.get("concepts", {}).get(concept)
    if not node:
        return f"No prior data for {concept}."

    proc   = node.get("procedural_confidence",   0.5)
    conc   = node.get("conceptual_confidence",   0.2)
    meta   = node.get("metacognitive_awareness", 0.3)
    roots  = [rb for rb in node.get("root_beliefs", [])
              if not rb.get("resolved", False)]

    lines = [
        f"{concept}: procedure={proc:.0%} concept={conc:.0%} meta={meta:.0%}"
    ]
    for rb in roots[:2]:
        lines.append(f"  ROOT BELIEF: \"{rb['belief']}\"")
    return "\n".join(lines)