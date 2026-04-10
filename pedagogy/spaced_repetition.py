# forgetting curve, interleaving scheduler
# pedagogy/spaced_repetition.py
"""
Schedules interleaved review of past concepts.
Uses exponential forgetting curve. Weaves review into
conversation naturally — never announces it as review.
"""
from datetime import datetime
from memory import belief_graph as bg
from config import SR_REVIEW_THRESHOLD
from typing import Optional

def get_review_prompt(
        graph:          dict,
        current_topic:  str,
        subject:        str,
) -> Optional[str]:
    """
    Returns a natural review instruction if any concept
    is due for retrieval practice.
    Returns None if nothing is due.
    """
    due = bg.get_due_reviews(graph)
    # Don't review the current topic — that's the lesson
    due = [d for d in due if d != current_topic]

    if not due:
        return None

    target = due[0]
    node   = graph.get("concepts", {}).get(target, {})
    proc   = node.get("procedural_confidence", 0.5)

    # Choose retrieval level based on confidence
    if proc < 0.50:
        review_type = "recall"
        instruction = (
            f"Briefly weave in a recall question about '{target}' — "
            f"something simple the student should remember from before. "
            f"Do this naturally, not as an announced review."
        )
    else:
        review_type = "application"
        instruction = (
            f"Briefly connect '{target}' to the current topic. "
            f"Ask the student how '{target}' relates to what we're doing now. "
            f"Keep it short — one question."
        )

    return instruction


