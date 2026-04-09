# updates belief_graph from session evidence
# feedback/belief_updater.py
"""
Updates belief_graph.json from session evidence.
Called at session end — runs once, accuracy matters more than speed.

Key principle: only update beliefs from RELIABLE evidence.
  - ANOMALY sessions: blocked entirely
  - CARELESS errors: do not reduce procedural confidence
  - OVERLOAD errors: do not reduce any confidence
  - CONCEPTUAL errors: reduce conceptual confidence, add root belief
  - Successful independent recall: increase stability
"""
import logging
from datetime import datetime, timedelta
from typing import Optional
from config import SR_STABILITY_GROWTH, SR_INITIAL_STABILITY

logger = logging.getLogger("SYRA.BeliefUpdater")


def update_from_session(
        belief_graph:    dict,
        session_turns:   list,
        session_anomaly: bool,
        subject:         str,
) -> dict:
    """
    Update belief_graph from all turns in a session.
    Returns updated graph — caller saves to disk.
    """
    if session_anomaly:
        logger.info("Belief update BLOCKED — anomaly session")
        return belief_graph

    if not session_turns:
        return belief_graph

    concepts = belief_graph.setdefault("concepts", {})

    # Group turns by topic
    turns_by_topic: dict = {}
    for turn in session_turns:
        topic = getattr(turn, "topic", None) or \
                (turn.get("topic") if isinstance(turn, dict) else None) or \
                "general"
        error = getattr(turn, "error_type", None) or \
                (turn.get("error_type") if isinstance(turn, dict) else "NONE") or \
                "NONE"
        student = getattr(turn, "student_text", None) or \
                  (turn.get("student_text") if isinstance(turn, dict) else "") or ""
        ai_resp = getattr(turn, "ai_response", None) or \
                  (turn.get("ai_response") if isinstance(turn, dict) else "") or ""
        zpd     = getattr(turn, "zpd_position", None) or \
                  (turn.get("zpd_position") if isinstance(turn, dict) else "IN") or "IN"

        turns_by_topic.setdefault(topic, []).append({
            "error":   error,
            "student": student,
            "ai":      ai_resp,
            "zpd":     zpd,
        })

    for topic, turns in turns_by_topic.items():
        node = concepts.setdefault(topic, {
            "procedural_confidence":   0.50,
            "conceptual_confidence":   0.20,
            "metacognitive_awareness": 0.30,
            "root_beliefs":            [],
            "error_history":           {},
            "stability":               SR_INITIAL_STABILITY,
            "last_reviewed":           datetime.now().isoformat(),
            "next_review_due":         datetime.now().isoformat(),
            "total_sessions":          0,
        })

        node["total_sessions"] = node.get("total_sessions", 0) + 1

        # Count error types this session
        error_counts = {}
        for t in turns:
            e = t["error"]
            error_counts[e] = error_counts.get(e, 0) + 1
            node.setdefault("error_history", {})[e] = (
                node["error_history"].get(e, 0) + 1
            )

        # Procedural update
        # Only update from PROCEDURAL errors and successful turns
        procedural_errors = error_counts.get("PROCEDURAL", 0)
        successful_turns  = error_counts.get("NONE", 0)
        total_turns       = len(turns)

        if total_turns > 0:
            success_rate = successful_turns / total_turns
            # Cap delta at 0.10 per session
            proc_delta = (success_rate - 0.5) * 0.10
            # Procedural errors push down, successes push up
            proc_delta -= procedural_errors * 0.04
            node["procedural_confidence"] = round(
                max(0.0, min(1.0,
                    node["procedural_confidence"] + proc_delta
                )), 3
            )

        # Conceptual update
        # Only from CONCEPTUAL errors — these carry root belief evidence
        conceptual_errors = error_counts.get("CONCEPTUAL", 0)
        if conceptual_errors > 0:
            # Each conceptual error reduces confidence slightly
            conc_delta = -conceptual_errors * 0.05
            node["conceptual_confidence"] = round(
                max(0.0, min(1.0,
                    node["conceptual_confidence"] + conc_delta
                )), 3
            )

        # Metacognitive update
        # Check for self-correction markers in student speech
        self_corrections = sum(
            1 for t in turns
            if any(m in t["student"].lower()
                   for m in ["wait", "actually", "i mean",
                              "let me check", "no wait"])
        )
        if self_corrections >= 2:
            node["metacognitive_awareness"] = round(
                min(1.0, node["metacognitive_awareness"] + 0.05), 3
            )

        # Spaced repetition stability update
        # Successful independent answers increase stability
        independent_successes = sum(
            1 for t in turns
            if t["error"] == "NONE"
            and len(t["student"].split()) > 5  # more than a yes/no
        )
        if independent_successes >= 2:
            node["stability"] = round(
                node.get("stability", SR_INITIAL_STABILITY)
                * SR_STABILITY_GROWTH, 2
            )
        elif error_counts.get("CONCEPTUAL", 0) >= 2:
            # Reset stability if deep confusion detected
            node["stability"] = SR_INITIAL_STABILITY

        # Update review schedule
        node["last_reviewed"]   = datetime.now().isoformat()
        node["next_review_due"] = (
            datetime.now()
            + timedelta(days=node["stability"])
        ).isoformat()

        logger.info(
            f"Belief updated: {topic} | "
            f"proc={node['procedural_confidence']:.0%} "
            f"conc={node['conceptual_confidence']:.0%}"
        )

    return belief_graph


def resolve_root_belief(
        belief_graph: dict,
        concept:      str,
        belief_text:  str,
) -> dict:
    """
    Mark a root belief as resolved.
    Called when SYRA detects a student has corrected
    a previously-held false belief.
    """
    node = belief_graph.get("concepts", {}).get(concept)
    if not node:
        return belief_graph

    for rb in node.get("root_beliefs", []):
        if belief_text.lower() in rb.get("belief", "").lower():
            rb["resolved"]     = True
            rb["resolved_date"] = datetime.now().isoformat()
            logger.info(f"Root belief resolved: {rb['belief']}")
            # Resolving a root belief boosts conceptual confidence
            node["conceptual_confidence"] = round(
                min(1.0, node["conceptual_confidence"] + 0.12), 3
            )
    return belief_graph


def flag_complexity_intimidation(
        belief_graph: dict,
        concept:      str,
) -> dict:
    """
    Records that student showed anticipatory anxiety
    before attempting a problem of this type.
    Informs future difficulty calibration.
    """
    node = belief_graph.get("concepts", {}).get(concept, {})
    if node:
        node["complexity_intimidation_count"] = (
            node.get("complexity_intimidation_count", 0) + 1
        )
    return belief_graph