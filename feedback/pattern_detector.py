# cross-session error pattern analysis
# feedback/pattern_detector.py
"""
Cross-session error pattern analysis.
After 3+ sessions, detects TRAIT-level patterns vs STATE-level patterns.

TRAIT: Niraj always makes sign errors (regardless of session state)
STATE: Niraj makes sign errors only when fatigue_score > 0.7

This distinction changes the intervention:
  TRAIT → teach the underlying concept or develop a checking habit
  STATE → manage the environmental conditions, flag as fatigue marker
"""
import logging
from collections import defaultdict
from typing import Optional

logger = logging.getLogger("SYRA.PatternDetector")


def analyse_patterns(
        session_logs: list[dict],
        min_sessions: int = 3,
) -> dict:
    """
    Analyse error patterns across all session logs.
    Returns a pattern report dict for teacher report and belief_updater.

    session_logs: list of to_log() dicts with turn data
    """
    if len(session_logs) < min_sessions:
        return {
            "sufficient_data": False,
            "sessions_analysed": len(session_logs),
            "patterns": [],
        }

    error_by_type = defaultdict(list)
    error_by_fatigue = defaultdict(list)
    error_by_topic = defaultdict(list)
    zpd_history = []

    for session in session_logs:
        fatigue = session.get("session_ctx", {}).get("fatigue_level", "low")
        is_anomaly = session.get("session_anomaly", False)

        for turn in session.get("turns", []):
            error_type = turn.get("error_type", "NONE")
            topic      = turn.get("topic", "unknown")
            zpd_pos    = turn.get("zpd_position", "IN")

            if error_type != "NONE":
                error_by_type[error_type].append({
                    "topic":   topic,
                    "fatigue": fatigue,
                    "anomaly": is_anomaly,
                })
                error_by_fatigue[fatigue].append(error_type)
                error_by_topic[topic].append(error_type)

            zpd_history.append(zpd_pos)

    patterns = []

    # Pattern 1: CARELESS errors under high fatigue
    careless_total  = len(error_by_type.get("CARELESS", []))
    careless_fatigue = sum(
        1 for e in error_by_type.get("CARELESS", [])
        if e["fatigue"] == "high"
    )
    if careless_total >= 3:
        fatigue_ratio = careless_fatigue / careless_total
        if fatigue_ratio >= 0.70:
            patterns.append({
                "type":       "STATE",
                "error":      "CARELESS",
                "trigger":    "high_fatigue",
                "confidence": round(fatigue_ratio, 2),
                "description": (
                    f"Careless errors strongly correlated with fatigue "
                    f"({careless_fatigue}/{careless_total} under high fatigue). "
                    "These are state-level, not trait-level. "
                    "Intervention: improve session timing, not concept teaching."
                ),
            })
        else:
            patterns.append({
                "type":       "TRAIT",
                "error":      "CARELESS",
                "trigger":    "general",
                "confidence": round(1 - fatigue_ratio, 2),
                "description": (
                    "Careless errors appear regardless of fatigue level. "
                    "Trait-level pattern. "
                    "Intervention: build self-checking habit explicitly."
                ),
            })

    # Pattern 2: CONCEPTUAL errors on specific topics
    for topic, errors in error_by_topic.items():
        conceptual_count = errors.count("CONCEPTUAL")
        if conceptual_count >= 2:
            patterns.append({
                "type":       "TRAIT",
                "error":      "CONCEPTUAL",
                "trigger":    topic,
                "confidence": min(0.95, conceptual_count / len(errors)),
                "description": (
                    f"Repeated conceptual errors on '{topic}' "
                    f"({conceptual_count} occurrences). "
                    "Deep misconception likely — target root belief specifically."
                ),
            })

    # Pattern 3: Dependency trajectory across sessions
    dep_by_session = [
        max([t.get("dependency_idx", 0) for t in s.get("turns", [])], default=0)
        for s in session_logs
    ]
    if len(dep_by_session) >= 3:
        trend = dep_by_session[-1] - dep_by_session[0]
        if trend > 0.20:
            patterns.append({
                "type":       "TREND",
                "error":      "DEPENDENCY",
                "trigger":    "cross_session",
                "confidence": 0.80,
                "description": (
                    f"Dependency index increasing across sessions "
                    f"({dep_by_session[0]:.2f} → {dep_by_session[-1]:.2f}). "
                    "Student becoming more passive over time. "
                    "Increase Socratic mode frequency."
                ),
            })

    # Pattern 4: ZPD — where does this student typically land?
    zpd_counts = {
        "ABOVE": zpd_history.count("ABOVE"),
        "IN":    zpd_history.count("IN"),
        "BELOW": zpd_history.count("BELOW"),
    }
    total_zpd = sum(zpd_counts.values()) or 1
    dominant_zpd = max(zpd_counts, key=zpd_counts.get)
    if zpd_counts[dominant_zpd] / total_zpd > 0.50:
        patterns.append({
            "type":        "ZPD_SIGNATURE",
            "error":       "NONE",
            "trigger":     "difficulty_calibration",
            "confidence":  round(zpd_counts[dominant_zpd] / total_zpd, 2),
            "description": (
                f"Student spends {zpd_counts[dominant_zpd]/total_zpd:.0%} "
                f"of turns in '{dominant_zpd}' ZPD. "
                + ("Content is consistently too hard — reduce default difficulty."
                   if dominant_zpd == "ABOVE" else
                   "Content is consistently too easy — increase default difficulty."
                   if dominant_zpd == "BELOW" else
                   "Difficulty calibration is good.")
            ),
        })

    return {
        "sufficient_data":   True,
        "sessions_analysed": len(session_logs),
        "patterns":          patterns,
        "error_summary":     {k: len(v) for k, v in error_by_type.items()},
        "zpd_distribution":  zpd_counts,
    }