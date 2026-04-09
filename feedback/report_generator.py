# teacher report: belief state, ZPD, affect
# feedback/report_generator.py
"""
Generates human-readable teacher report from session data.
Covers: belief state, ZPD trajectory, affect timeline,
error patterns, root beliefs, and next-session recommendations.
"""
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional

from feedback.pattern_detector import analyse_patterns
import memory.belief_graph as bg_module
from memory.profile_manager import ProfileManager

logger = logging.getLogger("SYRA.Report")


def generate_report(student_id: str) -> str:
    """
    Generate and print a teacher report for a student.
    Also saves to sessions/{student_id}/report_{date}.txt
    """
    sessions_dir = Path(f"sessions/{student_id}")
    if not sessions_dir.exists():
        print(f"  No session data found for {student_id}")
        return ""

    # Load all session logs
    session_logs = []
    for log_path in sorted(sessions_dir.glob("session_*.json")):
        try:
            log = json.loads(log_path.read_text(encoding="utf-8"))
            session_logs.append(log)
        except Exception:
            pass

    # Load profile and belief graph
    pm           = ProfileManager(student_id)
    belief_graph = bg_module.load(student_id)
    patterns     = analyse_patterns(session_logs)

    report_lines = [
        "=" * 60,
        f"SYRA TEACHER REPORT",
        f"Student: {student_id}",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"Sessions analysed: {len(session_logs)}",
        "=" * 60,
        "",
    ]

    # ── Profile summary ────────────────────────────────────────────────
    report_lines += ["STUDENT PROFILE", "-" * 40]
    ipc = pm.profile.get("ipc", {})
    ls  = pm.profile.get("learning_style", {})
    report_lines += [
        f"Archetype:          {ipc.get('archetype', 'unknown').upper()}",
        f"Dominance:          {ipc.get('dominance', 0.5):.2f}",
        f"Warmth:             {ipc.get('warmth', 0.6):.2f}",
        f"Pace:               {ipc.get('pace', 'medium')}",
        f"Processing style:   {ls.get('processing_style', 'unknown')}",
        f"Goal:               {ls.get('goal_type', 'unknown')}",
        f"Encouragement need: {ls.get('encouragement_need', 'medium')}",
        f"Depth preference:   {ls.get('depth_vs_breadth', 'balanced')}",
        "",
    ]

    # ── Belief graph ───────────────────────────────────────────────────
    report_lines += ["BELIEF STATE", "-" * 40]
    concepts = belief_graph.get("concepts", {})
    if concepts:
        for concept, node in concepts.items():
            proc  = node.get("procedural_confidence",   0.5)
            conc  = node.get("conceptual_confidence",   0.2)
            meta  = node.get("metacognitive_awareness", 0.3)
            roots = [rb for rb in node.get("root_beliefs", [])
                     if not rb.get("resolved", False)]
            errs  = node.get("error_history", {})

            report_lines += [
                f"  {concept}:",
                f"    Procedural:   {proc:.0%}",
                f"    Conceptual:   {conc:.0%}",
                f"    Metacognitive:{meta:.0%}",
            ]
            if roots:
                report_lines.append("    Root beliefs (unresolved):")
                for rb in roots:
                    report_lines.append(
                        f"      • \"{rb['belief']}\" "
                        f"(confidence {rb.get('confidence', 0.7):.0%})"
                    )
            if errs:
                err_str = ", ".join(
                    f"{k}:{v}" for k, v in errs.items()
                )
                report_lines.append(f"    Error history: {err_str}")
            report_lines.append("")
    else:
        report_lines += ["  No belief data yet — needs more sessions.", ""]

    # ── Session summary ────────────────────────────────────────────────
    report_lines += ["SESSION HISTORY", "-" * 40]
    for i, log in enumerate(session_logs[-5:], 1):   # last 5 sessions
        score   = log.get("comprehension_score", 0)
        cls     = log.get("session_classification", "NORMAL")
        topics  = log.get("topics_covered", [])
        insight = log.get("key_insight", "")
        anomaly = log.get("session_anomaly", False)
        date    = log.get("start_time", "")[:10]

        report_lines += [
            f"  Session {i} [{date}]:",
            f"    Score:          {score}/100  [{cls}]"
            + (" ⚠️ ANOMALY" if anomaly else ""),
            f"    Topics:         {', '.join(topics) or 'none recorded'}",
            f"    Key insight:    {insight}",
            "",
        ]

    # ── Cross-session patterns ─────────────────────────────────────────
    report_lines += ["CROSS-SESSION PATTERNS", "-" * 40]
    if patterns.get("sufficient_data"):
        for p in patterns.get("patterns", []):
            label = f"[{p['type']}] [{p['error']}]"
            report_lines += [
                f"  {label}",
                f"  {p['description']}",
                f"  Confidence: {p['confidence']:.0%}",
                "",
            ]
        zpd = patterns.get("zpd_distribution", {})
        total_zpd = sum(zpd.values()) or 1
        report_lines += [
            "  ZPD distribution:",
            f"    IN-ZPD:    {zpd.get('IN', 0)/total_zpd:.0%}",
            f"    Above-ZPD: {zpd.get('ABOVE', 0)/total_zpd:.0%}",
            f"    Below-ZPD: {zpd.get('BELOW', 0)/total_zpd:.0%}",
            "",
        ]
    else:
        report_lines += [
            f"  Insufficient data ({patterns.get('sessions_analysed', 0)} sessions). "
            f"Minimum {3} sessions needed.",
            "",
        ]

    # ── Anomaly log ────────────────────────────────────────────────────
    anomaly_log = pm.profile.get("anomaly_log", [])
    if anomaly_log:
        report_lines += ["ANOMALY SESSIONS", "-" * 40]
        for a in anomaly_log:
            report_lines.append(
                f"  {a.get('date', '')[:10]}: {a.get('reason', '')} "
                f"— profile update blocked"
            )
        report_lines.append("")

    # ── Recommendations ────────────────────────────────────────────────
    report_lines += ["RECOMMENDATIONS FOR NEXT SESSION", "-" * 40]
    recs = _generate_recommendations(
        pm.profile, belief_graph, patterns, session_logs
    )
    for rec in recs:
        report_lines.append(f"  • {rec}")
    report_lines += ["", "=" * 60]

    report_text = "\n".join(report_lines)

    # Print to terminal
    print(report_text)

    # Save to file
    out_path = Path(f"sessions/{student_id}") / \
        f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    out_path.write_text(report_text, encoding="utf-8")
    print(f"\n  Report saved → {out_path}")

    return report_text


def _generate_recommendations(
        profile:      dict,
        belief_graph: dict,
        patterns:     dict,
        session_logs: list,
) -> list[str]:
    recs = []

    # From belief graph
    for concept, node in belief_graph.get("concepts", {}).items():
        conc = node.get("conceptual_confidence", 0.5)
        proc = node.get("procedural_confidence",  0.5)
        roots = [rb for rb in node.get("root_beliefs", [])
                 if not rb.get("resolved", False)]

        if conc < 0.30 and proc > 0.60:
            recs.append(
                f"'{concept}': Strong procedure, weak conceptual understanding. "
                "Prioritise WHY questions over HOW questions next session."
            )
        if roots:
            recs.append(
                f"'{concept}': Unresolved root belief — '{roots[0]['belief']}'. "
                "Design a cognitive conflict question to surface this."
            )

    # From patterns
    for p in patterns.get("patterns", []):
        if p["type"] == "TREND" and p["error"] == "DEPENDENCY":
            recs.append(
                "Dependency increasing across sessions. "
                "Increase Socratic question ratio. Reduce answer-giving."
            )
        if p["type"] == "STATE" and p["error"] == "CARELESS":
            recs.append(
                "Careless errors correlate with fatigue. "
                "Schedule sessions earlier in day if possible. "
                "Consider shorter sessions."
            )

    # From last session
    if session_logs:
        last = session_logs[-1]
        rec_next = last.get("recommended_next_topic")
        if rec_next:
            recs.append(f"Continue from: {rec_next}")
        struggling = last.get("topics_struggling", [])
        if struggling:
            recs.append(
                f"Topics needing review: {', '.join(struggling[:3])}"
            )

    # ZPD calibration
    zpd = patterns.get("zpd_distribution", {})
    total = sum(zpd.values()) or 1
    if zpd.get("ABOVE", 0) / total > 0.50:
        recs.append(
            "Student consistently above ZPD — start next session at lower difficulty."
        )

    if not recs:
        recs.append("Continue current approach — student progressing normally.")

    return recs