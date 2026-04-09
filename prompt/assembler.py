# reads all modules, outputs system instruction
# prompt/assembler.py
"""
Reads from all cognitive and pedagogical layers.
Outputs one system instruction string per turn.

The assembler is a READER, not a THINKER.
All decisions are made by cognition/ and pedagogy/.
This file only translates those decisions into language.

Section order matters — later sections can override earlier ones.
The CONSTRAINTS section (from situational_gate) overrides everything.
"""
from typing import Optional
from dataclasses import dataclass

from cognition.belief_model      import BeliefModel
from cognition.zpd_estimator     import ZPDEstimate
from cognition.affect_tracker    import AffectState
from cognition.wm_monitor        import WMState, WMMonitor
from cognition.dependency_tracker import DependencyState
from cognition.error_classifier  import ErrorAnalysis
from pedagogy.situational_gate   import GateResult
from pedagogy.question_planner   import QuestionPlan
from intake.session_checker      import SessionContext
from intake.acoustic_analyzer    import AcousticVector
from config import MAX_PROMPT_WORDS, HISTORY_TURNS, RAG_CHARS


@dataclass
class AssemblerInput:
    """Everything the assembler needs. Collected by live_session.py per turn."""

    # Identity
    student_id:          str
    subject:             str
    grade:               int
    archetype:           str
    turn_num:            int

    # Session context
    session_ctx:         SessionContext

    # Acoustic / IPC
    acoustic:            AcousticVector

    # Cognitive state
    belief_snapshot:     str            # from BeliefModel.get_snapshot()
    zpd:                 ZPDEstimate
    affect:              AffectState
    wm:                  WMState
    dependency:          DependencyState
    error_analysis:      ErrorAnalysis
    consecutive_confused: int

    # Pedagogical decisions
    gate_result:         GateResult
    question_plan:       QuestionPlan
    wm_monitor:          WMMonitor     # for closure statements

    # Memory
    conversation_history: list[dict]   # last N turns
    unresolved_misconceptions: list[dict]
    topics_due_review:   list[str]
    spaced_review_prompt: Optional[str]

    # Content
    rag_context:         str
    style_mirror:        str


def build_system_prompt(inp: AssemblerInput) -> str:
    """
    Assembles the complete system instruction.
    Called once per turn in live_session.py.
    """
    parts = []

    # ── 1. Identity + voice rules ─────────────────────────────────────
    parts.append(
        f"You are SYRA, an adaptive AI tutor for Class {inp.grade} {inp.subject}. "
        "Use NCERT-aligned content only. "
        f"Always end with a question or challenge. "
        f"Voice output only — no markdown, bullets, or symbols. "
        f"Max {MAX_PROMPT_WORDS} words unless student asks for depth. "
        "Speak like a knowledgeable friend, not a formal system."
    )

    # ── 2. IPC style instruction ──────────────────────────────────────
    ipc = inp.acoustic
    if ipc.giving_up:
        style = (
            "Student is disengaging. Maximum warmth. "
            "Very short responses. One gentle question."
        )
    elif ipc.dominance > 0.65:
        style = "Student is assertive. Match energy. Be direct and confident."
    elif ipc.warmth > 0.65:
        style = "Student is warm and open. Be warm, encouraging, conversational."
    elif ipc.pace == "fast":
        style = "Student speaks fast. Keep responses punchy and energetic."
    elif ipc.pace == "slow":
        style = "Student speaks slowly and deliberately. Give space. Be patient."
    else:
        style = "Warm, encouraging, clear."
    parts.append(f"INTERACTION STYLE: {style}")

    # ── 3. Session context override ───────────────────────────────────
    if inp.session_ctx.anomaly_flag:
        parts.append(
            "SESSION CONTEXT: Student is fatigued and externally stressed. "
            f"Fatigue: {inp.session_ctx.fatigue_level}. "
            "Be warm and short. Do not push for depth today. "
            "Celebrate small wins. Reduce chunk size to 1."
        )
    elif inp.session_ctx.fatigue_level == "medium":
        parts.append(
            "SESSION CONTEXT: Student has moderate fatigue today. "
            "Keep sessions focused. Avoid very complex multi-step problems."
        )

    # ── 4. Belief snapshot ────────────────────────────────────────────
    if inp.belief_snapshot and inp.belief_snapshot != "Building belief model...":
        parts.append(
            f"STUDENT BELIEF STATE:\n{inp.belief_snapshot}"
        )

    # ── 5. ZPD position ───────────────────────────────────────────────
    zpd_instruction = _get_zpd_instruction(inp.zpd)
    if zpd_instruction:
        parts.append(f"ZPD: {zpd_instruction}")

    # ── 6. Affect state ───────────────────────────────────────────────
    from cognition.affect_tracker import AffectTracker
    affect_tracker_tmp = AffectTracker()
    affect_instr = affect_tracker_tmp.get_instruction(inp.affect)
    if affect_instr:
        parts.append(f"AFFECT: {affect_instr}")

    # ── 7. Working memory ─────────────────────────────────────────────
    wm_instr = inp.wm_monitor.get_instruction()
    if wm_instr:
        parts.append(f"WORKING MEMORY: {wm_instr}")

    # ── 8. Error response ─────────────────────────────────────────────
    from cognition.error_classifier import get_error_response_instruction
    error_instr = get_error_response_instruction(inp.error_analysis)
    if error_instr:
        parts.append(f"ERROR RESPONSE: {error_instr}")

    # ── 9. Pedagogical action ─────────────────────────────────────────
    parts.append(
        f"PEDAGOGICAL ACTION: {inp.question_plan.instruction}"
    )

    # ── 10. Spaced review ─────────────────────────────────────────────
    if inp.spaced_review_prompt:
        parts.append(
            f"SPACED REVIEW DUE: {inp.spaced_review_prompt}"
        )

    # ── 11. Unresolved misconceptions ─────────────────────────────────
    if inp.unresolved_misconceptions:
        mc = inp.unresolved_misconceptions[0]
        desc = mc.get("description", mc) if isinstance(mc, dict) else mc
        parts.append(
            f"KNOWN MISCONCEPTION: Student previously believed '{desc}'. "
            "If relevant, gently surface and correct."
        )

    # ── 12. CONSTRAINTS block (from situational gate) ─────────────────
    # This section overrides everything above.
    constraint_parts = []
    if inp.gate_result.constraint_text:
        constraint_parts.append(inp.gate_result.constraint_text)
    always_on = _get_always_on_constraints(inp)
    if always_on:
        constraint_parts.append(always_on)

    if constraint_parts:
        parts.append(
            "CONSTRAINTS (override everything above):\n"
            + "\n".join(constraint_parts)
        )

    # ── 13. Style mirror ──────────────────────────────────────────────
    if inp.style_mirror:
        parts.append(inp.style_mirror)

    # ── 14. Conversation history ──────────────────────────────────────
    if inp.conversation_history:
        recent = inp.conversation_history[-HISTORY_TURNS * 2:]
        lines  = []
        for turn in recent:
            role = "Student" if turn["role"] == "student" else "SYRA"
            lines.append(f"{role}: {turn['content']}")
        parts.append(
            "RECENT CONVERSATION (continue naturally):\n"
            + "\n".join(lines)
        )

    # ── 15. RAG content ───────────────────────────────────────────────
    if inp.rag_context:
        parts.append(
            f"NCERT CONTENT:\n{inp.rag_context[:RAG_CHARS]}"
        )

    return "\n\n".join(p for p in parts if p)


def _get_zpd_instruction(zpd: ZPDEstimate) -> str:
    if zpd.confidence < 0.30:
        return ""  # not enough data yet
    return {
        "ABOVE": (
            "Student is above their ZPD. Content too hard. "
            "Break into smaller pieces. No new concepts this turn."
        ),
        "IN":    "Student is in optimal ZPD. Maintain difficulty.",
        "BELOW": "Content may be too easy. Consider increasing challenge.",
    }.get(zpd.position, "")


def _get_always_on_constraints(inp: AssemblerInput) -> str:
    lines = []
    if inp.dependency.level in ("HIGH", "ALARM"):
        lines.append("Keep responses SHORT. Ask more, tell less.")
    if inp.wm.is_full and not inp.wm.overload_detected:
        lines.append(
            f"Do NOT introduce new concepts. "
            f"Active: {', '.join(inp.wm.active_concepts)}."
        )
    if inp.affect.frustration_type == "STUCK":
        lines.append(
            "Validate effort FIRST before any question."
        )
    return " | ".join(lines)