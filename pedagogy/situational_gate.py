# hard constraint layer, blocks illegal actions
# pedagogy/situational_gate.py
"""
Hard constraint layer — runs before every SYRA response.
Blocks question types that are harmful given the current cognitive state.

This is the fix for the Niraj Q3 mistake: asking "which is smaller, ½ or 6?"
when the student was already frustrated. That question would be BLOCKED here.

Gates are HARD — not suggestions. The prompt assembler cannot soften them.
The model receives them in a CONSTRAINTS block that overrides everything above.
"""
from dataclasses import dataclass
from typing import Optional
from cognition.affect_tracker    import AffectState
from cognition.zpd_estimator     import ZPDEstimate
from cognition.wm_monitor        import WMState
from cognition.dependency_tracker import DependencyState


QUESTION_TYPES = {
    "RETRIEVAL",        # "What is the formula?" — recall
    "COMPREHENSION",    # "How do you apply this?"
    "CAUSAL",           # "Why does this work?"
    "TRANSFER",         # "What if the coefficient was negative?"
    "WIN_CREATING",     # "You almost had it — what were the two numbers?"
    "SOCRATIC",         # Pure question, no hint
    "CLOSURE",          # "Lock that in" — WM offload
}


@dataclass
class GateResult:
    allowed:              bool
    blocked_type:         Optional[str]   # what was blocked
    reason:               str
    allowed_alternatives: list[str]       # what IS allowed
    constraint_text:      str             # injected into CONSTRAINTS block


class SituationalGate:
    """
    Evaluates the current cognitive state and returns
    which question types are permitted this turn.
    """

    def evaluate(
            self,
            proposed_question_type: str,
            affect:          AffectState,
            zpd:             ZPDEstimate,
            wm:              WMState,
            dependency:      DependencyState,
            turn_num:        int,
    ) -> GateResult:
        """
        Main gate check.
        Returns GateResult with allowed status and constraint text.
        """
        constraints = []
        blocked     = None
        reason      = ""
        alternatives= []

        # ── Gate 1: WM overload overrides everything ──────────────────
        if wm.overload_detected:
            if proposed_question_type not in ("CLOSURE", "WIN_CREATING"):
                blocked = proposed_question_type
                reason  = "Working memory overload — must run closure first"
                alternatives = ["CLOSURE"]
                constraints.append(
                    "CONSTRAINT: Working memory is overloaded. "
                    "Do NOT ask any new questions. "
                    "Run verbal closure on active concepts only."
                )

        # ── Gate 2: Dependency alarm forces Socratic only ─────────────
        elif dependency.alarm_triggered:
            if proposed_question_type not in ("SOCRATIC", "WIN_CREATING"):
                blocked = proposed_question_type
                reason  = "Dependency alarm — Socratic mode only"
                alternatives = ["SOCRATIC"]
                constraints.append(
                    "CONSTRAINT: Student is fully passive. "
                    "Ask ONE guiding question only. "
                    "Give zero answers, steps, or hints. "
                    "Force the student to produce something independently."
                )

        # ── Gate 3: STUCK frustration blocks retrieval ────────────────
        elif affect.frustration_type == "STUCK":
            if proposed_question_type in ("RETRIEVAL", "CAUSAL", "TRANSFER"):
                blocked = proposed_question_type
                reason  = (
                    f"Student is STUCK — {proposed_question_type} questions "
                    "increase pressure and worsen state"
                )
                alternatives = ["WIN_CREATING", "COMPREHENSION"]
                constraints.append(
                    "CONSTRAINT: Student is in STUCK frustration. "
                    "RETRIEVAL, CAUSAL, and TRANSFER questions are BLOCKED. "
                    "Ask ONE win-creating question that lets them succeed immediately. "
                    "Validate their effort first with one warm sentence."
                )

        # ── Gate 4: Above ZPD blocks CAUSAL and TRANSFER ─────────────
        elif zpd.position == "ABOVE" and zpd.confidence > 0.5:
            if proposed_question_type in ("CAUSAL", "TRANSFER"):
                blocked = proposed_question_type
                reason  = "Above ZPD — causal questions require IN-ZPD stability first"
                alternatives = ["COMPREHENSION", "WIN_CREATING"]
                constraints.append(
                    "CONSTRAINT: Student is above their ZPD. "
                    "CAUSAL and TRANSFER questions are blocked. "
                    "Reduce to COMPREHENSION level. "
                    "Ask about procedure, not theory."
                )

        # ── Gate 5: Rising frustration warns against tests ────────────
        elif affect.trajectory == "deteriorating" and affect.intensity > 0.35:
            if proposed_question_type == "RETRIEVAL":
                blocked = proposed_question_type
                reason  = "Frustration rising — retrieval feels like a test"
                alternatives = ["COMPREHENSION", "WIN_CREATING"]
                constraints.append(
                    "CONSTRAINT: Frustration is rising. "
                    "Avoid retrieval questions — they feel like being tested. "
                    "Use one warm acknowledgment before the next question."
                )

        # ── Gate 6: Early turns block CAUSAL ─────────────────────────
        elif turn_num < 3 and proposed_question_type == "CAUSAL":
            blocked = proposed_question_type
            reason  = "Too early for causal questions — build procedural first"
            alternatives = ["COMPREHENSION", "RETRIEVAL"]
            constraints.append(
                "CONSTRAINT: Causal questions require procedural foundation. "
                "Too early in session — establish procedure first."
            )

        # ── All gates passed ──────────────────────────────────────────
        if blocked is None:
            constraint_text = ""
            if constraints:
                constraint_text = "\n".join(constraints)
            return GateResult(
                allowed=True,
                blocked_type=None,
                reason="",
                allowed_alternatives=[proposed_question_type],
                constraint_text=constraint_text,
            )

        constraint_text = "\n".join(constraints)
        return GateResult(
            allowed=False,
            blocked_type=blocked,
            reason=reason,
            allowed_alternatives=alternatives,
            constraint_text=constraint_text,
        )

    def get_always_on_constraints(
            self,
            affect:     AffectState,
            dependency: DependencyState,
            wm:         WMState,
    ) -> str:
        """
        Returns constraints that are always included regardless
        of specific question type. Appended to every prompt.
        """
        lines = []

        if affect.frustration_type == "STUCK":
            lines.append(
                "Always validate effort before asking anything. "
                "One warm sentence minimum."
            )
        if dependency.level in ("HIGH", "ALARM"):
            lines.append(
                "Keep SYRA responses short. Ask more, tell less."
            )
        if wm.is_full:
            lines.append(
                f"Do not introduce new concepts. "
                f"Active concepts are already at capacity: "
                f"{', '.join(wm.active_concepts)}."
            )

        return " | ".join(lines) if lines else ""