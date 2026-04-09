# REFRAME/SIMPLIFY/CLOSE_RESET/DEPENDENCY_BREAK
# pedagogy/intervention_engine.py
"""
Decides WHICH intervention to deploy given the full cognitive state.
Four intervention types, each with entry conditions and exit conditions.

REFRAME:          Same concept, different angle or analogy
SIMPLIFY:         Drop difficulty, break into smaller pieces
CLOSE_AND_RESET:  WM overload — close concepts, pause new content
DEPENDENCY_BREAK: Switch to pure Socratic, force re-engagement

The engine does NOT build the prompt — it outputs a decision
that prompt/assembler.py translates into language.
"""
from dataclasses import dataclass
from typing import Optional
from cognition.affect_tracker    import AffectState
from cognition.zpd_estimator     import ZPDEstimate
from cognition.wm_monitor        import WMState
from cognition.dependency_tracker import DependencyState


INTERVENTION_TYPES = {
    "NONE",
    "REFRAME",
    "SIMPLIFY",
    "CLOSE_AND_RESET",
    "DEPENDENCY_BREAK",
}


@dataclass
class InterventionDecision:
    intervention_type: str
    rationale:         str
    instruction:       str
    urgency:           str   # LOW | MEDIUM | HIGH | CRITICAL


class InterventionEngine:

    def __init__(self):
        self._current:         str = "NONE"
        self._turns_in_state:  int = 0
        self._reframe_count:   int = 0

    def decide(
            self,
            affect:          AffectState,
            zpd:             ZPDEstimate,
            wm:              WMState,
            dependency:      DependencyState,
            consecutive_confused: int,
            current_topic:   Optional[str],
    ) -> InterventionDecision:
        """
        Evaluate cognitive state and return intervention decision.
        Priority order: WM overload → dependency alarm → stuck → above ZPD.
        """
        # Priority 1: WM overload — most disruptive, must fix first
        if wm.overload_detected:
            return self._close_and_reset(wm, current_topic)

        # Priority 2: Dependency alarm
        if dependency.alarm_triggered:
            return self._dependency_break(dependency)

        # Priority 3: STUCK frustration
        if affect.frustration_type == "STUCK":
            if affect.trajectory == "deteriorating":
                return self._reframe(current_topic, urgency="HIGH")
            return self._simplify(zpd, urgency="MEDIUM")

        # Priority 4: Above ZPD for multiple turns
        if zpd.position == "ABOVE" and zpd.confidence > 0.5:
            if consecutive_confused >= 3:
                return self._simplify(zpd, urgency="HIGH")
            elif consecutive_confused >= 2:
                return self._reframe(current_topic, urgency="MEDIUM")
            return InterventionDecision(
                intervention_type="NONE",
                rationale="Above ZPD but early — monitoring",
                instruction="",
                urgency="LOW",
            )

        # Priority 5: Rising frustration warning
        if (affect.trajectory == "deteriorating"
                and affect.intensity > 0.40):
            return InterventionDecision(
                intervention_type="REFRAME",
                rationale="Frustration rising — pre-emptive reframe",
                instruction=(
                    "Change your approach before frustration peaks. "
                    "Try a different angle or analogy. "
                    "One warm acknowledgment first."
                ),
                urgency="LOW",
            )

        # No intervention needed
        return InterventionDecision(
            intervention_type="NONE",
            rationale="Student in optimal state",
            instruction="",
            urgency="LOW",
        )

    def _close_and_reset(
            self, wm: WMState, topic: Optional[str]
    ) -> InterventionDecision:
        concepts = ", ".join(wm.active_concepts) if wm.active_concepts else topic or "current concept"
        return InterventionDecision(
            intervention_type="CLOSE_AND_RESET",
            rationale=f"WM overload: {wm.overload_error_count if hasattr(wm, 'overload_error_count') else '?'} overload errors",
            instruction=(
                "STOP. Do not introduce any new content. "
                f"Run verbal closure: 'Before we continue — let's lock in what we have. "
                f"We established that [summarise {concepts}]. Take a breath.' "
                "Then ask ONE simple question about what was just closed, "
                "not about anything new."
            ),
            urgency="CRITICAL",
        )

    def _dependency_break(
            self, dependency: DependencyState
    ) -> InterventionDecision:
        return InterventionDecision(
            intervention_type="DEPENDENCY_BREAK",
            rationale=(
                f"Dependency alarm: index={dependency.index:.2f} "
                f"for {dependency.turns_at_alarm} turns"
            ),
            instruction=(
                "Student has become fully passive. "
                "Ask ONE open question and then WAIT silently. "
                "Do not rephrase, hint, or fill the silence. "
                "The student must produce something independently "
                "before you respond with anything substantive."
            ),
            urgency="HIGH",
        )

    def _reframe(
            self, topic: Optional[str], urgency: str = "MEDIUM"
    ) -> InterventionDecision:
        self._reframe_count += 1
        approaches = [
            "Use a real-world analogy — something the student would encounter in daily life.",
            "Draw the concept visually through words — 'imagine you have...'",
            "Work backwards — start from the answer and ask what would produce it.",
            "Connect to something they said they were confident about earlier.",
            "Break the concept into the smallest possible first step.",
        ]
        approach = approaches[self._reframe_count % len(approaches)]
        return InterventionDecision(
            intervention_type="REFRAME",
            rationale=f"Student stuck on {topic or 'concept'} — trying new angle",
            instruction=(
                f"Current approach isn't working. {approach} "
                "Keep it very short. One idea only."
            ),
            urgency=urgency,
        )

    def _simplify(
            self, zpd: ZPDEstimate, urgency: str = "MEDIUM"
    ) -> InterventionDecision:
        return InterventionDecision(
            intervention_type="SIMPLIFY",
            rationale=f"Above ZPD (score={zpd.score:.2f}) — reducing difficulty",
            instruction=(
                "Step back significantly. "
                "Find the smallest sub-step the student CAN do correctly. "
                "Start there. Build up from one success. "
                "Do not try to continue from where the difficulty peaked."
            ),
            urgency=urgency,
        )