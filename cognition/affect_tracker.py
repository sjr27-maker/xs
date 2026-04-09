# trajectory + STUCK vs PRODUCTIVE classification
# cognition/affect_tracker.py
"""
Tracks emotional state trajectory across turns.
Critical distinction: PRODUCTIVE frustration vs STUCK frustration.

PRODUCTIVE: student is frustrated but still attempting independently.
            Keep going — this is where learning happens.
STUCK:      student is frustrated AND attempts are collapsing.
            Intervene now — before full disengagement.

The old architecture detected current state.
This tracks trajectory — direction and rate of change.
"""
from dataclasses import dataclass, field
from collections import deque
from typing import Optional
from config import AFFECT_WINDOW, STUCK_THRESHOLD, PRODUCTIVE_MAX


@dataclass
class AffectState:
    emotional_state: str    # neutral|engaged|frustrated|anxious|excited
    trajectory:      str    # stable|rising|deteriorating|recovering
    frustration_type: str   # PRODUCTIVE|STUCK|NONE
    intervention_needed: bool = False
    intensity:       float = 0.0   # 0.0 to 1.0


# Signal weights for frustration score
_SIGNAL_WEIGHTS = {
    "giving_up":       0.40,
    "high_fillers":    0.20,
    "energy_decline":  0.20,
    "errors_rising":   0.20,
}

# Signals that indicate STILL attempting (productive)
ATTEMPTING_MARKERS = {
    "let me try", "wait", "actually", "hold on", "i think",
    "maybe if i", "so if", "then that means", "let me check",
}


class AffectTracker:

    def __init__(self, window: int = AFFECT_WINDOW):
        self._window  = window
        self._history: deque = deque(maxlen=window)
        self._frustration_scores: deque = deque(maxlen=window)

    def update(
            self,
            ipc_giving_up:       bool,
            filler_count:        int,
            energy_trend:        str,   # rising|stable|declining
            error_type:          str,   # from error_classifier
            student_text:        str,
            consecutive_confused: int,
    ) -> AffectState:

        # Compute frustration score for this turn
        f_score = 0.0
        if ipc_giving_up:
            f_score += _SIGNAL_WEIGHTS["giving_up"]
        if filler_count >= 4:
            f_score += _SIGNAL_WEIGHTS["high_fillers"] * min(filler_count / 5.0, 1.0)
        if energy_trend == "declining":
            f_score += _SIGNAL_WEIGHTS["energy_decline"]
        if error_type in ("CONCEPTUAL", "OVERLOAD_INDUCED"):
            f_score += _SIGNAL_WEIGHTS["errors_rising"]
        if consecutive_confused >= 2:
            f_score = min(f_score + 0.15, 1.0)

        self._frustration_scores.append(f_score)

        # Classify emotional state
        avg_score = sum(self._frustration_scores) / len(self._frustration_scores)

        if   avg_score < 0.15: emotional_state = "engaged"
        elif avg_score < 0.35: emotional_state = "neutral"
        elif avg_score < 0.55: emotional_state = "frustrated"
        elif avg_score < 0.75: emotional_state = "highly_frustrated"
        else:                  emotional_state = "disengaged"

        # Trajectory — compare first half to second half of window
        scores = list(self._frustration_scores)
        if len(scores) >= 2:
            mid   = len(scores) // 2
            first = sum(scores[:mid]) / max(len(scores[:mid]), 1)
            last  = sum(scores[mid:]) / max(len(scores[mid:]), 1)
            delta = last - first
            if   delta >  0.10: trajectory = "deteriorating"
            elif delta < -0.10: trajectory = "recovering"
            else:               trajectory = "stable"
        else:
            trajectory = "stable"

        # PRODUCTIVE vs STUCK distinction
        # STUCK = frustrated AND not attempting independently
        text_lower    = student_text.lower()
        still_trying  = any(m in text_lower for m in ATTEMPTING_MARKERS)
        word_count    = len(student_text.split())

        if avg_score < PRODUCTIVE_MAX:
            frustration_type = "NONE"
        elif still_trying and word_count > 8:
            frustration_type = "PRODUCTIVE"
        elif ipc_giving_up or word_count < 5:
            frustration_type = "STUCK"
        elif avg_score > STUCK_THRESHOLD:
            frustration_type = "STUCK"
        else:
            frustration_type = "PRODUCTIVE"

        intervention_needed = (
            frustration_type == "STUCK"
            or trajectory == "deteriorating" and avg_score > 0.45
        )

        state = AffectState(
            emotional_state=emotional_state,
            trajectory=trajectory,
            frustration_type=frustration_type,
            intervention_needed=intervention_needed,
            intensity=round(avg_score, 3),
        )
        self._history.append(state)
        return state

    def get_instruction(self, state: AffectState) -> str:
        """Instruction string for prompt assembler."""
        if state.frustration_type == "STUCK":
            return (
                "Student is STUCK — not attempting independently anymore. "
                "Maximum warmth. Do not ask retrieval questions. "
                "Do not give the answer. Ask ONE simple question that "
                "creates an immediate small win. Validate their effort first."
            )
        if state.frustration_type == "PRODUCTIVE":
            return (
                "Student is productively struggling — this is good. "
                "Acknowledge the effort: 'you're on the right track'. "
                "Give a small nudge, not the answer. Let them work through it."
            )
        if state.trajectory == "deteriorating":
            return (
                "Affect deteriorating. Reduce pressure. "
                "One warm encouraging sentence before the next question."
            )
        if state.emotional_state == "engaged":
            return "Student is engaged — can handle slightly higher challenge."
        return ""

    def get_trajectory_summary(self) -> str:
        """For teacher report."""
        if not self._history:
            return "No affect data."
        states = list(self._history)
        stuck_count = sum(1 for s in states if s.frustration_type == "STUCK")
        prod_count  = sum(1 for s in states if s.frustration_type == "PRODUCTIVE")
        return (
            f"Turns: {len(states)} | "
            f"STUCK: {stuck_count} | PRODUCTIVE: {prod_count} | "
            f"Final trajectory: {states[-1].trajectory}"
        )