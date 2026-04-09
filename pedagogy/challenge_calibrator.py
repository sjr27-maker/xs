# ZPD-driven continuous difficulty scale
# pedagogy/challenge_calibrator.py
"""
Continuously calibrates question difficulty to keep student in ZPD.
Tracks which difficulty level produces optimal learning (IN-ZPD) for
this specific student historically.

Difficulty scale: 0.0 (trivial) to 1.0 (maximum challenge)
ZPD sweet spot: the difficulty range where student is IN-ZPD.
"""
from dataclasses import dataclass
from collections import deque
from typing import Optional
from cognition.zpd_estimator import ZPDEstimate
from config import ZPD_ABOVE_THRESHOLD, ZPD_BELOW_THRESHOLD


@dataclass
class ChalibState:
    current_difficulty: float
    zpd_sweet_spot:     float
    adjustment:         float   # how much we moved this turn
    direction:          str     # up | down | stable


class ChallengeCalibrator:
    """
    Adjusts difficulty per turn based on ZPD position.
    Tracks historical sweet spot per student.
    """

    def __init__(self, initial_difficulty: float = 0.50):
        self._difficulty      = initial_difficulty
        self._sweet_spot      = initial_difficulty
        self._history:  deque = deque(maxlen=10)
        self._in_zpd_levels:  list = []   # difficulties where student was IN-ZPD

    def update(
            self,
            zpd: ZPDEstimate,
            session_anomaly: bool = False,
    ) -> ChalibState:
        """
        Adjust difficulty based on ZPD estimate.
        Anomaly sessions: do not increase difficulty.
        """
        adjustment = 0.0

        if zpd.confidence < 0.3:
            # Not enough data — hold steady
            return ChalibState(
                current_difficulty = round(self._difficulty, 2),
                zpd_sweet_spot     = round(self._sweet_spot, 2),
                adjustment         = 0.0,
                direction          = "stable",
            )

        if zpd.position == "ABOVE":
            # Too hard — reduce, faster when anomaly session
            step = 0.12 if not session_anomaly else 0.18
            adjustment   = -min(step, self._difficulty - 0.10)
            self._difficulty = max(0.10, self._difficulty + adjustment)
            direction    = "down"

        elif zpd.position == "BELOW" and not session_anomaly:
            # Too easy — increase slowly
            step = 0.06
            adjustment   = min(step, 0.95 - self._difficulty)
            self._difficulty = min(0.95, self._difficulty + adjustment)
            direction    = "up"

        else:
            # IN-ZPD — record this level as good
            self._in_zpd_levels.append(self._difficulty)
            self._in_zpd_levels = self._in_zpd_levels[-10:]
            if self._in_zpd_levels:
                self._sweet_spot = round(
                    sum(self._in_zpd_levels) / len(self._in_zpd_levels), 3
                )
            direction = "stable"

        self._history.append({
            "difficulty": self._difficulty,
            "zpd":        zpd.position,
        })

        return ChalibState(
            current_difficulty = round(self._difficulty, 2),
            zpd_sweet_spot     = round(self._sweet_spot, 2),
            adjustment         = round(adjustment, 3),
            direction          = direction,
        )

    def get_difficulty_hint(self) -> str:
        """Instruction for prompt assembler."""
        d = self._difficulty
        if   d < 0.25: return "Very easy level — single-step problems only."
        elif d < 0.45: return "Easy level — familiar problem types with guidance."
        elif d < 0.60: return "Medium level — standard problems with some challenge."
        elif d < 0.75: return "Hard level — multi-step, requires synthesis."
        else:          return "Challenge level — novel problems, transfer required."

    def load_sweet_spot(self, stored_sweet_spot: float):
        """Load historical sweet spot from student profile."""
        if 0.1 <= stored_sweet_spot <= 0.95:
            self._sweet_spot  = stored_sweet_spot
            self._difficulty  = stored_sweet_spot