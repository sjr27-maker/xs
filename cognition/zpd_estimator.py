# ABOVE/IN/BELOW from multi-signal fusion
# cognition/zpd_estimator.py
"""
Estimates Zone of Proximal Development position each turn.
BELOW ZPD: content too easy — student bored, fast confident answers
IN ZPD:    optimal — effortful but independent, self-correcting
ABOVE ZPD: too hard — errors rising, dependency rising, latency rising

Returns a continuous score and a label.
Score: negative = above ZPD, zero = optimal, positive = below ZPD
"""
from dataclasses import dataclass
from collections import deque
from typing import Optional
from config import (
    ZPD_ABOVE_THRESHOLD, ZPD_BELOW_THRESHOLD,
    ZPD_LATENCY_WEIGHT, ZPD_CORRECTION_WEIGHT,
    ZPD_HEDGING_WEIGHT, ZPD_ERROR_WEIGHT,
)

HEDGING_WORDS = {
    "i think", "maybe", "perhaps", "not sure", "i guess",
    "probably", "i don't know", "kind of", "sort of",
    "i'm not confident", "i believe", "possibly",
}

SELF_CORRECTION_MARKERS = {
    "wait", "actually", "no wait", "let me check",
    "hold on", "let me redo", "i made a mistake",
    "correction", "sorry", "i mean",
}


@dataclass
class ZPDEstimate:
    score:      float   # -1.0 (far above) to +1.0 (far below)
    position:   str     # ABOVE | IN | BELOW
    confidence: float   # 0.0 to 1.0
    signals:    dict    # raw signal values for logging


class ZPDEstimator:
    """
    Maintains a rolling window of turn signals.
    Estimates ZPD position from multi-signal fusion.
    """

    def __init__(self, window: int = 4):
        self._window = window
        self._history: deque = deque(maxlen=window)
        self._baseline_latency: Optional[float] = None

    def update(
            self,
            student_text:        str,
            response_latency_ms: float,
            error_type:          str,   # from error_classifier
            filler_count:        int,
            giving_up:           bool,
    ) -> ZPDEstimate:
        """
        Add one turn's data and return updated ZPD estimate.
        """
        # Compute this turn's sub-signals
        latency_signal  = self._score_latency(response_latency_ms)
        correction_signal = self._score_self_correction(student_text)
        hedging_signal  = self._score_hedging(student_text)
        error_signal    = self._score_error(error_type)

        turn_data = {
            "latency_signal":   latency_signal,
            "correction_signal": correction_signal,
            "hedging_signal":   hedging_signal,
            "error_signal":     error_signal,
            "giving_up":        giving_up,
            "filler_count":     filler_count,
        }
        self._history.append(turn_data)

        # Update baseline latency on first turn
        if self._baseline_latency is None:
            self._baseline_latency = response_latency_ms

        return self._compute_estimate()

    def _score_latency(self, latency_ms: float) -> float:
        """
        Slow responses → above ZPD (student struggling).
        Fast responses → below ZPD (too easy).
        Returns -1.0 (very slow) to +1.0 (very fast).
        Normalised against session baseline.
        """
        if self._baseline_latency and self._baseline_latency > 0:
            ratio = latency_ms / self._baseline_latency
        else:
            # Absolute scale: <1500ms fast, >4000ms slow
            ratio = latency_ms / 2000.0

        if   ratio < 0.6:  return +0.8   # much faster than baseline = too easy
        elif ratio < 0.9:  return +0.3
        elif ratio < 1.3:  return  0.0   # near baseline = optimal
        elif ratio < 2.0:  return -0.4   # slower = harder
        else:              return -0.8   # much slower = above ZPD

    def _score_self_correction(self, text: str) -> float:
        """
        Self-correction is a ZPD signal: student IS thinking hard.
        Positive (IN-ZPD) if self-corrections present.
        """
        text_lower = text.lower()
        found = sum(1 for m in SELF_CORRECTION_MARKERS if m in text_lower)
        # 0 corrections = unknown, 1-2 = in ZPD, 3+ = struggling
        if   found == 0: return  0.0
        elif found <= 2: return +0.3   # productive struggle
        else:            return -0.2   # too much backtracking

    def _score_hedging(self, text: str) -> float:
        """
        Hedging language → above ZPD (uncertainty).
        Returns 0 (no hedging) to -1.0 (heavy hedging).
        """
        text_lower = text.lower()
        found      = sum(1 for h in HEDGING_WORDS if h in text_lower)
        return float(-min(found * 0.25, 1.0))

    def _score_error(self, error_type: str) -> float:
        """
        Error type impact on ZPD score.
        OVERLOAD and CONCEPTUAL most strongly signal above-ZPD.
        """
        return {
            "NONE":            0.0,
            "CARELESS":       -0.1,   # mild signal
            "PROCEDURAL":     -0.3,
            "CONCEPTUAL":     -0.4,
            "OVERLOAD_INDUCED": -0.5,
        }.get(error_type, 0.0)

    def _compute_estimate(self) -> ZPDEstimate:
        if not self._history:
            return ZPDEstimate(
                score=0.0, position="IN",
                confidence=0.0, signals={}
            )

        # Average each signal across window
        avg = lambda key: sum(
            t.get(key, 0.0) for t in self._history
        ) / len(self._history)

        lat  = avg("latency_signal")
        corr = avg("correction_signal")
        hedg = avg("hedging_signal")
        err  = avg("error_signal")

        # Giving-up overrides everything
        giving_up_count = sum(
            1 for t in self._history if t.get("giving_up", False)
        )
        if giving_up_count >= 2:
            return ZPDEstimate(
                score=-0.90, position="ABOVE",
                confidence=0.95,
                signals={"override": "giving_up"}
            )

        score = (
            lat  * ZPD_LATENCY_WEIGHT    +
            corr * ZPD_CORRECTION_WEIGHT +
            hedg * ZPD_HEDGING_WEIGHT    +
            err  * ZPD_ERROR_WEIGHT
        )

        # Confidence increases with window fullness
        confidence = len(self._history) / self._window

        if   score < ZPD_ABOVE_THRESHOLD:  position = "ABOVE"
        elif score > ZPD_BELOW_THRESHOLD:  position = "BELOW"
        else:                              position = "IN"

        return ZPDEstimate(
            score=round(float(score), 3),
            position=position,
            confidence=round(confidence, 2),
            signals={
                "latency": round(lat, 2),
                "correction": round(corr, 2),
                "hedging": round(hedg, 2),
                "error": round(err, 2),
            }
        )

    def get_challenge_hint(self, estimate: ZPDEstimate) -> str:
        """Instruction string for prompt assembler."""
        hints = {
            "ABOVE": (
                "Student is above their ZPD — content is too hard right now. "
                "Reduce difficulty by one step. Break into smaller pieces. "
                "Do not introduce any new concepts this turn."
            ),
            "IN": (
                "Student is in their ZPD — optimal learning zone. "
                "Maintain current difficulty. Allow productive struggle."
            ),
            "BELOW": (
                "Content may be too easy. "
                "Consider increasing challenge slightly or moving to the next concept."
            ),
        }
        return hints.get(estimate.position, "")