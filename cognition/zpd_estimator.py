# cognition/zpd_estimator.py
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
    "i'm not sure", "might be", "could be",
}

SELF_CORRECTION_MARKERS = {
    "wait", "actually", "no wait", "let me check",
    "hold on", "let me redo", "i made a mistake",
    "correction", "sorry", "i mean", "wait wait",
}


@dataclass
class ZPDEstimate:
    score:      float
    position:   str     # ABOVE | IN | BELOW
    confidence: float
    signals:    dict


class ZPDEstimator:

    def __init__(self, window: int = 4):
        self._window  = window
        self._history: deque = deque(maxlen=window)
        self._baseline_latency: Optional[float] = None
        self._latency_samples: list = []

    def update(
            self,
            student_text:        str,
            response_latency_ms: float,
            error_type:          str,
            filler_count:        int,
            giving_up:           bool,
    ) -> ZPDEstimate:

        latency_signal    = self._score_latency(response_latency_ms)
        correction_signal = self._score_self_correction(student_text)
        hedging_signal    = self._score_hedging(student_text)
        error_signal      = self._score_error(error_type)

        # Giving up is a strong ABOVE-ZPD signal — override score
        if giving_up:
            self._history.append({
                "latency_signal":    -0.80,
                "correction_signal": -0.20,
                "hedging_signal":    -0.40,
                "error_signal":      -0.40,
                "giving_up":         True,
                "filler_count":      filler_count,
            })
            return ZPDEstimate(
                score=-0.90,
                position="ABOVE",
                confidence=0.95,
                signals={"override": "giving_up"},
            )

        # High filler count contributes to above-ZPD
        filler_penalty = -min(filler_count * 0.04, 0.20)

        turn_data = {
            "latency_signal":    latency_signal,
            "correction_signal": correction_signal,
            "hedging_signal":    hedging_signal,
            "error_signal":      error_signal + filler_penalty,
            "giving_up":         False,
            "filler_count":      filler_count,
        }
        self._history.append(turn_data)

        # Update baseline latency (first 2 non-zero samples)
        if response_latency_ms > 0:
            self._latency_samples.append(response_latency_ms)
            if len(self._latency_samples) == 2 and self._baseline_latency is None:
                self._baseline_latency = sum(self._latency_samples) / 2

        return self._compute_estimate()

    def _score_latency(self, latency_ms: float) -> float:
        """
        Slow response → struggling → ABOVE ZPD (negative score).
        Fast response → too easy → BELOW ZPD (positive score).
        Zero latency (simulated/missing) → neutral.
        """
        if latency_ms <= 0:
            return 0.0   # no signal — do not penalise

        if self._baseline_latency and self._baseline_latency > 0:
            ratio = latency_ms / self._baseline_latency
        else:
            # Absolute scale: 1500ms = neutral
            ratio = latency_ms / 1500.0

        if   ratio < 0.60: return +0.80
        elif ratio < 0.90: return +0.30
        elif ratio < 1.30: return  0.00
        elif ratio < 2.00: return -0.40
        elif ratio < 3.00: return -0.65
        else:              return -0.80

    def _score_self_correction(self, text: str) -> float:
        text_lower = text.lower()
        found = sum(1 for m in SELF_CORRECTION_MARKERS if m in text_lower)
        if   found == 0: return  0.0
        elif found <= 2: return +0.3
        else:            return -0.2

    def _score_hedging(self, text: str) -> float:
        text_lower = text.lower()
        found = sum(1 for h in HEDGING_WORDS if h in text_lower)
        return float(-min(found * 0.20, 0.80))

    def _score_error(self, error_type: str) -> float:
        return {
            "NONE":             0.0,
            "CARELESS":        -0.15,
            "PROCEDURAL":      -0.35,
            "CONCEPTUAL":      -0.45,
            "OVERLOAD_INDUCED": -0.55,
        }.get(error_type, 0.0)

    def _compute_estimate(self) -> ZPDEstimate:
        if not self._history:
            return ZPDEstimate(
                score=0.0, position="IN",
                confidence=0.0, signals={}
            )

        def avg(key: str) -> float:
            vals = [t.get(key, 0.0) for t in self._history]
            return sum(vals) / len(vals)

        lat  = avg("latency_signal")
        corr = avg("correction_signal")
        hedg = avg("hedging_signal")
        err  = avg("error_signal")

        score = (
            lat  * ZPD_LATENCY_WEIGHT    +
            corr * ZPD_CORRECTION_WEIGHT +
            hedg * ZPD_HEDGING_WEIGHT    +
            err  * ZPD_ERROR_WEIGHT
        )

        confidence = len(self._history) / self._window

        if   score < ZPD_ABOVE_THRESHOLD: position = "ABOVE"
        elif score > ZPD_BELOW_THRESHOLD: position = "BELOW"
        else:                             position = "IN"

        return ZPDEstimate(
            score=round(float(score), 3),
            position=position,
            confidence=round(confidence, 2),
            signals={
                "latency":    round(lat,  3),
                "correction": round(corr, 3),
                "hedging":    round(hedg, 3),
                "error":      round(err,  3),
            },
        )

    def get_challenge_hint(self, estimate: ZPDEstimate) -> str:
        return {
            "ABOVE": (
                "Student is above their ZPD — content too hard. "
                "Reduce difficulty. Break into smaller pieces. "
                "Do not introduce any new concepts this turn."
            ),
            "IN":    "Student is in optimal ZPD. Maintain difficulty.",
            "BELOW": (
                "Content may be too easy. "
                "Consider increasing challenge slightly."
            ),
        }.get(estimate.position, "")