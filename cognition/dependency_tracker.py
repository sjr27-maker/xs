# word ratio, alarm at 0.7 x 3 turns
# cognition/dependency_tracker.py
"""
Tracks student dependency on SYRA across turns.
Dependency index = student words / (student words + SYRA words) per turn.

Low index (0.0–0.3):   Student speaking a lot — healthy independence.
Medium (0.3–0.5):      Normal tutoring exchange.
High (0.5–0.7):        Student leaning on SYRA — watch closely.
Alarm (>0.7 × 3 turns): Student fully passive — switch to Socratic mode.

What Niraj showed at Q5: dependency_index=0.78, 2 consecutive turns.
System should have caught it at Q4 when it crossed 0.60.
"""
from dataclasses import dataclass
from collections import deque
from config import DEPENDENCY_ALARM, DEPENDENCY_TURNS


@dataclass
class DependencyState:
    index:           float   # 0.0 (independent) to 1.0 (fully dependent)
    level:           str     # LOW | MEDIUM | HIGH | ALARM
    alarm_triggered: bool
    turns_at_alarm:  int
    trend:           str     # stable | rising | falling


class DependencyTracker:

    def __init__(self):
        self._history:     deque = deque(maxlen=8)
        self._alarm_streak: int  = 0

    def update(
            self,
            student_word_count: int,
            syra_word_count:    int,
    ) -> DependencyState:
        """
        Call after each complete exchange (student spoke + SYRA responded).
        student_word_count: words in student's turn
        syra_word_count: words in SYRA's response
        """
        total = student_word_count + syra_word_count
        if total == 0:
            index = 0.5
        else:
            # Index: proportion of words coming from SYRA
            # High SYRA proportion = high dependency
            index = syra_word_count / total

        self._history.append(index)

        # Alarm streak
        if index >= DEPENDENCY_ALARM:
            self._alarm_streak += 1
        else:
            self._alarm_streak = 0

        # Trend
        history = list(self._history)
        if len(history) >= 3:
            recent = history[-3:]
            older  = history[:-3] if len(history) > 3 else [history[0]]
            trend_val = sum(recent) / 3 - sum(older) / len(older)
            if   trend_val >  0.08: trend = "rising"
            elif trend_val < -0.08: trend = "falling"
            else:                   trend = "stable"
        else:
            trend = "stable"

        # Level classification
        avg_index = sum(history) / len(history)
        if   avg_index < 0.30: level = "LOW"
        elif avg_index < 0.50: level = "MEDIUM"
        elif avg_index < 0.70: level = "HIGH"
        else:                  level = "ALARM"

        alarm_triggered = self._alarm_streak >= DEPENDENCY_TURNS

        return DependencyState(
            index=round(index, 3),
            level=level,
            alarm_triggered=alarm_triggered,
            turns_at_alarm=self._alarm_streak,
            trend=trend,
        )

    def get_instruction(self, state: DependencyState) -> str:
        """Instruction string for prompt assembler."""
        if state.alarm_triggered:
            return (
                "DEPENDENCY ALARM: Student has been passive for "
                f"{state.turns_at_alarm} consecutive turns. "
                "Switch to PURE SOCRATIC MODE immediately. "
                "Ask one guiding question. Do NOT give any answers, "
                "steps, or hints beyond the question itself. "
                "Force re-engagement before continuing."
            )
        if state.level == "HIGH":
            return (
                "Dependency rising — student becoming passive. "
                "Ask a question that requires them to produce an answer, "
                "not just confirm yours. Reduce SYRA response length."
            )
        if state.trend == "rising" and state.level != "LOW":
            return (
                "Dependency trending upward. "
                "Shorten SYRA responses. Ask more, tell less."
            )
        return ""