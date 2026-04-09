# current session turns + conversation history
# memory/session_memory.py
"""
Short-term session state.
Stores turns, conversation history, cognitive state per turn.
Provides context dict for prompt assembler.
"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


@dataclass
class Turn:
    turn_num:       int
    student_text:   str
    ai_response:    str
    ipc_vector:     dict
    acoustic:       dict
    error_type:     str
    affect_state:   str
    zpd_position:   str
    wm_slots:       int
    dependency_idx: float
    timestamp:      str = field(
        default_factory=lambda: datetime.now().isoformat()
    )


class SessionMemory:

    def __init__(self, profile: dict, subject: str):
        self.subject  = subject
        self.turns:   list[Turn] = []
        self._profile = profile
        self._start   = datetime.now()

        ctx = profile.get("session_history", [])
        last = ctx[-1] if ctx else None
        self.context = {
            "returning":   bool(ctx),
            "total_sessions": len(ctx),
            "last_session": last,
            "unresolved_misconceptions": [
                m for m in profile.get("knowledge", {})
                   .get("misconceptions", [])
                if isinstance(m, dict) and not m.get("resolved", False)
            ][:3],
        }
        # Conversation history for prompt assembler
        self.conversation_history: list[dict] = []

    def add_turn(
            self,
            student_text:   str,
            ai_response:    str,
            ipc_vector:     dict,
            acoustic:       dict,
            error_type:     str     = "NONE",
            affect_state:   str     = "neutral",
            zpd_position:   str     = "IN",
            wm_slots:       int     = 0,
            dependency_idx: float   = 0.0,
    ):
        turn = Turn(
            turn_num       = len(self.turns) + 1,
            student_text   = student_text,
            ai_response    = ai_response,
            ipc_vector     = ipc_vector,
            acoustic       = acoustic,
            error_type     = error_type,
            affect_state   = affect_state,
            zpd_position   = zpd_position,
            wm_slots       = wm_slots,
            dependency_idx = dependency_idx,
        )
        self.turns.append(turn)

        # Update conversation history (for prompt context)
        if student_text:
            self.conversation_history.append({
                "role": "student", "content": student_text
            })
        if ai_response:
            self.conversation_history.append({
                "role": "syra", "content": ai_response
            })
        # Keep bounded
        self.conversation_history = self.conversation_history[-16:]

    def to_log(self) -> dict:
        """Serialisable session log for JSON save."""
        return {
            "subject":    self.subject,
            "start_time": self._start.isoformat(),
            "end_time":   datetime.now().isoformat(),
            "turn_count": len(self.turns),
            "turns": [
                {
                    "turn_num":       t.turn_num,
                    "student_text":   t.student_text,
                    "ai_response":    t.ai_response,
                    "ipc_vector":     t.ipc_vector,
                    "error_type":     t.error_type,
                    "affect_state":   t.affect_state,
                    "zpd_position":   t.zpd_position,
                    "wm_slots":       t.wm_slots,
                    "dependency_idx": t.dependency_idx,
                    "timestamp":      t.timestamp,
                }
                for t in self.turns
            ],
        }