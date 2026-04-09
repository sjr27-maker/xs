# WM slots, closure detection, overload flag
# cognition/wm_monitor.py
"""
Models working memory state — what concepts are currently active.
Working memory capacity: 4±1 chunks (Miller's law, conservative=4).

Critical insight: when WM is full, new information pushes out old.
This is why students make OVERLOAD_INDUCED errors — they knew the
procedure but ran out of cognitive space to execute it.

SYRA's verbal closure pattern ("lock that in") offloads WM
before introducing new concepts. Master tutors do this constantly.
This file implements it systematically.
"""
from dataclasses import dataclass, field
from collections import deque
from typing import Optional
from config import WM_CAPACITY, WM_OVERLOAD_ERRORS


@dataclass
class WMState:
    slots_used:          int
    active_concepts:     list[str]
    is_full:             bool
    overload_detected:   bool
    closure_needed:      bool
    closure_target:      Optional[str]   # concept to close first


class WMMonitor:

    def __init__(self, capacity: int = WM_CAPACITY):
        self._capacity = capacity
        self._active: list[str] = []
        self._overload_error_count = 0
        self._closed_concepts: list[str] = []

    def introduce_concept(self, concept: str) -> tuple[bool, Optional[str]]:
        """
        Call before introducing a new concept.
        Returns (can_introduce, concept_to_close_first).
        If WM is full, returns (False, oldest_concept).
        """
        if concept in self._active:
            return True, None   # already active, no slot needed

        if len(self._active) >= self._capacity:
            # Must close something first
            to_close = self._active[0]   # close oldest
            return False, to_close

        return True, None

    def mark_introduced(self, concept: str):
        """Call after successfully introducing a concept."""
        if concept not in self._active:
            self._active.append(concept)
            # Keep bounded at capacity+1 for safety
            if len(self._active) > self._capacity + 1:
                self._active.pop(0)

    def mark_closed(self, concept: str):
        """
        Call after verbal closure ("lock that in").
        Frees the WM slot.
        """
        if concept in self._active:
            self._active.remove(concept)
            self._closed_concepts.append(concept)

    def report_overload_error(self):
        """
        Called by error_classifier when OVERLOAD_INDUCED error detected.
        If threshold exceeded, triggers mandatory WM reset.
        """
        self._overload_error_count += 1

    def get_state(self) -> WMState:
        slots    = len(self._active)
        is_full  = slots >= self._capacity
        overload = self._overload_error_count >= WM_OVERLOAD_ERRORS
        closure_needed = is_full or overload

        closure_target = None
        if closure_needed and self._active:
            closure_target = self._active[0]   # oldest active concept

        return WMState(
            slots_used        = slots,
            active_concepts   = list(self._active),
            is_full           = is_full,
            overload_detected = overload,
            closure_needed    = closure_needed,
            closure_target    = closure_target,
        )

    def reset_overload_count(self):
        """Call after WM closure sequence completes."""
        self._overload_error_count = 0

    def get_closure_statement(self, concept: str) -> str:
        """
        Generates the verbal closure instruction for the prompt.
        This is what master tutors say naturally.
        SYRA says it explicitly — no AI tutor does this.
        """
        return (
            f"Before continuing, close the '{concept}' concept verbally. "
            f"Say something like: 'Okay — so we've established [key point about {concept}]. "
            f"Lock that in.' Then pause briefly before the next concept."
        )

    def get_instruction(self) -> str:
        """Instruction string for prompt assembler."""
        state = self.get_state()

        if state.overload_detected:
            return (
                "WORKING MEMORY OVERLOAD DETECTED. "
                "Stop all new content immediately. "
                f"Run verbal closure on: {state.active_concepts}. "
                "Do not ask any new questions until closure is complete. "
                "Say: 'Let's consolidate what we have before we go further.'"
            )
        if state.closure_needed and state.closure_target:
            return self.get_closure_statement(state.closure_target)
        if state.is_full:
            return (
                f"Working memory is full ({state.slots_used}/{self._capacity} slots). "
                "Do not introduce any new concepts this turn. "
                "Consolidate existing concepts first."
            )
        return (
            f"Working memory: {state.slots_used}/{self._capacity} slots used. "
            f"Active concepts: {', '.join(state.active_concepts) or 'none'}."
        )