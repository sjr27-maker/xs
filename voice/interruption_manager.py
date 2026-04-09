# 4-gate barge-in + SYRA self-interrupt
# voice/interruption_manager.py
"""
Manages all interruption and silence logic for the live session.

Handles four distinct audio situations the current architecture missed:

1. BARGE-IN:          Student speaks while SYRA is responding
                      → Stop SYRA, listen to student
                      4-gate system: VAD + RMS + sustained frames + cooldown

2. SELF-TALK:         Student muttering to themselves while solving
                      → Do NOT interrupt. Listen passively.
                      Detected by: low volume, no rising intonation,
                      "thinking" words (okay so, wait, hmm, let me)

3. PRODUCTIVE SILENCE: Student is thinking — solving takes time
                      → Do NOT intervene for up to 90 seconds
                      Detected by: recent attempt in progress,
                      no confused markers in last turn

4. CONFUSED SILENCE:  Student is stuck and not producing anything
                      → Intervene with a calibrated hint at 30s
                      Detected by: long pause after hard question,
                      no self-talk detected, ZPD=ABOVE

New from Session 2 observations:
  - Student took 3-5 minutes on LCM — productive silence, not confusion
  - Student mutters while solving — self-talk, not directed at SYRA
  - "I give up" mid-problem = frustration, not termination
  - Genuine termination = "ok bye", "stop session", "I'm done"
"""
import asyncio
import time
import logging
import numpy as np
from dataclasses import dataclass
from typing import Optional
from config import (
    BARGE_IN_RMS, BARGE_IN_SPEECH_GATE,
    BARGE_IN_COOLDOWN, IN_RATE,
)

logger = logging.getLogger("SYRA.InterruptionManager")

# Self-talk detection
SELF_TALK_MARKERS = {
    "okay so", "okay let me", "wait", "hmm", "let me see",
    "so if", "then that means", "so that gives", "um okay",
    "right so", "okay okay", "let me think", "hang on",
    "so the", "and then", "so first", "now if",
}

# Genuine termination signals (mid-problem "give up" is NOT in this list)
TERMINATION_SIGNALS = {
    "bye syra", "goodbye syra", "ok bye", "okay bye",
    "stop session", "end session", "that's it for today",
    "i'm done for today", "let's stop", "stop here",
}

# Frustration expressions — push through, do not terminate
FRUSTRATION_EXPRESSIONS = {
    "i give up", "this is too hard", "i can't do this",
    "i don't know anything", "forget it", "i quit",
    "i hate this", "this doesn't make sense", "i'm so bad at this",
}

# Silence thresholds
PRODUCTIVE_SILENCE_LIMIT   = 90.0   # seconds before intervening on productive silence
CONFUSED_SILENCE_LIMIT     = 30.0   # seconds before intervening on confused silence
SELF_TALK_RMS_CEILING      = 0.05   # quieter than normal speech
MIN_DIRECTED_SPEECH_LENGTH = 4      # words — shorter is likely self-talk
DISTRACTION_RMS_SPIKE      = 3.0    # ratio of sudden background RMS increase


@dataclass
class AudioSituation:
    """Classified audio state for one frame window."""
    situation_type:  str   # BARGE_IN | SELF_TALK | SILENCE | DIRECTED_SPEECH
    confidence:      float
    rms:             float
    text_hint:       Optional[str] = None   # partial transcript if available


@dataclass
class SilenceState:
    """Tracks ongoing silence period."""
    started_at:          float
    duration:            float
    prior_turn_had_error: bool
    zpd_position:        str
    last_student_text:   str

    @property
    def intervention_type(self) -> str:
        """What kind of silence is this likely to be?"""
        # If student was in ZPD and last turn was an attempt, give space
        if self.zpd_position == "IN" and not self.prior_turn_had_error:
            return "PRODUCTIVE"
        # If ZPD is ABOVE, likely confused
        if self.zpd_position == "ABOVE":
            return "CONFUSED"
        # Check for self-talk markers in last student speech
        last_lower = self.last_student_text.lower()
        if any(m in last_lower for m in ["let me", "okay so", "wait"]):
            return "PRODUCTIVE"
        # Default: confused if long enough
        if self.duration > 45:
            return "CONFUSED"
        return "PRODUCTIVE"

    @property
    def should_intervene(self) -> bool:
        itype = self.intervention_type
        if itype == "CONFUSED":
            return self.duration >= CONFUSED_SILENCE_LIMIT
        return self.duration >= PRODUCTIVE_SILENCE_LIMIT


class InterruptionManager:
    """
    Manages all interruption decisions for the live session.
    Called frame-by-frame in the send loop.
    """

    def __init__(self):
        self._last_barge_time     = 0.0
        self._consec_speech       = 0
        self._silence_start: Optional[float] = None
        self._baseline_rms        = None
        self._rms_history         = []
        self._is_ai_speaking      = False
        self._last_student_text   = ""
        self._zpd_position        = "IN"
        self._prior_had_error     = False

    def update_context(
            self,
            zpd_position:        str,
            prior_had_error:     bool,
            last_student_text:   str,
    ):
        """Called after each turn to update context for silence classification."""
        self._zpd_position      = zpd_position
        self._prior_had_error   = prior_had_error
        self._last_student_text = last_student_text

    def set_ai_speaking(self, speaking: bool):
        self._is_ai_speaking = speaking

    def classify_audio_frame(
            self,
            chunk:     np.ndarray,
            has_vad:   bool,
            rms:       float,
            text_so_far: str = "",
    ) -> AudioSituation:
        """
        Classify a single audio frame into a situation type.
        Called every FRAME_SAMPLES (20ms).
        """
        now = time.time()

        # Update RMS history for distraction detection
        self._rms_history.append(rms)
        if len(self._rms_history) > 50:
            self._rms_history.pop(0)

        # Update baseline on first few frames
        if self._baseline_rms is None and len(self._rms_history) >= 20:
            self._baseline_rms = sum(self._rms_history) / len(self._rms_history)

        # ── Barge-in check ────────────────────────────────────────────
        if has_vad:
            self._consec_speech += 1
            self._silence_start  = None   # reset silence timer
        else:
            self._consec_speech = 0

        if (
            self._is_ai_speaking
            and has_vad
            and rms > BARGE_IN_RMS
            and self._consec_speech >= BARGE_IN_SPEECH_GATE
            and now - self._last_barge_time > BARGE_IN_COOLDOWN
        ):
            return AudioSituation(
                situation_type="BARGE_IN",
                confidence=0.90,
                rms=rms,
            )

        # ── Self-talk check ───────────────────────────────────────────
        if has_vad and not self._is_ai_speaking:
            is_self_talk = self._detect_self_talk(rms, text_so_far)
            if is_self_talk:
                return AudioSituation(
                    situation_type="SELF_TALK",
                    confidence=0.75,
                    rms=rms,
                    text_hint=text_so_far[:60] if text_so_far else None,
                )

        # ── Silence check ─────────────────────────────────────────────
        if not has_vad and not self._is_ai_speaking:
            if self._silence_start is None:
                self._silence_start = now
            return AudioSituation(
                situation_type="SILENCE",
                confidence=0.90,
                rms=rms,
            )

        # ── Directed speech ───────────────────────────────────────────
        if has_vad and not self._is_ai_speaking:
            return AudioSituation(
                situation_type="DIRECTED_SPEECH",
                confidence=0.80,
                rms=rms,
                text_hint=text_so_far[:60] if text_so_far else None,
            )

        return AudioSituation(
            situation_type="SILENCE",
            confidence=0.50,
            rms=rms,
        )

    def _detect_self_talk(self, rms: float, text: str) -> bool:
        """
        Detect if the student is talking to themselves, not SYRA.

        Signals:
          1. Low volume (below normal speech threshold)
          2. Contains thinking/working markers
          3. Short utterance without question intonation
          4. No SYRA-directed words
        """
        # Volume gate — self-talk is quieter
        if rms > BARGE_IN_RMS * 1.5:
            return False   # too loud to be self-talk

        if not text:
            # Low-volume audio with no clear text — likely self-talk
            return rms < SELF_TALK_RMS_CEILING

        text_lower = text.lower().strip()

        # Explicit self-talk markers
        has_thinking_marker = any(m in text_lower for m in SELF_TALK_MARKERS)
        if has_thinking_marker:
            return True

        # Very short utterance = likely not directed
        word_count = len(text_lower.split())
        if word_count < MIN_DIRECTED_SPEECH_LENGTH and rms < BARGE_IN_RMS:
            return True

        # Contains SYRA's name = directed
        if "syra" in text_lower:
            return False

        return False

    def get_silence_state(
            self,
            zpd_position:      str,
            prior_had_error:   bool,
            last_student_text: str,
    ) -> Optional[SilenceState]:
        """Returns current silence state if in silence, else None."""
        if self._silence_start is None:
            return None
        return SilenceState(
            started_at=self._silence_start,
            duration=time.time() - self._silence_start,
            prior_turn_had_error=prior_had_error,
            zpd_position=zpd_position,
            last_student_text=last_student_text,
        )

    def get_silence_intervention(
            self,
            silence_state: SilenceState,
    ) -> Optional[str]:
        """
        Returns intervention text to inject if silence is too long.
        Returns None if we should keep waiting.
        """
        if not silence_state.should_intervene:
            return None

        if silence_state.intervention_type == "CONFUSED":
            return (
                "[SILENCE INTERVENTION: Student has been silent and appears stuck. "
                "Give ONE small hint — not the answer. Something to restart thinking. "
                "Keep it very short. Warm tone.]"
            )
        else:
            # Productive silence — gentle check-in only
            return (
                "[SILENCE INTERVENTION: Student has been thinking for a long time. "
                "Gently check: 'Take your time — want a small nudge or are you still working it?' "
                "Do NOT give the answer or a hint unless they ask.]"
            )

    def classify_text_intent(self, text: str) -> str:
        """
        Classify student speech intent.
        Returns: TERMINATION | FRUSTRATION | DIRECTED | SELF_TALK
        """
        text_lower = text.lower().strip()

        # Check genuine termination first
        if any(s in text_lower for s in TERMINATION_SIGNALS):
            return "TERMINATION"

        # Check frustration expression (mid-problem give-up)
        if any(s in text_lower for s in FRUSTRATION_EXPRESSIONS):
            return "FRUSTRATION"

        # Check self-talk markers
        if any(m in text_lower for m in SELF_TALK_MARKERS):
            word_count = len(text_lower.split())
            if word_count < 8:
                return "SELF_TALK"

        return "DIRECTED"

    def get_push_through_instruction(self, frustration_text: str) -> str:
        """
        When student expresses "I give up" mid-problem,
        SYRA should push through, not terminate.
        Returns instruction for the prompt assembler.
        """
        return (
            "IMPORTANT: Student said something like 'I give up' but this is "
            "a frustration expression mid-problem, NOT a session termination signal. "
            "Do NOT end the session. Do NOT give the full answer. "
            "Acknowledge the frustration warmly with ONE sentence. "
            "Then create the smallest possible win — "
            "guide them to the very next micro-step only. "
            "If they genuinely want to end, they will say 'bye' or 'stop'."
        )

    def check_distraction(self) -> bool:
        """
        Detect sudden background noise spike — likely distraction.
        TV, someone calling the student, external sounds.
        Returns True if distraction likely.
        """
        if not self._baseline_rms or len(self._rms_history) < 10:
            return False

        recent_avg = sum(self._rms_history[-5:]) / 5
        if recent_avg > self._baseline_rms * DISTRACTION_RMS_SPIKE:
            logger.info(
                f"Distraction detected: RMS spike "
                f"{recent_avg:.3f} vs baseline {self._baseline_rms:.3f}"
            )
            return True
        return False

    def update_barge_time(self):
        """Call after confirmed barge-in."""
        self._last_barge_time = time.time()
        self._consec_speech   = 0
        self._silence_start   = None