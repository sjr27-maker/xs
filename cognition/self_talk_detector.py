# cognition/self_talk_detector.py
"""
Detects student self-talk during problem solving.

Self-talk while solving is a POSITIVE metacognitive signal.
Piaget called it 'egocentric speech'. Vygotsky called it
'inner speech externalised'. Both agree: it means active thinking.

SYRA should:
  - NOT interrupt self-talk
  - Extract cognitive signals from it passively
  - Use it to estimate ZPD and confusion state more accurately

From Session 2: Niraj spoke to himself while doing LCM.
The question was: "How will our system understand that?"
Answer: volume + content + context pattern.
"""
import re
from dataclasses import dataclass
from typing import Optional

SELF_TALK_OPENERS = {
    "okay so", "okay let me", "so if", "wait wait", "hmm",
    "let me see", "right so", "so then", "and then if",
    "so the answer", "if i", "that gives me", "so that's",
    "okay okay", "so first", "now", "let me try",
    "so multiply", "so the lcm", "so factors of",
}

DIRECTED_MARKERS = {
    "syra", "can you", "help me", "what is", "how do i",
    "i don't understand", "what should", "tell me",
    "explain", "is this right", "did i",
}


@dataclass
class SpeechClassification:
    speech_type:      str   # DIRECTED | SELF_TALK | AMBIGUOUS
    confidence:       float
    cognitive_signal: str   # THINKING | CONFUSED | ATTEMPTING | UNKNOWN
    word_count:       int
    contains_maths:   bool


def classify_speech(
        text:           str,
        rms:            float,
        baseline_rms:   Optional[float] = None,
) -> SpeechClassification:
    """
    Classify whether student speech is directed at SYRA
    or is self-talk during problem solving.
    """
    if not text or not text.strip():
        return SpeechClassification(
            speech_type="AMBIGUOUS", confidence=0.5,
            cognitive_signal="UNKNOWN", word_count=0,
            contains_maths=False,
        )

    text_lower  = text.lower().strip()
    words       = text_lower.split()
    word_count  = len(words)

    # Volume signal — self-talk is quieter
    if baseline_rms and baseline_rms > 0:
        volume_ratio = rms / baseline_rms
    else:
        volume_ratio = 1.0

    is_quiet = volume_ratio < 0.55

    # Content signals
    has_directed_marker  = any(m in text_lower for m in DIRECTED_MARKERS)
    has_selftalk_opener  = any(text_lower.startswith(m) or m in text_lower[:30]
                               for m in SELF_TALK_OPENERS)
    has_maths            = bool(re.search(
        r'\d+|plus|minus|times|divide|lcm|hcf|factor|split|multiply',
        text_lower
    ))
    ends_with_question   = text.strip().endswith("?")

    # Decision logic
    if has_directed_marker:
        speech_type = "DIRECTED"
        confidence  = 0.90
    elif has_selftalk_opener and (is_quiet or word_count < 8):
        speech_type = "SELF_TALK"
        confidence  = 0.80
    elif is_quiet and has_maths and not ends_with_question:
        # Quiet, maths-related, not a question — almost certainly self-talk
        speech_type = "SELF_TALK"
        confidence  = 0.75
    elif ends_with_question:
        speech_type = "DIRECTED"
        confidence  = 0.70
    elif word_count < 5 and is_quiet:
        speech_type = "SELF_TALK"
        confidence  = 0.65
    else:
        speech_type = "AMBIGUOUS"
        confidence  = 0.50

    # Cognitive signal from self-talk content
    cognitive_signal = "UNKNOWN"
    if speech_type == "SELF_TALK":
        if any(w in text_lower for w in ["wait", "no", "that's wrong", "mistake"]):
            cognitive_signal = "SELF_CORRECTING"   # strong positive
        elif any(w in text_lower for w in ["so", "that gives", "therefore", "so that"]):
            cognitive_signal = "REASONING"          # positive
        elif any(w in text_lower for w in ["i don't", "confused", "no idea", "what"]):
            cognitive_signal = "CONFUSED"
        else:
            cognitive_signal = "ATTEMPTING"

    return SpeechClassification(
        speech_type=speech_type,
        confidence=confidence,
        cognitive_signal=cognitive_signal,
        word_count=word_count,
        contains_maths=has_maths,
    )


def get_selftalk_instruction() -> str:
    """
    Instruction for prompt assembler when self-talk detected.
    SYRA should not interrupt but can monitor.
    """
    return (
        "Student is currently talking to themselves while working through "
        "the problem. This is a positive sign — they are actively reasoning. "
        "Do NOT interrupt. Do NOT respond yet. "
        "Wait for them to address you directly or finish their attempt."
    )