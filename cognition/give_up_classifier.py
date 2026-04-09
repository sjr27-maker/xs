# cognition/give_up_classifier.py
"""
Distinguishes genuine session termination from frustration expression.

From Session 2 observation:
"If the student says he wants to give up, he will not actually mean it
sometimes. The system must recognize and push through. If he actually
wants to end, he will say 'ok bye or something'."

Cognitive science basis:
Frustration tolerance varies. Mid-problem "I give up" is almost always
a venting behaviour — the student needs validation and a micro-win,
not termination. True disengagement follows a trajectory: multiple
frustration expressions + dependency alarm + giving_up IPC signal.

Genuine termination is always between problems, never during one.
"""
from dataclasses import dataclass

# Explicit termination — only these end the session
TERMINATION_PHRASES = {
    "bye syra", "goodbye syra", "ok bye", "okay bye",
    "bye bye", "see you", "good night syra",
    "stop session", "end session", "stop here",
    "that's it for today", "i'm done for today",
    "let's stop here", "we can stop", "enough for today",
    "i'll continue tomorrow",
}

# Frustration expressions — push through
FRUSTRATION_PHRASES = {
    "i give up", "i cant do this", "i can't do this",
    "this is too hard", "i don't get anything",
    "forget it", "whatever", "i quit",
    "this doesn't make sense", "i'm terrible at this",
    "i'll never understand", "i'm so bad",
    "just tell me the answer", "i don't care anymore",
}


@dataclass
class GiveUpClassification:
    intent:      str    # TERMINATION | FRUSTRATION | AMBIGUOUS
    confidence:  float
    push_through: bool  # True = SYRA should push through
    instruction: str    # for prompt assembler


def classify_give_up(
        text:              str,
        is_mid_problem:    bool,
        zpd_position:      str,
        turns_in_session:  int,
        giving_up_ipc:     bool,
) -> GiveUpClassification:
    """
    Classify whether student's give-up signal is genuine or frustration.

    is_mid_problem: True if student is currently working on a problem
    zpd_position: from zpd_estimator — ABOVE = more likely genuine give-up
    turns_in_session: more turns = more investment = frustration more likely
    giving_up_ipc: acoustic signal from ipc_classifier
    """
    text_lower = text.lower().strip()

    # Explicit termination phrase — always respect
    if any(phrase in text_lower for phrase in TERMINATION_PHRASES):
        return GiveUpClassification(
            intent="TERMINATION",
            confidence=0.95,
            push_through=False,
            instruction="Student genuinely wants to end the session. "
                        "Wrap up warmly. Brief summary of what they achieved. "
                        "No pressure. 'See you next time.'",
        )

    # Frustration phrase
    is_frustration = any(phrase in text_lower for phrase in FRUSTRATION_PHRASES)
    if not is_frustration:
        return GiveUpClassification(
            intent="AMBIGUOUS",
            confidence=0.50,
            push_through=False,
            instruction="",
        )

    # It's a frustration phrase — determine if genuine or venting
    # Mid-problem "give up" is almost always venting
    if is_mid_problem:
        return GiveUpClassification(
            intent="FRUSTRATION",
            confidence=0.88,
            push_through=True,
            instruction=(
                "Student said 'I give up' or similar MID-PROBLEM. "
                "This is frustration expression, NOT termination. "
                "Do NOT end the session. "
                "Response: ONE warm acknowledgment sentence. "
                "Then: the smallest possible next micro-step. "
                "Example: 'Totally fair — this one is tricky. "
                "What's 8 times 21? Just that one thing.' "
                "If they truly want to stop, they will say 'bye' or 'stop'."
            ),
        )

    # Between problems — check context
    # Many turns in + high ZPD + acoustic giving_up = more likely genuine
    genuine_score = 0.0
    if turns_in_session > 8:
        genuine_score += 0.2   # invested, frustration more expected
    if zpd_position == "ABOVE":
        genuine_score += 0.25  # genuinely struggling
    if giving_up_ipc:
        genuine_score += 0.30  # acoustically confirmed
    if turns_in_session < 3:
        genuine_score += 0.15  # too early = more likely genuine today

    if genuine_score > 0.55:
        return GiveUpClassification(
            intent="FRUSTRATION",
            confidence=genuine_score,
            push_through=True,
            instruction=(
                "Student expressing give-up frustration between problems. "
                "Context suggests they are genuinely struggling today. "
                "Options: (1) Acknowledge and offer a much easier warm-down problem, "
                "(2) Acknowledge and offer to end if they've covered enough. "
                "'That was a tough one. Want to try one quick easy one to "
                "finish on a good note, or call it here?' — let them choose."
            ),
        )

    # Default: push through with acknowledgment
    return GiveUpClassification(
        intent="FRUSTRATION",
        confidence=0.70,
        push_through=True,
        instruction=(
            "Student expressing frustration. Push through gently. "
            "One warm sentence, then continue with a smaller step."
        ),
    )