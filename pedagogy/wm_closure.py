# generates 'lock that in' statements
# pedagogy/wm_closure.py
"""
Generates verbal closure statements for working memory offload.
Called by wm_monitor when WM is full or overload is detected.

Master human tutors do this constantly and naturally.
No AI tutor does it systematically. This is a genuine differentiator.

Pattern: "Okay — so we've established [X]. Lock that in."
Then SYRA pauses briefly before continuing to new content.
"""
from typing import Optional


# Closure templates — varied so they don't sound repetitive
_CLOSURE_TEMPLATES = [
    "Before we go further — let's lock in what we have. We've established that {concept_summary}. Got it?",
    "Pause. We've confirmed: {concept_summary}. Keep that as our anchor and let's build on it.",
    "Good — so the key thing we've nailed is: {concept_summary}. That's solid. Now from there...",
    "Let's checkpoint. You've worked out that {concept_summary}. That's the foundation. Everything else builds on that.",
    "Hold that — {concept_summary} — that's yours now. With that locked in, let's take the next step.",
]

_template_index = 0


def get_closure_instruction(
        concepts:         list[str],
        concept_snapshot: str = "",
) -> str:
    """
    Returns the verbal closure instruction for prompt assembler.

    concepts:         list of concept names to close
    concept_snapshot: brief description of what was established
    """
    global _template_index

    if not concepts:
        return (
            "Before continuing, summarise what the student has understood so far. "
            "Help them articulate it in their own words. "
            "Then pause before the next concept."
        )

    concept_str = " and ".join(concepts[:2])

    if concept_snapshot:
        summary = concept_snapshot
    else:
        summary = f"the key ideas about {concept_str}"

    template = _CLOSURE_TEMPLATES[_template_index % len(_CLOSURE_TEMPLATES)]
    _template_index += 1

    closure_phrase = template.format(concept_summary=summary)

    return (
        f"Run verbal closure now before any new content. "
        f"Say something like: '{closure_phrase}' "
        f"Then wait for the student to confirm before continuing. "
        f"Do not introduce anything new until they have confirmed."
    )


def get_wm_status_for_prompt(
        slots_used:      int,
        capacity:        int,
        active_concepts: list[str],
) -> str:
    """
    Short WM status string for prompt assembler — always included.
    """
    remaining = capacity - slots_used
    if remaining == 0:
        return (
            f"WM FULL ({slots_used}/{capacity} slots). "
            f"Active: {', '.join(active_concepts)}. "
            "No new concepts until one is closed."
        )
    elif remaining == 1:
        return (
            f"WM nearly full ({slots_used}/{capacity}). "
            f"One slot remaining. Use it carefully."
        )
    return f"WM: {slots_used}/{capacity} slots used."