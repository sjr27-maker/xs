# IPC vector -> tone instruction string
# style/ipc_adapter.py
"""
Converts IPC vector into a concrete tone instruction string.
Called by prompt/assembler.py — output goes directly into prompt.

IPC dimensions:
  dominance: 0=submissive/uncertain → 1=assertive/confident
  warmth:    0=cold/task-focused   → 1=warm/relationship-focused
  pace:      slow | medium | fast
  giving_up: bool — immediate override
"""
from dataclasses import dataclass


@dataclass
class IPCInstruction:
    tone_label:   str
    instruction:  str
    voice_hint:   str   # for Gemini voice tuning via system prompt


# Archetype base instructions
_ARCHETYPE_BASE = {
    "maya": (
        "Warm, gentle, patient. Student is soft-spoken and needs emotional safety. "
        "Be encouraging before correcting. Never blunt. Use 'we' language: "
        "'let's look at this together'."
    ),
    "arjun": (
        "Direct, confident, peer-like. Student is assertive and responds to challenge. "
        "Match their energy. Be concise. Skip excessive encouragement — "
        "they find it patronising. Get to the point."
    ),
    "lina": (
        "Warm but structured. Clear, calm, organised. "
        "Student appreciates clarity and logical flow. "
        "Encourage effort without being effusive."
    ),
}


def get_ipc_instruction(
        adapted_ipc: dict,
        session_anomaly: bool = False,
) -> IPCInstruction:
    """
    Convert adapted IPC vector into tone instruction.
    session_anomaly softens everything regardless of IPC.
    """
    dom       = adapted_ipc.get("dominance",   0.5)
    warm      = adapted_ipc.get("warmth",      0.6)
    pace      = adapted_ipc.get("pace",        "medium")
    giving_up = adapted_ipc.get("giving_up",   False)
    archetype = adapted_ipc.get("archetype",   "lina")
    fatigue   = adapted_ipc.get("fatigue_score", 0.0)

    # Immediate override — giving up
    if giving_up:
        return IPCInstruction(
            tone_label  = "disengaging",
            instruction = (
                "Student is giving up. Maximum warmth, minimum pressure. "
                "Acknowledge effort first. One very simple question only. "
                "Do not push for depth. Create one small win."
            ),
            voice_hint  = "Speak very gently and slowly.",
        )

    # Anomaly session softens everything
    if session_anomaly or fatigue > 0.70:
        return IPCInstruction(
            tone_label  = "fatigue_mode",
            instruction = (
                "Student is fatigued. Keep responses very short. "
                "Warm and low-pressure. No complex multi-step problems. "
                "Celebrate small wins. Let them set the pace."
            ),
            voice_hint  = "Speak warmly and unhurried.",
        )

    # Derive from IPC dimensions
    base = _ARCHETYPE_BASE.get(archetype, _ARCHETYPE_BASE["lina"])

    # Pace modifier
    if pace == "fast":
        pace_hint = "Student speaks fast — keep responses punchy and energetic."
    elif pace == "slow":
        pace_hint = "Student speaks slowly — be patient, give space, do not rush."
    else:
        pace_hint = ""

    # Dominance modifier
    if dom > 0.70:
        dom_hint = "Student is assertive. Challenge them slightly — they engage with difficulty."
    elif dom < 0.35:
        dom_hint = "Student is uncertain. Build confidence explicitly before pushing forward."
    else:
        dom_hint = ""

    # Warmth modifier
    if warm > 0.70:
        warm_hint = "Student is warm and relational. Personal connection matters to them."
    elif warm < 0.35:
        warm_hint = "Student is task-focused. Skip pleasantries, respect their time."
    else:
        warm_hint = ""

    modifiers = " ".join(h for h in [pace_hint, dom_hint, warm_hint] if h)

    instruction = f"{base} {modifiers}".strip()

    # Voice hint for Gemini
    if warm > 0.65 and dom < 0.50:
        voice_hint = "Speak warmly and gently."
    elif dom > 0.65:
        voice_hint = "Speak with confidence and energy."
    else:
        voice_hint = "Speak clearly and calmly."

    return IPCInstruction(
        tone_label  = archetype,
        instruction = instruction,
        voice_hint  = voice_hint,
    )