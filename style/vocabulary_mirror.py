# slang extraction, frequency gate, mirror instruction
# style/vocabulary_mirror.py
"""
Tracks student vocabulary, slang, formality, and humor across sessions.
Builds a style profile that SYRA uses to mirror the student's register.

Core principle: only mirror CONFIRMED patterns.
  - Slang confirmed after 3 appearances across sessions
  - Humor mirroring only after session 2 and 4+ humor events
  - Formality register updated via rolling window
  - Never mirror frustration, negativity, or profanity

Runs as end-of-session batch (more accurate than per-turn).
"""
import os
import json
import logging
from typing import Optional
from google import genai
from google.genai import types
from config import EXTRACT_MODEL

logger  = logging.getLogger("SYRA.VocabMirror")
_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# Whitelist of safe informal markers — never mirrors anything outside this
SAFE_INFORMAL = {
    "bro", "yaar", "dude", "man", "buddy", "mate", "boss",
    "da", "bhai", "legend", "solid", "clean", "real talk",
    "fire", "lowkey", "highkey", "vibe", "goat", "based",
    "no cap", "fr", "bet", "lit", "iconic", "valid", "fam",
    "ngl", "tbh", "slay", "bruh", "aye",
}

EXTRACT_PROMPT = """Analyze the student's speech across this session.

Turns:
{turns}

Return ONLY valid JSON:
{{
  "vocabulary_level": "simple|casual|academic|mixed",
  "confirmed_slang": [],
  "humor_style": "none|playful|sarcastic|self_deprecating",
  "humor_count": 0,
  "formality": "very_informal|informal|neutral",
  "sentence_length": "short|medium|long",
  "enthusiasm_markers": [],
  "example_phrases": [],
  "energy_level": "low|medium|high"
}}

Rules:
- confirmed_slang: only words used MORE THAN ONCE in the session
- humor_count: how many clear humor attempts the student made
- example_phrases: 2-3 phrases that capture HOW this student talks
- enthusiasm_markers: informal words like "bro", "yaar", "da", "fire" actually used
- Be conservative — only include what was clearly present"""


def extract_session_style(turns: list) -> dict:
    """
    End-of-session batch style extraction.
    More accurate than per-turn because it sees full context.
    Returns style dict for update_style_profile().
    """
    if len(turns) < 3:
        return {}

    # Build transcript from turns
    lines = []
    for t in turns[-12:]:
        text = ""
        if hasattr(t, "student_text"):
            text = t.student_text
        elif isinstance(t, dict):
            text = t.get("student_text", "")
        if text:
            lines.append(f"Student: {text}")

    if not lines:
        return {}

    transcript = "\n".join(lines)

    try:
        resp = _client.models.generate_content(
            model=EXTRACT_MODEL,
            contents=EXTRACT_PROMPT.format(turns=transcript),
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.1,
                thinking_config=types.ThinkingConfig(thinking_level="minimal"),
                max_output_tokens=400,
            ),
        )
        data = json.loads(resp.text)

        # Filter slang through whitelist
        raw_slang = data.get("confirmed_slang", [])
        safe_slang = [
            w.lower().strip() for w in raw_slang
            if w.lower().strip() in SAFE_INFORMAL
        ]
        data["confirmed_slang"] = safe_slang

        # Filter enthusiasm markers through whitelist
        raw_markers = data.get("enthusiasm_markers", [])
        data["enthusiasm_markers"] = [
            m.lower().strip() for m in raw_markers
            if m.lower().strip() in SAFE_INFORMAL
        ]

        return data

    except Exception as e:
        logger.debug(f"Style extraction failed: {e}")
        return {}


def update_style_profile(
        existing:    dict,
        new_session: dict,
) -> dict:
    """
    Merge new session style data into existing profile.
    Frequency-gated: slang confirmed after 2+ sessions.
    """
    if not new_session:
        return existing

    profile = existing.copy()

    # Slang frequency counting across sessions
    counts = profile.get("slang_counts", {})
    for word in new_session.get("confirmed_slang", []):
        w = word.lower().strip()
        if w in SAFE_INFORMAL:
            counts[w] = counts.get(w, 0) + 1
    profile["slang_counts"] = counts

    # Confirmed after 2+ sessions
    profile["confirmed_slang"] = [
        w for w, c in counts.items() if c >= 2
    ][:8]   # cap at 8 words

    # Humor accumulation
    profile["humor_count"] = (
        profile.get("humor_count", 0)
        + new_session.get("humor_count", 0)
    )
    if new_session.get("humor_style") and new_session["humor_style"] != "none":
        humor_history = profile.get("humor_history", [])
        humor_history.append(new_session["humor_style"])
        profile["humor_history"] = humor_history[-8:]
        vals = [v for v in humor_history if v]
        profile["humor_style"] = max(set(vals), key=vals.count) if vals else "none"

    # Formality rolling update
    form_hist = profile.get("formality_history", [])
    if new_session.get("formality"):
        form_hist.append(new_session["formality"])
        profile["formality_history"] = form_hist[-8:]
        vals = [v for v in form_hist if v]
        profile["formality"] = max(set(vals), key=vals.count) if vals else "neutral"

    # Vocabulary level
    vocab_hist = profile.get("vocab_history", [])
    if new_session.get("vocabulary_level"):
        vocab_hist.append(new_session["vocabulary_level"])
        profile["vocab_history"] = vocab_hist[-8:]
        vals = [v for v in vocab_hist if v]
        profile["vocabulary_level"] = max(set(vals), key=vals.count) if vals else "casual"

    # Sentence length
    len_hist = profile.get("length_history", [])
    if new_session.get("sentence_length"):
        len_hist.append(new_session["sentence_length"])
        profile["length_history"] = len_hist[-8:]
        vals = [v for v in len_hist if v]
        profile["sentence_length"] = max(set(vals), key=vals.count) if vals else "medium"

    # Enthusiasm markers accumulate
    markers = set(profile.get("enthusiasm_markers", []))
    for m in new_session.get("enthusiasm_markers", []):
        if m.lower().strip() in SAFE_INFORMAL:
            markers.add(m.lower().strip())
    profile["enthusiasm_markers"] = list(markers)

    # Example phrases — keep freshest unique ones
    examples = profile.get("example_phrases", [])
    examples.extend(new_session.get("example_phrases", []))
    profile["example_phrases"] = list(dict.fromkeys(examples))[-5:]

    # Energy level
    if new_session.get("energy_level"):
        profile["energy_level"] = new_session["energy_level"]

    # Sessions analysed
    profile["sessions_analysed"] = profile.get("sessions_analysed", 0) + 1

    return profile


def get_mirror_instruction(
        style_profile: dict,
        session_count: int,
) -> str:
    """
    Returns mirror instruction string for prompt assembler.
    Gates:
      - Minimum 2 sessions before any mirroring
      - Slang only after confirmed (2+ sessions)
      - Humor only after 4+ events AND session 2+
      - Never mirrors within first 8 turns of a session
    """
    if not style_profile or session_count < 2:
        return ""

    sessions_analysed = style_profile.get("sessions_analysed", 0)
    if sessions_analysed < 2:
        return ""

    slang     = style_profile.get("confirmed_slang", [])
    humor_ct  = style_profile.get("humor_count", 0)
    humor_sty = style_profile.get("humor_style", "none")
    formality = style_profile.get("formality", "neutral")
    markers   = style_profile.get("enthusiasm_markers", [])
    sent_len  = style_profile.get("sentence_length", "medium")
    examples  = style_profile.get("example_phrases", [])

    parts = ["STYLE MIRRORING (natural, never forced):"]

    # Formality
    if formality in ("very_informal", "informal"):
        parts.append(
            "Student speaks very casually. Match their register — "
            "no stiff formal language. Contractions, relaxed phrasing."
        )

    # Slang — only confirmed words
    if slang and sessions_analysed >= 2:
        parts.append(
            f"These words appear naturally in this student's speech: "
            f"{', '.join(slang[:4])}. "
            "Use one or two if the moment genuinely fits — never force it."
        )

    # Enthusiasm markers
    safe_markers = [m for m in markers if m in SAFE_INFORMAL]
    if safe_markers and session_count >= 3:
        parts.append(
            f"Student uses '{safe_markers[0]}' naturally. "
            "You can use it once per session if it fits."
        )

    # Sentence length
    length_map = {
        "short":  "Student speaks in short bursts. Keep responses punchy.",
        "long":   "Student gives detailed answers. Depth is welcome.",
    }
    if sent_len in length_map:
        parts.append(length_map[sent_len])

    # Humor — high gate
    if humor_ct >= 4 and humor_sty != "none" and session_count >= 2:
        humor_map = {
            "playful":         "Student has a playful sense of humor. "
                               "A light natural joke is welcome when the moment fits. "
                               "Never forced. Never at their expense.",
            "self_deprecating": "Student uses self-aware humor. "
                                "You can gently match that tone occasionally.",
            "sarcastic":       "Student uses dry humor. "
                               "You can match lightly — never at their expense.",
        }
        if humor_sty in humor_map:
            parts.append(humor_map[humor_sty])

    # Example phrase context
    if examples and sessions_analysed >= 3:
        parts.append(
            f"Their natural register sounds like: '{examples[-1]}'. "
            "Match that energy and vocabulary level."
        )

    parts.append(
        "Goal: feel like a knowledgeable friend who speaks their language, "
        "not a formal system."
    )

    return "\n".join(parts)