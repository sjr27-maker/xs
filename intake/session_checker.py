# 3-question spoken check-in at session start
# intake/session_checker.py
"""
3-question spoken check-in at session start.
Sets session_context flags before any teaching begins.
If anomaly_flag=True, profile_manager will block all updates.

Takes ~45 seconds. Non-negotiable — runs every session.
"""
import os
import json
import logging
from dataclasses import dataclass
from typing import Optional
from dotenv import load_dotenv
from google import genai
from google.genai import types

from intake.recorder  import record_until_enter, save_wav
from intake.transcriber import transcribe
from output.tts_client  import stream_tts_and_play
from config import EXTRACT_MODEL

load_dotenv()
logger  = logging.getLogger("SYRA.SessionChecker")
_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))


@dataclass
class SessionContext:
    fatigue_level:   str   = "low"     # low | medium | high
    external_load:   bool  = False     # other cognitive demands today
    time_pressure:   bool  = False     # limited time available
    anomaly_flag:    bool  = False     # True blocks profile updates
    emotional_state: str   = "neutral" # neutral | anxious | excited | sad
    session_note:    str   = ""        # free text for teacher report


CHECK_IN_QUESTIONS = [
    "Hey — before we start, how are you feeling right now? Tired, fresh, stressed?",
    "Did you just come from something demanding — like exams, lots of homework, or a rough day?",
    "How much time do you have today? Are we in a hurry or do we have space to explore?",
]

EXTRACTION_PROMPT = """Analyze these three student check-in responses.

Q1 (energy/feeling): {q1}
Q2 (external load): {q2}
Q3 (time pressure): {q3}

Return ONLY valid JSON:
{{
  "fatigue_level": "low|medium|high",
  "external_load": true|false,
  "time_pressure": true|false,
  "anomaly_flag": true|false,
  "emotional_state": "neutral|anxious|excited|sad|frustrated",
  "session_note": "one sentence for teacher report"
}}

anomaly_flag=true if: fatigue_level=high AND (external_load OR time_pressure).
This means the session data should not update the permanent student profile."""


def _speak(text: str, archetype: str = "lina"):
    """Speak to student using warm neutral IPC."""
    neutral_ipc = {
        "dominance": 0.5, "warmth": 0.75,
        "pace": "medium", "giving_up": False,
    }
    stream_tts_and_play(text, neutral_ipc, archetype)


def _record_and_transcribe() -> str:
    """Record one answer, return transcript."""
    input("  [ Press Enter to answer ]")
    audio = record_until_enter()
    path  = save_wav(audio)
    try:
        text, _, _ = transcribe(path)
        return text.strip()
    finally:
        import os as _os
        if _os.path.exists(path):
            _os.unlink(path)


def _extract_context(answers: list[str]) -> SessionContext:
    """Use Gemini to classify the three answers into structured flags."""
    prompt = EXTRACTION_PROMPT.format(
        q1=answers[0] or "no response",
        q2=answers[1] or "no response",
        q3=answers[2] or "no response",
    )
    try:
        resp = _client.models.generate_content(
            model=EXTRACT_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.1,
                thinking_config=types.ThinkingConfig(thinking_level="minimal"),
                max_output_tokens=200,
            ),
        )
        data = json.loads(resp.text)
        return SessionContext(
            fatigue_level   = data.get("fatigue_level",   "low"),
            external_load   = data.get("external_load",   False),
            time_pressure   = data.get("time_pressure",   False),
            anomaly_flag    = data.get("anomaly_flag",    False),
            emotional_state = data.get("emotional_state", "neutral"),
            session_note    = data.get("session_note",    ""),
        )
    except Exception as e:
        logger.warning(f"Session context extraction failed: {e}")
        return SessionContext()  # safe defaults — assume normal


def run_session_check(
        student_name: str = "there",
        archetype:    str = "lina",
) -> SessionContext:
    """
    Run the 3-question check-in and return SessionContext.
    Called once at session start, before any teaching.
    """
    print("\n  [ Session check-in ]")
    _speak(
        f"Hey {student_name}! Before we dive in, I just want to "
        f"check how you're doing — three quick questions.",
        archetype
    )

    answers = []
    for i, question in enumerate(CHECK_IN_QUESTIONS):
        print(f"\n  SYRA: {question}")
        _speak(question, archetype)
        answer = _record_and_transcribe()
        if answer:
            print(f"  Student: {answer}")
        else:
            print("  [ No response — using default ]")
        answers.append(answer)

    ctx = _extract_context(answers)

    # Confirm anomaly to student warmly
    if ctx.anomaly_flag:
        msg = (
            "Got it — sounds like today might not be the easiest. "
            "Let's keep things light and just see how far we get. "
            "No pressure at all."
        )
        _speak(msg, archetype)
        print(f"\n  [Session flagged: ANOMALY — profile updates blocked]")
        print(f"  Reason: fatigue={ctx.fatigue_level}, "
              f"external_load={ctx.external_load}, "
              f"time_pressure={ctx.time_pressure}")
    elif ctx.fatigue_level == "medium":
        _speak(
            "Okay, sounds good. Let's keep things focused today.",
            archetype
        )
    else:
        _speak(
            "Great — let's make the most of this session.",
            archetype
        )

    logger.info(
        f"Session context: fatigue={ctx.fatigue_level} "
        f"anomaly={ctx.anomaly_flag} "
        f"emotion={ctx.emotional_state}"
    )
    return ctx