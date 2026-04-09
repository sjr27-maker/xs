# CARELESS/PROCEDURAL/CONCEPTUAL/OVERLOAD_INDUCED
# cognition/error_classifier.py
"""
Classifies every student error into one of four types.
Each type triggers a different SYRA response — this is critical.

CARELESS:         Knows correct procedure, execution slip (tiredness, distraction)
                  → Light redirect, continue. Do NOT reduce belief scores.
PROCEDURAL:       Wrong step sequence or incomplete technique
                  → Targeted step correction. Reduce procedural_confidence slightly.
CONCEPTUAL:       False underlying belief generating the error
                  → Root belief investigation. Target the WHY.
OVERLOAD_INDUCED: Correct knowledge, execution failed from WM overload
                  → WM closure + pause. Do NOT reduce belief scores.
"""
import os
import json
import logging
from dataclasses import dataclass
from typing import Optional
from google import genai
from google.genai import types
from config import EXTRACT_MODEL

logger  = logging.getLogger("SYRA.ErrorClassifier")
_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

ERROR_TYPES = {"NONE", "CARELESS", "PROCEDURAL", "CONCEPTUAL", "OVERLOAD_INDUCED"}


@dataclass
class ErrorAnalysis:
    error_type:      str            # from ERROR_TYPES
    detected:        bool           = False
    description:     str            = ""
    root_belief:     Optional[str]  = None   # for CONCEPTUAL errors
    correct_version: Optional[str]  = None   # what the correct answer was
    confidence:      float          = 0.80


CLASSIFY_PROMPT = """Analyze this student's error in the context of the exchange.

Student said: "{student_text}"
SYRA responded: "{ai_text}"
Concept: {concept}
Session fatigue level: {fatigue_level}
Consecutive confused turns: {consecutive_confused}

Was there an error? If yes, classify it:

CARELESS: Student clearly knows the correct procedure but made a slip.
  Signs: previously correct on this, tiredness/fatigue context, single isolated wrong step
  Examples: sign mistake (-3 instead of +3), wrote x instead of x²

PROCEDURAL: Student attempted but applied the wrong steps or incomplete technique.
  Signs: systematic pattern, wrong method consistently, step left out
  Examples: took 3x as common but forgot x, didn't verify with LCM

CONCEPTUAL: Student has a false underlying belief generating the error.
  Signs: explains reasoning that is internally consistent but wrong
  Examples: "LCM gives you the numbers to use" (wrong — it's for verification)

OVERLOAD_INDUCED: Student knows the material but failed due to cognitive overload.
  Signs: high consecutive_confused, high fatigue, previously correct, incomplete writing
  Examples: wrote partial answer and stopped, made error mid-step they knew

Return ONLY valid JSON:
{{
  "detected": true|false,
  "error_type": "NONE|CARELESS|PROCEDURAL|CONCEPTUAL|OVERLOAD_INDUCED",
  "description": "one sentence describing the specific error",
  "root_belief": "the false belief if CONCEPTUAL, else null",
  "correct_version": "what the correct answer/step should have been",
  "confidence": 0.85
}}"""


def classify_error(
        student_text:        str,
        ai_text:             str,
        concept:             str,
        fatigue_level:       str = "low",
        consecutive_confused: int = 0,
) -> ErrorAnalysis:
    """
    Classify the error type in this exchange.
    Called synchronously — fast enough (~200ms) to not block conversation.
    """
    # Quick screen — if no error signal in the exchange, skip API call
    no_error_signals = [
        "exactly", "correct", "that's right", "perfect",
        "well done", "you got it", "spot on",
    ]
    if any(s in ai_text.lower() for s in no_error_signals):
        return ErrorAnalysis(error_type="NONE", detected=False)

    try:
        resp = _client.models.generate_content(
            model=EXTRACT_MODEL,
            contents=CLASSIFY_PROMPT.format(
                student_text=student_text,
                ai_text=ai_text,
                concept=concept,
                fatigue_level=fatigue_level,
                consecutive_confused=consecutive_confused,
            ),
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.1,
                thinking_config=types.ThinkingConfig(thinking_level="minimal"),
                max_output_tokens=250,
            ),
        )
        data = json.loads(resp.text)

        error_type = data.get("error_type", "NONE")
        if error_type not in ERROR_TYPES:
            error_type = "NONE"

        result = ErrorAnalysis(
            error_type      = error_type,
            detected        = data.get("detected", False),
            description     = data.get("description", ""),
            root_belief     = data.get("root_belief"),
            correct_version = data.get("correct_version"),
            confidence      = float(data.get("confidence", 0.80)),
        )

        if result.detected:
            logger.info(
                f"Error detected: {error_type} in {concept} "
                f"[confidence={result.confidence:.0%}]"
            )
            if result.root_belief:
                logger.info(f"Root belief: {result.root_belief}")

        return result

    except Exception as e:
        logger.debug(f"Error classification failed: {e}")
        return ErrorAnalysis(error_type="NONE", detected=False)


def get_error_response_instruction(analysis: ErrorAnalysis) -> str:
    """
    Returns the instruction string for the prompt assembler
    based on error type. This is what changes SYRA's response.
    """
    if not analysis.detected:
        return ""

    instructions = {
        "CARELESS": (
            "A careless slip was detected (not a knowledge gap). "
            "Acknowledge lightly: 'small slip there' and move forward. "
            "Do not dwell. Do not re-explain the concept."
        ),
        "PROCEDURAL": (
            f"A procedural error was detected: {analysis.description}. "
            "Walk through the specific step that was wrong. "
            f"Correct version: {analysis.correct_version}. "
            "Ask student to try the corrected step independently."
        ),
        "CONCEPTUAL": (
            f"A conceptual error was detected rooted in a false belief: "
            f"'{analysis.root_belief}'. "
            "Do NOT just correct the surface error. "
            "Ask a question that creates cognitive conflict with this belief. "
            "Help student discover the contradiction themselves."
        ),
        "OVERLOAD_INDUCED": (
            "The student's working memory is overloaded — this is an execution "
            "failure, not a knowledge gap. "
            "STOP introducing new content. "
            "Say: 'Let's pause — you actually know this. Let's clear the board.' "
            "Run WM closure. Reduce to one concept at a time."
        ),
    }
    return instructions.get(analysis.error_type, "")