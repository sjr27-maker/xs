# cognition/error_classifier.py
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

FULL_CONFIRMATION_PHRASES = [
    "that's exactly right",
    "that's correct",
    "you got it",
    "perfect, that's it",
    "spot on",
    "well done, that's",
    "yes, exactly",
    "absolutely correct",
    "that is right",
    "correct, well done",
]


@dataclass
class ErrorAnalysis:
    error_type:      str
    detected:        bool           = False
    description:     str            = ""
    root_belief:     Optional[str]  = None
    correct_version: Optional[str]  = None
    confidence:      float          = 0.80


CLASSIFY_PROMPT = """Analyze this tutoring exchange for student errors.

Student said: "{student_text}"
SYRA responded: "{ai_text}"
Concept: {concept}
Session fatigue level: {fatigue_level}
Consecutive confused turns: {consecutive_confused}

Was there a student error? If yes, classify it:

CARELESS: Student knows the correct procedure but made an execution slip.
  Signs: previously correct on this, tiredness/fatigue context, single isolated wrong step.
  Examples: sign mistake (-3 instead of +3), wrote x instead of x squared.

PROCEDURAL: Student applied wrong steps or incomplete technique.
  Signs: systematic pattern, wrong method consistently, step left out.
  Examples: took 3x as common but forgot x, did HCF instead of LCM.

CONCEPTUAL: Student has a false underlying belief generating the error.
  Signs: explains reasoning that is internally consistent but fundamentally wrong.
  Examples: believes LCM gives splitting numbers rather than being a verification tool.

OVERLOAD_INDUCED: Student knows the material but failed due to cognitive overload.
  Signs: high consecutive_confused, high fatigue, previously correct on this, incomplete writing, gave up.
  Examples: wrote partial answer and stopped, made error mid-step they clearly knew.

Return ONLY valid JSON:
{{
  "detected": true,
  "error_type": "NONE|CARELESS|PROCEDURAL|CONCEPTUAL|OVERLOAD_INDUCED",
  "description": "one sentence describing the specific error",
  "root_belief": "the false underlying belief if CONCEPTUAL, else null",
  "correct_version": "what the correct answer or step should have been",
  "confidence": 0.85
}}

If no student error exists, return:
{{"detected": false, "error_type": "NONE", "description": "", "root_belief": null, "correct_version": null, "confidence": 0.90}}"""


def classify_error(
        student_text:         str,
        ai_text:              str,
        concept:              str,
        fatigue_level:        str = "low",
        consecutive_confused: int = 0,
) -> ErrorAnalysis:
    """
    Classify the error type in this exchange.
    Fixed: quick screen now only skips on FULL confirmation phrases,
    not any occurrence of words like 'correct' or 'exactly'.
    """
    if not student_text or not student_text.strip():
        return ErrorAnalysis(error_type="NONE", detected=False)

    # Fixed quick screen — only skip if AI is fully confirming the answer
    # NOT just because the word "correct" or "exactly" appears somewhere
    ai_lower = ai_text.lower()
    if any(phrase in ai_lower for phrase in FULL_CONFIRMATION_PHRASES):
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
        raw = resp.text.strip()
        if "```" in raw:
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        data = json.loads(raw.strip())

        error_type = data.get("error_type", "NONE")
        if error_type not in ERROR_TYPES:
            error_type = "NONE"

        result = ErrorAnalysis(
            error_type=error_type,
            detected=data.get("detected", False),
            description=data.get("description", ""),
            root_belief=data.get("root_belief"),
            correct_version=data.get("correct_version"),
            confidence=float(data.get("confidence", 0.80)),
        )

        if result.detected:
            logger.info(
                f"Error: {error_type} in {concept} "
                f"[conf={result.confidence:.0%}]"
            )
            if result.root_belief:
                logger.info(f"Root belief: {result.root_belief}")

        return result

    except json.JSONDecodeError as e:
        logger.debug(f"Error classifier JSON parse failed: {e}")
        return ErrorAnalysis(error_type="NONE", detected=False)
    except Exception as e:
        logger.debug(f"Error classifier failed: {e}")
        return ErrorAnalysis(error_type="NONE", detected=False)


def get_error_response_instruction(analysis: ErrorAnalysis) -> str:
    """
    Returns instruction string for prompt assembler based on error type.
    Each error type triggers a different SYRA response.
    """
    if not analysis.detected or analysis.error_type == "NONE":
        return ""

    instructions = {
        "CARELESS": (
            "A careless slip was detected — not a knowledge gap. "
            "Acknowledge lightly: 'small slip there' and move forward. "
            "Do not re-explain the concept. Do not dwell."
        ),
        "PROCEDURAL": (
            f"A procedural error was detected: {analysis.description}. "
            "Walk through the specific wrong step. "
            f"Correct version: {analysis.correct_version or 'see context'}. "
            "Ask student to try the corrected step independently."
        ),
        "CONCEPTUAL": (
            f"A conceptual error rooted in a false belief: "
            f"'{analysis.root_belief or analysis.description}'. "
            "Do NOT just correct the surface error. "
            "Ask a question that creates cognitive conflict with this belief. "
            "Help the student discover the contradiction themselves."
        ),
        "OVERLOAD_INDUCED": (
            "The student's working memory is overloaded — "
            "this is an execution failure, not a knowledge gap. "
            "STOP introducing new content. "
            "Say: 'Let us pause — you actually know this. "
            "Let us clear the board.' "
            "Run WM closure. Reduce to one concept at a time."
        ),
    }
    return instructions.get(analysis.error_type, "")