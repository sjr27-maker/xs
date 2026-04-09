# comprehension + session classification
# feedback/session_scorer.py
"""
End-of-session scoring and classification.
Runs once when session ends — cost doesn't matter here,
accuracy does.
"""
import os
import json
import logging
from typing import Optional
from google import genai
from google.genai import types
from config import ANALYSIS_MODEL

logger  = logging.getLogger("SYRA.SessionScorer")
_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

SCORE_PROMPT = """Analyze this tutoring session transcript.
Subject: {subject}

Turns:
{transcript}

Return ONLY valid JSON:
{{
  "comprehension_score": 0,
  "session_classification": "NORMAL|ANOMALY|BREAKTHROUGH|DEPENDENT",
  "topics_covered": [],
  "topics_struggling": [],
  "new_misconceptions": [],
  "recommended_next_topic": null,
  "zpd_trajectory": "stable|improving|deteriorating",
  "dependency_trajectory": "stable|rising|falling",
  "key_insight": "one sentence — the most important thing to know about this student from this session"
}}

session_classification:
  NORMAL:       Standard session, reliable data
  ANOMALY:      Environmental factors contaminated signals — do not update base profile
  BREAKTHROUGH: Student achieved genuine conceptual understanding of something difficult
  DEPENDENT:    Student was consistently passive — dependency alarm triggered multiple times

comprehension_score: 0-100 based on quality of student explanations and correct answers."""


def score_session(
        turns:          list,
        subject:        str,
        session_anomaly: bool = False,
) -> dict:
    """
    Score the session and classify it.
    session_anomaly from session_checker overrides classification.
    """
    if not turns:
        return {
            "comprehension_score":    0,
            "session_classification": "ANOMALY" if session_anomaly else "NORMAL",
            "topics_covered":         [],
            "topics_struggling":      [],
            "new_misconceptions":     [],
            "recommended_next_topic": None,
            "zpd_trajectory":         "stable",
            "dependency_trajectory":  "stable",
            "key_insight":            "No turns recorded.",
        }

    # Build transcript string
    lines = []
    for t in turns:
        if hasattr(t, "student_text"):
            # Turn dataclass
            if t.student_text:
                lines.append(f"Student: {t.student_text}")
            if t.ai_response:
                lines.append(
                    f"SYRA [{t.error_type}|{t.zpd_position}|{t.affect_state}]: "
                    f"{t.ai_response}"
                )
        elif isinstance(t, dict):
            if t.get("student_text"):
                lines.append(f"Student: {t['student_text']}")
            if t.get("ai_response"):
                lines.append(f"SYRA: {t['ai_response']}")

    transcript = "\n".join(lines[:60])   # cap to avoid token overflow

    try:
        resp = _client.models.generate_content(
            model=ANALYSIS_MODEL,
            contents=SCORE_PROMPT.format(
                subject=subject,
                transcript=transcript,
            ),
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.1,
                thinking_config=types.ThinkingConfig(thinking_level="minimal"),
                max_output_tokens=500,
            ),
        )
        result = json.loads(resp.text)

        # Anomaly override
        if session_anomaly:
            result["session_classification"] = "ANOMALY"

        logger.info(
            f"Session scored: {result['comprehension_score']}/100 "
            f"[{result['session_classification']}]"
        )
        return result

    except Exception as e:
        logger.error(f"Session scoring failed: {e}")
        return {
            "comprehension_score":    0,
            "session_classification": "ANOMALY" if session_anomaly else "NORMAL",
            "topics_covered":         [],
            "topics_struggling":      [],
            "new_misconceptions":     [],
            "recommended_next_topic": None,
            "zpd_trajectory":         "stable",
            "dependency_trajectory":  "stable",
            "key_insight":            "Scoring failed.",
        }