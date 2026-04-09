# Deepgram raw HTTP. Unchanged.
# intake/transcriber.py
import os
import httpx
import json
from dotenv import load_dotenv

load_dotenv()

DEEPGRAM_URL = "https://api.deepgram.com/v1/listen"
DEEPGRAM_KEY = os.getenv("DEEPGRAM_API_KEY")


def transcribe(audio_path: str) -> tuple[str, int, float]:
    """
    Transcribe audio file using Deepgram Nova-3.
    Returns: (transcript, filler_count, duration_seconds)
    Raw HTTP — immune to SDK version changes.
    """
    with open(audio_path, "rb") as f:
        audio_data = f.read()

    params = {
        "model":        "nova-3",
        "language":     "en-IN",
        "smart_format": "true",
        "punctuate":    "true",
        "filler_words": "true",
        "utterances":   "true",
    }
    headers = {
        "Authorization": f"Token {DEEPGRAM_KEY}",
        "Content-Type":  "audio/wav",
    }

    try:
        response = httpx.post(
            DEEPGRAM_URL,
            params=params,
            headers=headers,
            content=audio_data,
            timeout=30.0,
        )
        response.raise_for_status()
        result = response.json()

    except httpx.HTTPStatusError as e:
        print(f"  [STT HTTP error: {e.response.status_code}]")
        return "", 0, 0.0
    except Exception as e:
        print(f"  [STT error: {e}]")
        return "", 0, 0.0

    try:
        channel    = result["results"]["channels"][0]
        alt        = channel["alternatives"][0]
        transcript = alt.get("transcript", "").strip()
        words      = alt.get("words", [])
        fillers    = sum(1 for w in words if w.get("type") == "filler")
        duration   = result.get("metadata", {}).get("duration", 0.0)
        return transcript, fillers, duration

    except (KeyError, IndexError) as e:
        print(f"  [STT parse error: {e}]")
        print(f"  Raw: {json.dumps(result)[:200]}")
        return "", 0, 0.0