# output/speaker.py
"""
All SYRA voice output through Gemini native audio.
Replaces ElevenLabs entirely.

Two modes:
  speak()      — one-shot TTS for onboarding and session_checker
                 Uses Gemini TTS model, plays via StreamingPlayer
  LivePlayer   — used by live_session.py to play streaming PCM
                 from Gemini 3.1 Flash Live responses

Voice selection maps to archetype:
  maya  → Leda   (warm, gentle)
  lina  → Aoede  (clear, calm)
  arjun → Charon (direct, confident)
"""
import os
import threading
import logging
import numpy as np
import sounddevice as sd
from google import genai
from google.genai import types
from dotenv import load_dotenv
from config import OUT_RATE

load_dotenv()
logger  = logging.getLogger("SYRA.Speaker")
_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# Gemini TTS model for one-shot speech
TTS_MODEL = "gemini-2.5-flash-preview-tts"

VOICE_MAP = {
    "maya":  "Leda",    # warm, gentle female
    "lina":  "Aoede",   # clear, calm female — default
    "arjun": "Charon",  # direct, confident male
}
DEFAULT_VOICE = "Aoede"


# ── Streaming player (shared by both modes) ───────────────────────────

class StreamingPlayer:
    """
    Continuous PCM playback via sounddevice output callback.
    Thread-safe. Feed raw int16 bytes from any thread.
    No gaps, no speed artifacts.
    """

    def __init__(self, rate: int = OUT_RATE):
        self._buf    = np.array([], dtype=np.int16)
        self._lock   = threading.Lock()
        self._stream = sd.OutputStream(
            samplerate=rate,
            channels=1,
            dtype="int16",
            callback=self._callback,
            blocksize=2400,   # 100ms at 24kHz
        )
        self._stream.start()

    def _callback(self, outdata, frames, time_info, status):
        with self._lock:
            take = min(len(self._buf), frames)
            if take:
                outdata[:take, 0] = self._buf[:take]
                self._buf = self._buf[take:]
            if take < frames:
                outdata[take:, 0] = 0

    def feed(self, raw_bytes: bytes):
        """Add PCM bytes to playback buffer."""
        arr = np.frombuffer(raw_bytes, dtype=np.int16).copy()
        with self._lock:
            self._buf = np.concatenate([self._buf, arr])

    def clear(self):
        """Immediate silence — for barge-in."""
        with self._lock:
            self._buf = np.array([], dtype=np.int16)

    @property
    def playing(self) -> bool:
        with self._lock:
            return len(self._buf) > 0

    def wait_until_done(self, timeout: float = 15.0):
        """Block until buffer empties or timeout."""
        import time
        elapsed = 0.0
        while self.playing and elapsed < timeout:
            time.sleep(0.05)
            elapsed += 0.05

    def close(self):
        self._stream.stop()
        self._stream.close()


# Module-level player — shared across all speak() calls
_player: StreamingPlayer | None = None


def _get_player() -> StreamingPlayer:
    global _player
    if _player is None:
        _player = StreamingPlayer(OUT_RATE)
    return _player


# ── One-shot TTS (onboarding, session_checker) ────────────────────────

def speak(
        text:      str,
        archetype: str  = "lina",
        wait:      bool = True,
) -> None:
    """
    Speak text using Gemini TTS.
    Blocks until audio finishes playing if wait=True.

    Used by: onboarding, session_checker, any non-live context.
    """
    if not text or not text.strip():
        return

    voice = VOICE_MAP.get(archetype, DEFAULT_VOICE)

    try:
        response = _client.models.generate_content(
            model=TTS_MODEL,
            contents=text,
            config=types.GenerateContentConfig(
                response_modalities=["AUDIO"],
                speech_config=types.SpeechConfig(
                    voice_config=types.VoiceConfig(
                        prebuilt_voice_config=types.PrebuiltVoiceConfig(
                            voice_name=voice
                        )
                    )
                ),
            ),
        )

        # Extract audio bytes from response
        audio_data = None
        for part in response.candidates[0].content.parts:
            if hasattr(part, "inline_data") and part.inline_data:
                audio_data = part.inline_data.data
                break

        if not audio_data:
            logger.warning(f"TTS returned no audio for: {text[:50]}")
            return

        player = _get_player()
        player.feed(audio_data)

        if wait:
            player.wait_until_done()

    except Exception as e:
        logger.error(f"TTS error: {e}")
        # Fallback: print text so session doesn't silently fail
        print(f"  [SYRA speaks]: {text}")


def speak_async(text: str, archetype: str = "lina") -> None:
    """Non-blocking speak — fire and forget."""
    threading.Thread(
        target=speak,
        args=(text, archetype, True),
        daemon=True,
    ).start()


def stop_speaking() -> None:
    """Immediately silence output — used for barge-in."""
    if _player:
        _player.clear()


def is_speaking() -> bool:
    """True if audio is currently playing."""
    return _player.playing if _player else False


def close_speaker() -> None:
    """Call at session end to release audio resources."""
    global _player
    if _player:
        _player.close()
        _player = None