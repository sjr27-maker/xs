# sounddevice PCM player, no pygame
# output/streaming_player.py
"""
Standalone streaming PCM player.
Can be imported independently by voice/live_session.py
without pulling in the full speaker.py TTS machinery.

Uses sounddevice output callback — continuous, gap-free.
Feed raw int16 bytes from any thread.
"""
import threading
import numpy as np
import sounddevice as sd
from config import OUT_RATE


class StreamingPlayer:
    """
    Continuous PCM playback via sounddevice output callback.
    Thread-safe. No pygame dependency.
    No gaps, no speed artifacts, no queue size limits.

    Usage:
        player = StreamingPlayer()
        player.feed(raw_pcm_bytes)   # call from any thread
        player.wait_until_done()
        player.close()
    """

    def __init__(self, rate: int = OUT_RATE):
        self._buf    = np.array([], dtype=np.int16)
        self._lock   = threading.Lock()
        self._rate   = rate
        self._stream = sd.OutputStream(
            samplerate=rate,
            channels=1,
            dtype="int16",
            callback=self._callback,
            blocksize=2400,   # 100ms at 24kHz — smooth, low latency
        )
        self._stream.start()

    def _callback(self, outdata, frames, time_info, status):
        with self._lock:
            take = min(len(self._buf), frames)
            if take:
                outdata[:take, 0] = self._buf[:take]
                self._buf = self._buf[take:]
            if take < frames:
                outdata[take:, 0] = 0   # silence pad — no click

    def feed(self, raw_bytes: bytes):
        """
        Add PCM bytes to playback buffer.
        Can be called from any thread.
        Bytes must be 16-bit little-endian PCM at self._rate.
        """
        if not raw_bytes:
            return
        arr = np.frombuffer(raw_bytes, dtype=np.int16).copy()
        with self._lock:
            self._buf = np.concatenate([self._buf, arr])

    def clear(self):
        """
        Immediate silence — discard all buffered audio.
        Used for barge-in interruption.
        """
        with self._lock:
            self._buf = np.array([], dtype=np.int16)

    @property
    def playing(self) -> bool:
        """True if audio is currently buffered and playing."""
        with self._lock:
            return len(self._buf) > 0

    @property
    def buffer_duration_ms(self) -> float:
        """How many milliseconds of audio are buffered."""
        with self._lock:
            return (len(self._buf) / self._rate) * 1000

    def wait_until_done(self, timeout: float = 15.0):
        """
        Block until buffer empties or timeout expires.
        Used in onboarding and session_checker where
        we need to wait for SYRA to finish speaking.
        """
        import time
        elapsed = 0.0
        while self.playing and elapsed < timeout:
            time.sleep(0.05)
            elapsed += 0.05

    def close(self):
        """Release audio resources. Call at session end."""
        try:
            self._stream.stop()
            self._stream.close()
        except Exception:
            pass