# sounddevice, 320-sample frames. Unchanged.
# intake/recorder.py
import numpy as np
import sounddevice as sd
import tempfile
import os
import scipy.io.wavfile as wav_io
from dotenv import load_dotenv
from config import IN_RATE, FRAME_SAMPLES

load_dotenv()
DEVICE_INDEX = int(os.getenv("DEVICE_INDEX", "1"))


def record_until_enter() -> np.ndarray:
    """
    Record mic audio until student presses Enter.
    Returns float32 numpy array.
    """
    frames = []

    def callback(indata, frame_count, time_info, status):
        frames.append(indata.copy().flatten())

    with sd.InputStream(
        samplerate=IN_RATE,
        channels=1,
        device=DEVICE_INDEX,
        dtype="float32",
        blocksize=FRAME_SAMPLES,
        callback=callback,
    ):
        input()   # blocks until Enter

    if not frames:
        return np.zeros(IN_RATE, dtype=np.float32)
    return np.concatenate(frames)


def save_wav(audio: np.ndarray) -> str:
    """
    Save float32 audio array to a temp WAV file.
    Returns path to file. Caller must delete.
    """
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    pcm = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
    wav_io.write(tmp.name, IN_RATE, pcm)
    return tmp.name