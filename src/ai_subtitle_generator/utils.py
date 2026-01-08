import shutil
import subprocess
import sys
import os
import numpy as np

def check_ffmpeg():
    """Check if ffmpeg is available and provide platform-specific installation instructions"""
    if shutil.which("ffmpeg") is None:
        print("❌ Error: ffmpeg is not installed or not in PATH.")
        print("Please install ffmpeg:")
        if sys.platform == "darwin":
            print("  brew install ffmpeg")
        elif sys.platform == "linux":
            print("  sudo apt install ffmpeg")
        elif sys.platform == "win32":
            print("  Download from https://ffmpeg.org/download.html and add to PATH")
        sys.exit(1)
    else:
        print("✅ ffmpeg is available")

def format_time(seconds):
    """Format seconds into VTT time format (HH:MM:SS.mmm)"""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds - int(seconds)) * 1000)
    return f"{h:02}:{m:02}:{s:02}.{ms:03}"

def resample_audio(audio_array: np.ndarray, src_rate: int, target_rate: int = 16000) -> np.ndarray:
    """Resample a mono float32 audio array to the target sample rate."""
    if audio_array.size == 0:
        return audio_array.astype(np.float32, copy=False)

    if src_rate == target_rate:
        return audio_array.astype(np.float32, copy=False)

    src_rate = float(src_rate)
    target_rate = float(target_rate)

    target_length = max(1, int(round(audio_array.shape[0] * target_rate / src_rate)))
    if target_length == audio_array.shape[0]:
        return audio_array.astype(np.float32, copy=False)

    x_old = np.linspace(0.0, 1.0, num=audio_array.shape[0], endpoint=False)
    x_new = np.linspace(0.0, 1.0, num=target_length, endpoint=False)
    resampled = np.interp(x_new, x_old, audio_array.astype(np.float32, copy=False))
    return resampled.astype(np.float32, copy=False)
