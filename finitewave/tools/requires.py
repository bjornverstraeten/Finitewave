"""
Helper functions to check for required dependencies in examples.
"""

import shutil

def require_import(module: str, install: str):
    try:
        __import__(module)
    except ImportError as e:
        raise ImportError(
            f"This example requires `{module}`.\n"
            f"Install with:\n  {install}"
        ) from e

def require_ffmpeg_binary():
    if shutil.which("ffmpeg") is None:
        raise RuntimeError(
            "This example requires system FFmpeg (missing `ffmpeg` in PATH).\n"
            "Install FFmpeg:\n"
            "  - Ubuntu/Debian: sudo apt-get install ffmpeg\n"
            "  - macOS: brew install ffmpeg\n"
            "  - Conda: conda install -c conda-forge ffmpeg\n"
            "  - Windows: winget install Gyan.FFmpeg"
        )