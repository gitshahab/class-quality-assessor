
import os
import subprocess
from functools import lru_cache

from faster_whisper import WhisperModel


@lru_cache(maxsize=2)
def load_whisper_model(model_size: str = "base") -> WhisperModel:
    """Load and reuse a CPU-based Whisper model."""
    return WhisperModel(
        model_size,
        device="cpu",
        compute_type="int8",
    )


def extract_audio_from_video(
    video_path: str,
    output_wav_path: str,
) -> str:
    """Extract mono, 16 kHz, 16-bit PCM audio from a video."""

    if not os.path.isfile(video_path):
        raise FileNotFoundError(
            f"Input video file not found: {video_path}"
        )

    output_dir = os.path.dirname(output_wav_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    command = [
        "ffmpeg",
        "-y",
        "-i", video_path,
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", "16000",
        "-ac", "1",
        output_wav_path,
    ]

    try:
        subprocess.run(
            command,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(
            "FFmpeg was not found. Install FFmpeg and add it to PATH."
        ) from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f"FFmpeg could not extract audio: {exc.stderr[-2000:]}"
        ) from exc

    return output_wav_path


def transcribe_audio_file(
    audio_path: str,
    model_size: str = "base",
) -> dict:
    """Transcribe audio and return text, timestamps, and language info."""

    if not os.path.isfile(audio_path):
        raise FileNotFoundError(
            f"Input audio file not found: {audio_path}"
        )

    model = load_whisper_model(model_size)

    segments, info = model.transcribe(
        audio_path,
        beam_size=5,
        vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 500},
    )

    timestamps = []
    transcript_parts = []

    # Transcription runs as we iterate through the generator.
    for segment in segments:
        text = segment.text.strip()

        if not text:
            continue

        transcript_parts.append(text)
        timestamps.append({
            "start": round(segment.start, 2),
            "end": round(segment.end, 2),
            "text": text,
        })

    return {
        "full_transcript": " ".join(transcript_parts),
        "segments": timestamps,
        "language": info.language,
        "language_probability": round(
            info.language_probability, 2
        ),
    }
