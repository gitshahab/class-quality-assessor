
import json
from pathlib import Path

from src.audio import transcribe_audio_file
from src.scoring import score_assessment

# Project paths
ROOT = Path(__file__).resolve().parent
AUDIO_PATH = ROOT / "data" / "extracted_audio.wav"
VIDEO_JSON_PATH = ROOT / "video_analysis.json"

# 1. Validate input files
if not AUDIO_PATH.exists():
    raise FileNotFoundError(f"Audio file not found: {AUDIO_PATH}")

if not VIDEO_JSON_PATH.exists():
    raise FileNotFoundError(f"Video JSON not found: {VIDEO_JSON_PATH}")

# 2. Load existing video-analysis results
with VIDEO_JSON_PATH.open("r", encoding="utf-8") as file:
    video_result = json.load(file)

# 3. Transcribe the audio locally using Whisper
print("\nTranscribing audio... This may take a while.")

transcription_result = transcribe_audio_file(
    str(AUDIO_PATH),
    model_size="base",
)

# 4. Extract transcript and segments
# Supports the dictionary format expected from the audio pipeline.
if isinstance(transcription_result, dict):
    transcript = transcription_result.get(
        "full_transcript", ""
    )
    segments = transcription_result.get("segments", [])

elif isinstance(transcription_result, str):
    transcript = transcription_result
    segments = []

else:
    raise TypeError(
        "Unexpected transcription result type: "
        f"{type(transcription_result).__name__}. "
        "Inspect transcribe_audio_file() in src/audio.py."
    )

if not transcript or not str(transcript).strip():
    raise ValueError(
        "Transcription returned no text. "
        "Check the return format in src/audio.py."
    )

print(f"\nTranscript characters: {len(transcript)}")
print("\nTranscript preview:")
print(str(transcript)[:1000])

# 5. Run combined audio + video scoring
print("\nCalculating assessment score...")

assessment = score_assessment(
    video_analysis=video_result,
    transcript=str(transcript),
    segments=segments,
)

# 6. Display results
print("\nCombined assessment result:")
print(json.dumps(assessment, indent=2, ensure_ascii=False))

# 7. Save results for inspection
output_path = ROOT / "combined_assessment.json"

with output_path.open("w", encoding="utf-8") as file:
    json.dump(assessment, file, indent=2, ensure_ascii=False)

print(f"\nSaved combined assessment to: {output_path}")
