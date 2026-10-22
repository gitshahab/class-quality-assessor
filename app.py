import os
import streamlit as st

from src.audio import (
    extract_audio_from_video,
    transcribe_audio_file,
)

st.set_page_config(
    page_title="Tutor QA Pipeline - Audio Test",
    page_icon="🎙️",
    layout="wide",
)

st.title("🎙️ Phase 1: Audio Processing & Transcription Test")
st.caption("Extract audio, filter silence, and generate a timestamped transcript.")

os.makedirs("data", exist_ok=True)

uploaded_file = st.file_uploader(
    "Upload a class recording sample (MP4 or WAV)",
    type=["mp4", "wav"],
)

if uploaded_file is not None:
    # Prevent directory components in the uploaded filename.
    safe_name = os.path.basename(uploaded_file.name)
    input_path = os.path.join("data", safe_name)

    with open(input_path, "wb") as file:
        file.write(uploaded_file.getbuffer())

    st.success(f"Recording staged: {safe_name}")

    if st.button("Run Audio Pipeline", type="primary"):
        try:
            with st.spinner("Processing audio and transcribing..."):
                target_audio = input_path

                if os.path.splitext(safe_name)[1].lower() == ".mp4":
                    st.info("Extracting audio with FFmpeg...")
                    target_audio = os.path.join(
                        "data", "extracted_audio.wav"
                    )
                    extract_audio_from_video(
                        input_path,
                        target_audio,
                    )

                result = transcribe_audio_file(
                    target_audio,
                    model_size="base",
                )

            st.subheader("Results Summary")
            col1, col2 = st.columns(2)

            col1.metric("Detected language", result["language"])
            col2.metric(
                "Language confidence",
                f'{result["language_probability"]:.0%}',
            )

            st.subheader("Full Transcript")

            if result["full_transcript"]:
                st.text_area(
                    "Generated transcript",
                    value=result["full_transcript"],
                    height=250,
                )
            else:
                st.warning(
                    "No speech was detected. Try a clearer recording "
                    "or review the VAD settings."
                )

            st.subheader("Timestamped Segments")
            st.json(result["segments"])

        except Exception as exc:
            st.error(f"Pipeline failed: {exc}")
