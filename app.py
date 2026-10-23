import os
import json
import streamlit as st
import re

from src.audio import extract_audio_from_video, transcribe_audio_file
from src.video import analyze_video
from src.scoring import score_assessment
from src.ml_model import predict_lesson


def build_ml_features(video_analysis, transcript):
    """Convert audio transcript and video analysis into model features."""

    metadata = video_analysis.get("video_metadata", {})
    visual = video_analysis.get("visual_checks", {})

    lighting = visual.get("lighting", {})
    black_screen = visual.get("black_screen", {})
    face_visibility = visual.get("face_visibility", {})

    text = transcript or ""

    introduction_present = int(bool(re.search(
        r"\b(hello|hi everyone|good morning|good afternoon|"
        r"welcome|how are you|how was your day|nice to meet you)\b",
        text,
        re.IGNORECASE,
    )))

    homework_present = int(bool(re.search(
        r"\b(homework|assignment|practice questions|worksheet|"
        r"solve these|for next class|complete the exercise)\b",
        text,
        re.IGNORECASE,
    )))

    closure_present = int(bool(re.search(
        r"\b(to summarize|in summary|let's recap|today we learned|"
        r"that's all for today|see you next class|goodbye|bye everyone)\b",
        text,
        re.IGNORECASE,
    )))

    question_count = len(re.findall(
        r"\?|\b(what|why|how|when|where|which|who|"
        r"can you|could you|do you|did you|any questions)\b",
        text,
        re.IGNORECASE,
    ))

    return {
        "duration_minutes": (
            float(metadata["duration_seconds"]) / 60
            if metadata.get("duration_seconds") is not None
            else None
        ),
        "introduction_present": introduction_present,
        "question_marker_count": question_count,
        "homework_present": homework_present,
        "closure_present": closure_present,
        "average_brightness": lighting.get("average_frame_brightness"),
        "dark_frame_ratio": lighting.get("dark_sampled_frame_ratio"),
        "black_frame_ratio": black_screen.get("sampled_frame_ratio"),
        "face_detection_ratio": face_visibility.get(
            "face_detection_ratio"
        ),
    }


DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

st.set_page_config(
    page_title="Tutor QA Pipeline",
    page_icon="🎓",
    layout="wide",
)

st.title("🎓 Tutor QA Pipeline")
st.caption(
    "Phase 1: local audio transcription · Phase 2: local video indicators. "
    "Automated checks support human review; they do not grade teaching quality."
)

uploaded_file = st.file_uploader(
    "Upload a class recording (MP4 or WAV)",
    type=["mp4", "wav"],
)

if uploaded_file is not None:
    safe_name = os.path.basename(uploaded_file.name)
    input_path = os.path.join(DATA_DIR, safe_name)

    # Avoid rewriting the file on every Streamlit rerun when possible.
    file_bytes = uploaded_file.getvalue()
    if (
        not os.path.exists(input_path)
        or os.path.getsize(input_path) != len(file_bytes)
    ):
        with open(input_path, "wb") as file:
            file.write(file_bytes)

    st.success(f"Recording staged locally: {safe_name}")
    extension = os.path.splitext(safe_name)[1].lower()

    audio_tab, video_tab, assessment_tab = st.tabs(
    ["🎙️ Audio analysis", "🎞️ Video analysis", "🧠 Combined assessment"])

    with audio_tab:
        st.write("Extract speech and produce a timestamped transcript.")
        if st.button("Run Audio Pipeline", type="primary", key="run_audio"):
            try:
                with st.spinner("Processing audio and transcribing..."):
                    target_audio = input_path
                    if extension == ".mp4":
                        target_audio = os.path.join(DATA_DIR, "extracted_audio.wav")
                        extract_audio_from_video(input_path, target_audio)

                    result = transcribe_audio_file(target_audio, model_size="base")

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
                        "No speech was detected. Try a clearer recording or review VAD settings."
                    )

                st.subheader("Timestamped Segments")
                st.json(result["segments"])
            except Exception as exc:
                st.error(f"Audio pipeline failed: {exc}")

    with video_tab:
        if extension != ".mp4":
            st.info("Video analysis requires an MP4 video file. Upload an MP4 to use Phase 2.")
        else:
            st.write(
                "Checks sampled frames for possible black screens, overall darkness, "
                "near-identical frames, and approximate face visibility."
            )
            sample_interval = st.select_slider(
                "Sampling interval (seconds)",
                options=[1, 2, 3, 5],
                value=1,
                help="One second is more detailed but takes longer to process.",
            )
            if st.button("Run Video Pipeline", type="primary", key="run_video"):
                try:
                    with st.spinner("Analyzing video locally with OpenCV..."):
                        result = analyze_video(
                            input_path,
                            sample_interval_seconds=float(sample_interval),
                        )

                    metadata = result["video_metadata"]
                    st.subheader("Video Summary")
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Duration", f'{metadata["duration_seconds"] / 60:.1f} min')
                    c2.metric("Resolution", f'{metadata["width"]} × {metadata["height"]}')
                    c3.metric("FPS", f'{metadata["fps"]:.2f}')
                    c4.metric("Frames sampled", metadata["sampled_frames"])

                    checks = result["visual_checks"]
                    st.subheader("Visual checks — review indicators, not grades")
                    b1, b2, b3, b4 = st.columns(4)
                    b1.metric("Black screen", checks["black_screen"]["status"].replace("_", " ").title())
                    b2.metric("Lighting", checks["lighting"]["status"].replace("_", " ").title())
                    b3.metric("Near-identical frames", checks["frozen_frames"]["status"].replace("_", " ").title())
                    face_ratio = checks["face_visibility"]["face_detection_ratio"]
                    b4.metric(
                        "Face detection samples",
                        f'{face_ratio:.1%}' if face_ratio is not None else "Not assessed",
                    )

                    with st.expander("Black-screen details"):
                        st.write(checks["black_screen"]["note"])
                        st.json(checks["black_screen"])
                    with st.expander("Lighting details"):
                        st.write(checks["lighting"]["note"])
                        st.json(checks["lighting"])
                    with st.expander("Near-identical-frame details"):
                        st.warning(checks["frozen_frames"]["note"])
                        st.json(checks["frozen_frames"])
                    with st.expander("Face-detection details"):
                        st.write(checks["face_visibility"]["note"])
                        st.json(checks["face_visibility"])

                    st.download_button(
                        "Download video analysis JSON",
                        data=json.dumps(result, indent=2),
                        file_name="video_analysis.json",
                        mime="application/json",
                    )
                    st.caption(
                        "Privacy: this implementation analyzes the uploaded video locally "
                        "with OpenCV and does not send the video to an external AI API. "
                        "The staged file remains in the local data folder until you remove it."
                    )
                except Exception as exc:
                    st.error(f"Video pipeline failed: {exc}")
    with assessment_tab:
        st.subheader("Combined lesson assessment")

        st.write(
            "Run audio transcription and video analysis together, then "
            "compare the heuristic assessment with the experimental ML prediction."
        )

        if extension != ".mp4":
            st.info(
                "The combined assessment requires an MP4 recording "
                "because the model uses both audio-transcript and video features."
            )
        elif st.button(
            "Run Combined Assessment",
            type="primary",
            key="run_combined",
        ):
            try:
                with st.spinner(
                    "Running audio transcription, video analysis, and assessment..."
                ):
                    target_audio = os.path.join(
                        DATA_DIR, "extracted_audio.wav"
                    )

                    extract_audio_from_video(input_path, target_audio)

                    audio_result = transcribe_audio_file(
                        target_audio,
                        model_size="base",
                    )

                    transcript = audio_result.get("full_transcript", "")
                    segments = audio_result.get("segments", [])

                    video_result = analyze_video(
                        input_path,
                        sample_interval_seconds=float(sample_interval),
                    )

                    heuristic_result = score_assessment(
                        video_analysis=video_result,
                        transcript=transcript,
                        segments=segments,
                    )

                    features = build_ml_features(
                        video_result,
                        transcript,
                    )

                    model_path = "models/lesson_quality_model.joblib"

                    if not os.path.isfile(model_path):
                        st.error(
                            "Trained ML model not found. "
                            "Train it first using src.ml_model.train_model()."
                        )
                    else:
                        ml_result = predict_lesson(features, model_path)

                        st.subheader("1. Heuristic assessment")

                        c1, c2, c3 = st.columns(3)

                        c1.metric(
                            "Heuristic score",
                            (
                                f'{heuristic_result["score"]}/100'
                                if heuristic_result["score"] is not None
                                else "Not available"
                            ),
                        )

                        c2.metric(
                            "Heuristic rating",
                            heuristic_result["rating"],
                        )

                        c3.metric(
                            "Evidence coverage",
                            f'{heuristic_result["evidence_coverage_percent"]:.1f}%',
                        )

                        st.subheader("2. Experimental ML prediction")

                        st.metric(
                            "Predicted category",
                            ml_result["predicted_label"],
                        )

                        st.warning(
                            "Experimental only: this model was trained on "
                            "illustrative synthetic examples. Its prediction "
                            "is not a reliable teaching-quality grade. "
                            "Human review is required."
                        )

                        st.subheader("3. Features supplied to the model")
                        st.json(features)

                        st.subheader("4. Heuristic evidence")
                        st.json(heuristic_result["criteria"])

                        report = {
                            "heuristic_assessment": heuristic_result,
                            "ml_prediction": ml_result,
                            "ml_features": features,
                            "human_review_required": True,
                        }

                        st.download_button(
                            "Download combined assessment JSON",
                            data=json.dumps(report, indent=2),
                            file_name="combined_assessment.json",
                            mime="application/json",
                        )

            except Exception as exc:
                st.error(f"Combined assessment failed: {exc}")
else:
    st.info("Upload an MP4 to run both audio and video analysis, or a WAV to run audio analysis.")