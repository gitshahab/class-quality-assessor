
"""Explainable baseline scoring for the Class Quality Assessor.

This is a heuristic prototype, not a validated measure of teaching quality.
All results should be reviewed by a human.
"""
import re


WEIGHTS = {
    "introduction_and_rapport": 15,
    "questions_and_interaction": 20,
    "homework_or_practice": 15,
    "lesson_closure": 10,
    "lighting": 10,
    "black_screen": 10,
    "possible_freeze": 10,
    "face_visibility": 10,
}

TRANSCRIPT_PATTERNS = {
    "introduction_and_rapport": (
        r"\b(hello|hi everyone|good morning|good afternoon|welcome|"
        r"how are you|how was your day|nice to meet you)\b"
    ),
    "homework_or_practice": (
        r"\b(homework|assignment|practice questions|worksheet|"
        r"solve these|for next class|complete the exercise)\b"
    ),
    "lesson_closure": (
        r"\b(to summarize|in summary|let's recap|today we learned|"
        r"that's all for today|see you next class|goodbye|bye everyone)\b"
    ),
}


def score_assessment(video_analysis=None, transcript=None, segments=None):
    """Score available transcript/video evidence without treating missing data as failure."""
    video_analysis = video_analysis or {}
    visual = video_analysis.get("visual_checks", {})
    metadata = video_analysis.get("video_metadata", {})
    duration = float(metadata.get("duration_seconds") or 0)

    if not transcript and segments:
        transcript = " ".join(
            str(segment.get("text", ""))
            for segment in segments
            if isinstance(segment, dict)
        )

    transcript_available = isinstance(transcript, str) and bool(transcript.strip())
    criteria = []

    def add(name, value, evidence, caveat=""):
        weight = WEIGHTS[name]
        score_percent = None if value is None else round(
            max(0.0, min(1.0, value)) * 100, 1
        )
        item = {
            "criterion": name,
            "weight": weight,
            "score_percent": score_percent,
            "weighted_points": (
                None if score_percent is None
                else round(weight * score_percent / 100, 2)
            ),
            "status": (
                "not_assessed" if value is None
                else "observed" if value > 0
                else "not_observed"
            ),
            "evidence": evidence,
        }
        if caveat:
            item["caveat"] = caveat
        criteria.append(item)

    # Transcript indicators: keyword heuristics, not semantic understanding.
    if transcript_available:
        text = transcript.strip()

        for name, pattern in TRANSCRIPT_PATTERNS.items():
            match = re.search(pattern, text, re.IGNORECASE)
            add(
                name,
                1.0 if match else 0.0,
                f'Matched phrase: "{match.group(0)}"'
                if match else "No matching phrase detected.",
                "Keyword-based check; paraphrases and transcription errors can be missed.",
            )

        questions = re.findall(
            r"\?|"
            r"\b(what|why|how|when|where|which|who|"
            r"can you|could you|do you|did you|any questions)\b",
            text,
            re.IGNORECASE,
        )
        question_count = len(questions)
        question_score = (
            1.0 if question_count >= 8
            else 0.65 if question_count >= 4
            else 0.3 if question_count >= 1
            else 0.0
        )
        add(
            "questions_and_interaction",
            question_score,
            f"Detected {question_count} question-like markers or phrases.",
            "Does not prove individual student interaction or student responses.",
        )
    else:
        for name in (
            "introduction_and_rapport",
            "questions_and_interaction",
            "homework_or_practice",
            "lesson_closure",
        ):
            add(name, None, "Transcript unavailable; run audio transcription.")

    # Visual indicators use the actual keys returned by src/video.py.
    lighting = visual.get("lighting")
    if isinstance(lighting, dict):
        brightness = lighting.get("average_frame_brightness")
        dark_ratio = lighting.get("dark_sampled_frame_ratio")
        if isinstance(brightness, (int, float)):
            dark_ratio = float(dark_ratio or 0)
            value = (
                1.0 if brightness >= 90
                else 0.8 if brightness >= 70
                else 0.6 if brightness >= 50
                else 0.35
            )
            if dark_ratio >= 0.6:
                value = min(value, 0.4)
            elif dark_ratio >= 0.3:
                value = min(value, 0.65)
            add(
                "lighting",
                value,
                f"Average brightness {brightness}; dark-frame ratio {dark_ratio:.1%}.",
                "Whole-frame brightness is not face-specific; thresholds are heuristic.",
            )
        else:
            add("lighting", None, "Brightness value unavailable.")
    else:
        add("lighting", None, "Lighting analysis unavailable.")

    black = visual.get("black_screen")
    if isinstance(black, dict):
        ratio = float(black.get("sampled_frame_ratio") or 0)
        count = int(black.get("sampled_frame_count") or 0)
        value = max(0.0, 1.0 - min(ratio * 10, 1.0))
        add(
            "black_screen",
            value,
            f"{count} sampled frame(s) flagged; ratio {ratio:.2%}.",
            "Check timestamps; transitions may be false positives.",
        )
    else:
        add("black_screen", None, "Black-screen analysis unavailable.")

    frozen = visual.get("frozen_frames")
    if isinstance(frozen, dict):
        intervals = frozen.get("intervals", [])

        candidate_seconds = sum(
            max(
                0.0,
                float(interval.get("duration_seconds") or 0),
            )
            for interval in intervals
            if isinstance(interval, dict)
        )

        add(
            "possible_freeze",
            None,
            (
                f"{len(intervals)} static-scene candidate interval(s); "
                f"{candidate_seconds:.1f}s flagged for review."
            ),
            (
                "Not included in the numerical score because "
                "video-only frame comparison cannot reliably "
                "distinguish a frozen feed from static teaching slides. "
                "Confirm manually before drawing conclusions."
            ),
        )
    else:
        add(
            "possible_freeze",
            None,
            "Freeze analysis unavailable.",
        )

    face = visual.get("face_visibility")
    if isinstance(face, dict):
        ratio = face.get("face_detection_ratio")
        detector_name = face.get("detector_used", "Face detector")  # Optional: reads YuNet
        if isinstance(ratio, (int, float)):
            # Low detection can reflect detector limitations, not absence of a teacher.
            value = (
                1.0 if ratio >= 0.7
                else 0.7 if ratio >= 0.4
                else 0.4 if ratio >= 0.15
                else 0.2
            )
            add(
                "face_visibility",
                value,
                f"Face detector fired in {ratio:.1%} of sampled frames.",
                "Approximate face-detection indicator; not proof that a teacher is absent.",
            )
        else:
            add("face_visibility", None, "Face detection ratio unavailable.")
    else:
        add("face_visibility", None, "Face visibility analysis unavailable.")

    assessed = [item for item in criteria if item["weighted_points"] is not None]
    assessed_weight = sum(item["weight"] for item in assessed)
    total_weight = sum(WEIGHTS.values())
    coverage = assessed_weight / total_weight

    score = (
        sum(item["weighted_points"] for item in assessed) / assessed_weight * 100
        if assessed_weight else None
    )

    # Do not issue a teaching rating from video-only signals.
    if not transcript_available or coverage < 0.70:
        rating = "Insufficient evidence"
    elif score < 40:
        rating = "Poor"
    elif score < 60:
        rating = "Average"
    elif score < 80:
        rating = "Good"
    else:
        rating = "Excellent"

    return {
        "score": None if score is None else round(score, 1),
        "rating": rating,
        "evidence_coverage_percent": round(coverage * 100, 1),
        "human_review_required": True,
        "criteria": criteria,
        "limitations": [
            "Heuristic prototype; not a validated teaching-quality measure.",
            "Transcript keyword checks can miss meaning and paraphrases.",
            "Visual indicators can produce false positives.",
            "Use this result to support human review, not as an automatic employment decision.",
        ],
    }
