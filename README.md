---
title: Class Quality Assessor
emoji: 🎓
colorFrom: blue
colorTo: yellow
sdk: docker
app_port: 8501
pinned: false
---

# 🎓 Class Quality Assessor

**A local-first prototype for analyzing recorded lessons using speech transcription, visual indicators, and experimental machine learning.**

Class Quality Assessor is a Streamlit application that helps reviewers inspect recorded classes by extracting speech, generating timestamped transcripts, and identifying potentially relevant visual indicators such as black screens, low lighting, near-identical frames, and approximate face visibility.

The goal is to support **consistent, evidence-based human review** of recorded lessons—not to replace educators or automatically determine teaching quality.

> **Project status:** Experimental prototype. The heuristic scoring system is not validated, and the current ML model was trained on illustrative synthetic data. Do not use its scores or predictions for consequential decisions about teachers or students.

## Table of Contents

- [Problem](#problem)
- [Approach](#approach)
- [Features](#features)
- [Technology Stack](#technology-stack)
- [Architecture](#architecture)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [Training the ML Model](#training-the-ml-model)
- [Dataset Format](#dataset-format)
- [Privacy and Data Handling](#privacy-and-data-handling)
- [Limitations](#limitations)
- [Roadmap](#roadmap)
- [Contributing](#contributing)
- [License](#license)

## Problem

Reviewing recorded lessons manually can be time-consuming and inconsistent, especially when reviewers need to inspect many recordings.

Several review tasks can be partially assisted by software:

- Extracting spoken content and timestamps from recordings.
- Identifying lesson segments that may contain introductions, questions, assignments, or closures.
- Detecting potential recording issues, including black screens, low brightness, or long periods of near-identical frames.
- Summarizing available evidence to help reviewers focus their attention.

However, observable recording indicators do not necessarily measure teaching effectiveness. A lecture can be effective without frequent questions or visible faces, and a technically clear video does not guarantee good instruction.

This project explores how automated indicators can make the review process more efficient while keeping human judgment central.

## Approach

The application combines three components.

### 1. Audio analysis

The audio pipeline extracts audio from an MP4 recording when necessary and uses Faster-Whisper to produce a transcript with timestamped segments and detected-language information.

The transcript can be inspected for textual indicators such as introductions, questions, homework instructions, and lesson closures.

### 2. Video analysis

The video pipeline uses OpenCV to sample frames and calculate visual indicators, including:

- Black-screen samples.
- Average brightness and the proportion of dark samples.
- Potential frozen or near-identical frame intervals.
- Approximate face-detection frequency using a YuNet model.

These are review indicators, not definitive findings. For example, a static slide may be intentional, and face detection does not establish student engagement or teaching quality.

### 3. Assessment and experimental ML

The combined assessment brings the audio and video outputs together.

- **Heuristic assessment:** applies predefined rules and weights to available evidence.
- **Experimental ML prediction:** uses a scikit-learn classification pipeline trained on labeled examples to predict a category.
- **Evidence inspection:** displays the features and criteria behind the assessment.
- **Export:** allows the combined report to be downloaded as JSON.

The ML model currently uses synthetic illustrative training examples. Its predictions are therefore demonstrations of the pipeline, not validated estimates of real-world lesson quality.

## Features

- Local Streamlit interface.
- MP4 audio extraction and transcription.
- WAV transcription support.
- Timestamped transcript segments.
- Video metadata and sampled-frame analysis.
- Black-screen, lighting, near-identical-frame, and approximate face-detection indicators.
- Rule-based assessment with evidence coverage.
- Experimental scikit-learn classification.
- Downloadable video-analysis and combined-assessment JSON reports.
- Human-review warnings.

## Technology Stack

| Component              | Technology       |
| ---------------------- | ---------------- |
| User interface         | Streamlit        |
| Audio transcription    | Faster-Whisper   |
| Audio/video processing | FFmpeg, PyAV     |
| Video analysis         | OpenCV           |
| Face detection         | YuNet ONNX model |
| Data processing        | pandas           |
| Machine learning       | scikit-learn     |
| Model persistence      | joblib           |
| Language               | Python           |

## Architecture

```text
                    ┌──────────────────────┐
                    │   Streamlit UI       │
                    │  MP4 / WAV upload    │
                    └──────────┬───────────┘
                               │
                   ┌───────────┴───────────┐
                   │                       │
           ┌───────▼────────┐      ┌───────▼────────┐
           │  Audio pipeline│      │ Video pipeline │
           │                │      │                │
           │ FFmpeg / PyAV  │      │ OpenCV         │
           │ Faster-Whisper │      │ YuNet / ONNX   │
           └───────┬────────┘      └───────┬────────┘
                   │                       │
                   ▼                       ▼
             Transcript              Visual indicators
                   │                       │
                   └───────────┬───────────┘
                               ▼
                    ┌──────────────────────┐
                    │ Combined assessment  │
                    │                      │
                    │ Heuristic scoring    │
                    │ ML feature extraction│
                    │ Experimental model   │
                    └──────────┬───────────┘
                               ▼
                    ┌──────────────────────┐
                    │ Results and evidence  │
                    │ JSON report export    │
                    └──────────────────────┘
```

### Project structure

```text
class-quality-assessor/
├── app.py
├── requirements.txt
├── src/
│   ├── audio.py
│   ├── video.py
│   ├── scoring.py
│   ├── ml_model.py
│   └── explain.py
├── models/
│   └── face_detection_yunet_2023mar.onnx
├── data/
│   └── lesson_quality_labels.csv
└── README.md
```

`models/lesson_quality_model.joblib` is generated when the ML model is trained. Other generated files, such as extracted audio and analysis reports, may also be created during execution.

This is a logical overview; the exact contents of `data/` and `models/` can vary depending on which files have been generated locally.

## Getting Started

### Prerequisites

Install the following before running the application:

- Python 3.11 or newer; Python 3.13 has been used during development.
- Git.
- FFmpeg, available from your terminal through `PATH`.
- The YuNet ONNX model at `models/face_detection_yunet_2023mar.onnx`.

You will also need an internet connection when installing dependencies or downloading required model files. Once the dependencies and model files are present, the processing pipelines are designed to run locally.

### 1. Clone the repository

Replace the placeholder with the URL of your repository.

```bash
git clone <repository-url>
cd class-quality-assessor
```

### 2. Create a virtual environment

**Windows — PowerShell**

```powershell
py -3.13 -m venv venv
.\venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, you can run the environment's Python executable directly instead of changing your execution policy:

```powershell
.\venv\Scripts\python.exe --version
```

**Linux / macOS**

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Python dependencies

Upgrade pip and install the pinned project dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The project currently pins its main dependencies in `requirements.txt`, including Streamlit, Faster-Whisper, PyAV, and OpenCV.

### 4. Install FFmpeg

FFmpeg must be installed separately from the Python packages.

**Windows — WinGet**

```powershell
winget install --id Gyan.FFmpeg -e
```

Open a new terminal after installation and verify:

```powershell
where.exe ffmpeg
ffmpeg -version
```

If WinGet reports that FFmpeg is already installed but the command is not found, locate `ffmpeg.exe` and add its containing `bin` directory to your user `PATH`.

**macOS — Homebrew**

```bash
brew install ffmpeg
ffmpeg -version
```

**Ubuntu / Debian**

```bash
sudo apt update
sudo apt install ffmpeg
ffmpeg -version
```

### 5. Verify the YuNet model

Check that this file exists:

```text
models/face_detection_yunet_2023mar.onnx
```

If it is missing, obtain the corresponding YuNet face-detection model from the official OpenCV Zoo repository and place it at that path. The video pipeline may not be able to perform face detection without it.

OpenCV Zoo: https://github.com/opencv/opencv_zoo

### 6. Prepare the local data directory

The application creates the `data/` directory automatically. It is used for staged recordings and generated artifacts.

If you want to run the combined assessment, you also need a trained ML model at:

```text
models/lesson_quality_model.joblib
```

The training instructions below explain how to generate it.

### 7. Launch the application

```bash
python -m streamlit run app.py
```

Open the local URL printed in the terminal, usually:

http://localhost:8501

If you are not using an activated virtual environment on Windows, run:

```powershell
.\venv\Scripts\python.exe -m streamlit run app.py
```

## Usage

### Audio analysis

1. Upload an MP4 or WAV file.
2. Open the **Audio analysis** tab.
3. Click **Run Audio Pipeline**.
4. Review the detected language, transcript, and timestamped segments.

### Video analysis

1. Upload an MP4 file.
2. Open the **Video analysis** tab.
3. Choose a sampling interval.
4. Click **Run Video Pipeline**.
5. Review the visual indicators and download the JSON report.

A smaller sampling interval analyzes more frames and can increase processing time.

### Combined assessment

1. Upload an MP4 file.
2. Ensure the trained ML model exists.
3. Open **Combined assessment**.
4. Click **Run Combined Assessment**.
5. Review the heuristic score, evidence coverage, experimental ML prediction, and model features.
6. Download the combined assessment JSON.

The combined workflow runs audio transcription and video analysis again. It can therefore take longer than reviewing previously generated outputs.

## Training the ML Model

The classifier is an experimental pipeline based on human-labeled lesson examples.

Prepare a CSV containing the features and a target column named `human_label`. See [Dataset Format](#dataset-format).

Run training from the repository root:

```powershell
.\venv\Scripts\python.exe -c "from src.ml_model import train_model; print(train_model('data/lesson_quality_labels.csv'))"
```

On Linux or macOS, with the virtual environment activated:

```bash
python -c "from src.ml_model import train_model; print(train_model('data/lesson_quality_labels.csv'))"
```

The training function checks the dataset, splits it into training and test sets, fits a preprocessing and classification pipeline, reports evaluation metrics, and saves the model to:

```text
models/lesson_quality_model.joblib
```

The initial implementation requires at least 20 usable labeled examples and at least two classes, with enough examples per class for a stratified split. This is only a technical minimum—not a sufficient dataset size for a trustworthy model.

**Important:** do not treat the evaluation metrics from a tiny or synthetic dataset as evidence of real-world accuracy. A useful evaluation requires representative real lessons, reliable labels, a leakage-resistant split, and testing on independent recordings.

## Dataset Format

The expected CSV columns are:

```csv
duration_minutes,introduction_present,question_marker_count,homework_present,closure_present,average_brightness,dark_frame_ratio,black_frame_ratio,face_detection_ratio,human_label
```

Feature meanings:

| Feature                 | Description                                             |
| ----------------------- | ------------------------------------------------------- |
| `duration_minutes`      | Lesson duration in minutes                              |
| `introduction_present`  | Whether an introductory phrase was detected             |
| `question_marker_count` | Number of detected question markers or question phrases |
| `homework_present`      | Whether homework-related language was detected          |
| `closure_present`       | Whether a closing phrase was detected                   |
| `average_brightness`    | Average sampled-frame brightness                        |
| `dark_frame_ratio`      | Proportion of sampled frames considered dark            |
| `black_frame_ratio`     | Proportion of sampled frames considered black           |
| `face_detection_ratio`  | Proportion of sampled frames with a detected face       |
| `human_label`           | Human-assigned category used as the prediction target   |

The exact labeling policy should be documented before collecting real training data. Labels should reflect a consistent rubric rather than assumptions that a particular visual or speaking pattern automatically indicates good or poor teaching.

The CSV schema alone does not make a dataset valid. Use consented, appropriately protected data and establish a review process for labeling disagreements.

## Privacy and Data Handling

The application is designed for local-first processing:

- Audio transcription and video analysis run through local processing libraries and models.
- The current application does not require an external AI API for its main analysis pipelines.
- Uploaded recordings are staged in the local `data/` directory.
- Generated reports can contain transcript-derived and video-derived information.

**Local processing does not automatically mean all data is deleted or completely isolated.** Staged recordings, extracted audio, generated reports, model files, and application logs may persist on disk.

Before using recordings containing students, teachers, or other identifiable people:

- Obtain the necessary permissions and follow applicable privacy requirements.
- Restrict access to the machine and project directory.
- Avoid committing recordings, transcripts, or personally identifiable information to Git.
- Establish retention and deletion procedures.
- Review any future integrations that might transmit data externally.

## Limitations

This is a prototype and has several important limitations.

**Heuristic scoring is not validated.** The scoring weights and textual rules are predefined design choices, not empirically established measures of instructional quality.

**The ML model is experimental.** The current illustrative synthetic training data cannot establish predictive performance on real lessons.

**Speech analysis can be imperfect.** Transcription errors, accents, background noise, language choice, and technical vocabulary can affect extracted features.

**Text patterns are context-limited.** A question may not contain a question mark, and an effective introduction or closure may use language not covered by the current patterns.

**Video indicators are not semantic understanding.** Darkness, a static slide, or an undetected face does not necessarily indicate a recording defect or poor teaching.

**Sampling can miss events.** A short black screen or brief visual change may occur between sampled frames.

**Face detection is not engagement detection.** A detected face does not prove attention, comprehension, or participation.

**Generalization is unknown.** Performance may differ across subjects, languages, teaching styles, camera setups, lighting conditions, and classroom formats.

**Human review remains necessary.** Automated scores and categories should not be used alone for teacher evaluation, employment decisions, student grading, or other high-impact outcomes.

## Roadmap

The long-term goal is to evolve the prototype into a reliable, scalable, privacy-conscious lesson-review platform. The following stages describe a proposed direction, not features that are already implemented.

### Phase 1 — Reliable local prototype

- [x] Local audio transcription.
- [x] Video metadata and sampled-frame analysis.
- [x] Rule-based assessment prototype.
- [x] Experimental ML training and inference.
- [x] JSON report export.
- [ ] Add automated tests for audio, video, scoring, and ML feature extraction.
- [ ] Improve error messages and input validation.
- [ ] Prevent stale analysis when different uploads have the same filename and size.
- [ ] Add configurable model paths and application settings.

### Phase 2 — Trustworthy evaluation

- [ ] Define a transparent, educator-reviewed assessment rubric.
- [ ] Collect consented, representative, human-labeled lesson data.
- [ ] Measure inter-reviewer agreement and document ambiguous cases.
- [ ] Replace synthetic training data with a properly governed dataset.
- [ ] Evaluate class imbalance, per-class performance, and calibration where appropriate.
- [ ] Test on held-out recordings from different lessons and teaching contexts.
- [ ] Compare automated outputs with independent human review.

### Phase 3 — Better evidence and usability

- [ ] Add timestamped evidence for detected events.
- [ ] Improve confidence and uncertainty reporting.
- [ ] Support configurable assessment criteria by teaching context.
- [ ] Add structured report summaries and reviewer feedback.
- [ ] Improve accessibility and multilingual support.
- [ ] Provide clear explanations of missing or unreliable evidence.

### Phase 4 — Scale and operational readiness

- [ ] Separate processing logic from the Streamlit interface.
- [ ] Introduce a versioned API for integrations with learning-management systems.
- [ ] Add background jobs and queues for long recordings.
- [ ] Store metadata and reports using a documented persistence layer.
- [ ] Add authentication, role-based access, and audit logs where needed.
- [ ] Implement configurable retention, deletion, and access controls.
- [ ] Add observability, resource limits, and reproducible deployments.
- [ ] Benchmark processing speed, memory usage, and cost per recorded hour.
- [ ] Evaluate deployment options for on-premises and appropriately secured cloud environments.

### Scale objective

The intended direction is a system that can process large volumes of lesson recordings, return traceable evidence to reviewers, integrate with existing education platforms, and maintain privacy and reliability as usage grows.

No throughput, accuracy, or production-readiness target should be claimed until it has been defined and measured through representative benchmarks.

## Contributing

Contributions, bug reports, and suggestions are welcome.

1. Fork the repository.
2. Create a feature branch.
3. Make a focused change.
4. Run the application and relevant checks.
5. Open a pull request explaining the change and how it was tested.

For a basic syntax check:

```powershell
.\venv\Scripts\python.exe -m py_compile app.py src\audio.py src\video.py src\scoring.py src\ml_model.py
```

On Linux or macOS, with the environment activated:

```bash
python -m py_compile app.py src/audio.py src/video.py src/scoring.py src/ml_model.py
```

This command checks syntax only; it does not replace automated tests or end-to-end validation.

Please avoid including real recordings, transcripts, credentials, or other sensitive data in issues and pull requests.

## License

MIT License

---

**Built to make lesson review more systematic, transparent, and evidence-driven—with human judgment at the center.**
