### Prompt 1: Plan Project goals and requirement

**Tool:** Cluade sonnet 5.5
**Prompt:** The challenge
An online education company, 8x8, conducts classes for children. Some classes are excellent, some
good, some average, and others poor. We have a small set of recordings representing different levels
of quality.
Your challenge: Build an AI-powered application that can assess the quality of an online class
recording.
What makes one class better than another? Is it only what the teacher says, or could the way a class
sounds and what happens on screen also matter? Which signals are meaningful, and which might be
misleading? These are for you to explore. We are not prescribing what to examine or how to build the
solution.
What we will give you
A small collection of sample 8x8 class recordings to study and compare.
What we expect
Build a simple, working application that:
• Accepts a new class recording as input (upload or link).
• Classifies its quality as Poor, Average, Good, or Excellent.
• Explains its assessment with specific, useful observations rather than generic statements.
We will test your application with a recording you have not seen before. The system does not need to
be perfect, but it should genuinely process the input and produce a reasoned result.
Your approach is up to you
Choose any appropriate AI tools, models, APIs, frameworks, or coding assistants, subject to the
data-handling rules below. You do not need to train your own model or build a production-grade
platform. A simple interface is enough. We are interested in your judgement, experimentation, and
ability to make something work.
What to submit
• Working application: A hosted link where we can provide a recording and see the assessment.
A simple Streamlit, Gradio, or similar prototype is fine.
• Source code: A GitHub repository with a short README explaining how to run it. Do not include
secrets or credentials.
• AI interaction history: Relevant prompts and conversations from the AI assistants you used for
research, development, and debugging. Share exports or available logs; redact personal
information, credentials, and unrelated material.
• Brief explanation (maximum 300 words): Your chosen approach and why; what worked or
failed; the main limitations.
How we will evaluate your work
Evaluation area Weight
Quality and usefulness of the AI assessment 35%
Working application and code quality 25%
Problem-solving and effective use of AI 30%
Clarity of approach 10%
Ground rules
• Submit within the challenge window above; the assignment is intended to take approximately 4-5
hours.
• This is individual work. AI assistance is encouraged; having another person complete the
assignment is not.

HUMANITI AI • AI BUILD CHALLENGE 3
• Keep recordings confidential. Do not publish, redistribute, or expose them through public
applications or repositories. Process recordings only using services specifically permitted by the
organiser, especially if children are identifiable.
• Make sure your submitted application and links are accessible for testing.
A final note
At Humaniti AI, we work on real business problems where the answer is not always obvious. We are
looking for builders, researchers, and problem-solvers who can explore a problem, make sensible
decisions, and deliver a practical result.
We are not testing how much code you can write in five hours. We are testing what you can accomplish
with AI in five hours.

So my overview is that I'm going to build this with Gemini 3.1 pro like I did when I build photobe, plan with cluade and implement with Gemini, first I want to understand the requirement and AI models that are available and utilise all the free and opensource resources.
today I want to set all repo and streamlit, every step should clear so that I will explain it well like I did at Amazon's photobe project, and one thing I literally no idea about prompt history please let me know a little bit, and I'm not gonna be use Agentic coding instead I'd copy paste becuase I have to remember all the workflow and code structure, I know you got me

**Outcome:** Understanding the brief and setting up

1. Read this rule first

“Process recordings only using services specifically permitted by the organiser, especially if children are identifiable.”

This is the biggest trap in the brief. Sending class videos of children to the Gemini API (or any cloud API) may not be allowed. Before using any cloud AI on the recordings, email Anupam and ask something like:

“Could you confirm which AI services are permitted for processing the recordings (e.g. Gemini API, OpenAI, or only local/open-source models)?”

Save the reply for your AI history and the approach note. Until you hear back, design for the safe path:

Local/open-source processing on the recordings: Whisper, librosa, OpenCV.
Cloud LLMs only on derived text and numbers (transcript, metrics), and only if permitted. Even a transcript can contain children’s names, so it’s worth asking about.
Gemini for building the code (copy-paste workflow) is fine. That’s coding help, not recording processing.
The app must not store uploads: process, then delete.

Interpreting this carefully is part of your “judgement” score.

2. What the brief really asks
   Requirement What it means for you
   Input: upload or link File uploader (and maybe URL)
   Output: Poor / Average / Good / Excellent 4-class label
   Explain with specific observations Evidence like “Teacher spoke 78% of the time; no student responses in 40 min”
   Tested on an unseen recording Must generalize, not be tuned to the samples

The questions in the brief are hints: what the teacher says, how it sounds, what happens on screen, which signals are misleading. Your job is to explore these with the samples.

3. Signals to explore

Audio/speech (Whisper transcript + audio analysis):

Teacher vs student talk ratio, and number of student turns
Question count, and open-ended vs closed questions
Silence/dead air, long pauses, interruptions
Speech pace, tone variety (monotone vs engaged), volume
Praise and feedback language

Visual (OpenCV sampling a frame every few seconds):

Is the screen a slide, whiteboard, or just a face?
Screen activity: static for 20 minutes vs changing content
Whether the student appears on camera and is engaged (don’t identify anyone)

Technical quality:

Audio noise, echo, drops, video freezes

Misleading signals (mention these in your note; it shows maturity):

A long transcript doesn’t mean a good class.
A high teacher talk ratio can be fine for a young child’s explanation phase.
Silence can be thinking time, not a bad class.
Poor audio can reflect the student’s internet, not the teacher.
Fluent speech doesn’t equal good pedagogy. 4. Model options

Verify free-tier limits yourself, since they change often.

Task Options
Speech to text faster-whisper (open source, local, CPU-capable; small or base model)
Audio features librosa, webrtcvad/silero-vad (voice activity), pyannote (speaker diarization, but heavy and needs a HF token)
Video frames OpenCV, ffmpeg
Reasoning/scoring Local LLM via Ollama (too heavy to host free), or Gemini/other API if permitted
Classifier Rule-based scoring or a small scikit-learn model on your extracted features 5. Suggested architecture (simple and defensible)
Upload video
→ ffmpeg (extract audio + sample frames)
→ Feature extraction (talk ratio, silence, questions, frame changes...)
→ Whisper transcript
→ Scoring: rubric of weighted signals → Poor/Average/Good/Excellent
→ Explanation: evidence-backed observations
→ Delete temp files

Combine a transparent rubric with an LLM explanation layer. The rubric makes the label defensible, and the LLM (if permitted) writes the observations from the numbers, so it can’t hallucinate what it never saw.

6. Hosting reality check
   Streamlit Community Cloud: about 1 GB RAM, so Whisper plus video processing may crash it.
   Hugging Face Spaces (free CPU): about 16 GB RAM and Docker/Streamlit support, so it’s the better pick for this task. Streamlit apps run there fine.

Decision: build in Streamlit, host on Hugging Face Spaces, with Streamlit Cloud as a fallback. Test with a large file early, because video upload size limits and processing time are the real risks.

7. Today’s setup
   Create the GitHub repo: humaniti-class-quality-assessor, with README, Python .gitignore, and MIT license. Start it private and make it shareable later.
   Clone and set up locally:
   git clone <url>
   cd humaniti-class-quality-assessor
   python -m venv venv
   source venv/bin/activate # Windows: venv\Scripts\activate
   pip install streamlit
   Folder structure:
   app.py
   src/ (audio.py, video.py, scoring.py, explain.py)
   prompts/ (your AI history logs)
   docs/ (approach notes, experiments)
   data/ (gitignored: recordings go here, never committed)
   requirements.txt
   packages.txt (needed for ffmpeg on HF/Streamlit)
   Add to .gitignore: data/, _.mp4, _.mkv, \*.wav, .env, venv/, .streamlit/secrets.toml
   Hello world app.py with a file uploader that just shows the file name and size, then deploy it on Hugging Face Spaces to confirm the pipeline works end to end.
   Test ffmpeg on the host: add ffmpeg to packages.txt and check it runs. This is a common deployment failure.
8. Prompt history

It’s simply the record of what you asked the AI tools and what they replied. They want to see how you think and work with AI: how you break down the problem, correct the AI, debug, and iterate. That’s the 30% “Problem-solving and effective use of AI” score.

Good history looks like:

Clear, scoped prompts (“Write a function that computes teacher/student talk ratio from a Whisper segments list”)
You pushing back: “This crashed on long files, here’s the error”
Experiments: “I compared signals on 4 recordings, here are the results, what do they suggest?”
Decisions with reasons

Bad history: one giant “build the whole app” prompt, or no visible iteration.

How to capture it (since you’re copy-pasting):

Keep one chat per phase: Research, Design, Build, Debug.
After each session, paste the conversation into prompts/01_research.md, 02_design.md, and so on. Gemini and Claude both have a share/export option, but pasting into markdown is the most reliable.
Use this format per entry:

### Prompt 12: Fix ffmpeg error

**Tool:** Gemini 3.1 Pro
**Prompt:** ...
**Outcome:** Fixed by adding packages.txt
Redact before submitting: API keys, any names of children or teachers, and the recordings’ content. Don’t paste the confidential brief into the repo.

You can log this planning chat too. It’s good research history.

9. Your copy-paste workflow
   Claude (here): plan, architecture, and decisions.
   Gemini: generate each module, one file per prompt, giving it the folder structure and function signatures each time.
   You: paste it in, run it, and write down why it works, so you can explain every file in the interview.
   Rule: never paste code you can’t explain in one sentence.
