# Sarathi 🎙️

**Sarathi** ("साथी/सारथी" — companion, guide, charioteer in Nepali) is a
Nepali-language voice assistant that helps visually impaired users understand
what's around them. Point a camera at a scene — through a photo or live
video — and Sarathi detects nearby objects and describes them out loud in
Nepali, including which **direction** they're in (left / center / right) and
how **close** they are (near / far).

Built during a 48-hour hackathon with one guiding principle: **reliability
over ambition**. Every component is a pretrained model or a well-tested
library — no custom model training, no fragile LLM API dependency, no risk
of surprise failures mid-demo.

> *"बायाँतिर नजिकै एउटा मानिस र दायाँतिर टाढा एउटा कुर्सी छ।"*
> ("To the left, nearby, a person, and to the right, far away, a chair.")

---

## Team

Built at [ORCHID HACKX] by **Team Star**:
- **Sweta Aryal**
- **Dristi Shakya**
- **Purnima Pant**

---

## Why we built this

Nepal has very few accessible technology tools built specifically for
Nepali-speaking visually impaired users. Most existing assistive apps assume
English fluency or depend on paid, internet-heavy AI services that aren't
practical for everyday, low-cost use in Nepal. Sarathi is our attempt at a
lightweight, low-cost, reliable starting point — something that works with
just a camera and a laptop/phone, with no specialized hardware or paid
subscriptions required.

---

## Features

- **📷 Photo capture** — take a photo directly from the browser and get an
  instant spoken Nepali description of what's in it.
- **🖼️ Photo upload** — upload an existing image for the same analysis.
- **🔴 Live mode** — keep the camera running continuously; Sarathi
  automatically announces updated descriptions as the scene changes, with
  direction and distance called out for every object, without needing to
  take a photo each time.
- Everything runs from a single command (`streamlit run app.py`) — Live
  Mode is launched from a button inside the same app, so there's only one
  thing to run.

---

## How it works (pipeline)

```
 Camera / Photo
       |
       v
+----------------------+
|   YOLOv8 (COCO)       |  <- pretrained object detector, runs locally, no API
|   detects objects     |
+----------+------------+
           |  (label, bounding box)
           v
+----------------------+
| Direction + Distance   |  <- simple geometry: box position -> left/center/right
|   estimation           |     box size -> near/far
+----------+------------+
           |
           v
+----------------------+
| Nepali dictionary      |  <- hand-built English-to-Nepali object lookup table
| + sentence template    |     (no translation API, just a Python dict)
+----------+------------+
           |  Nepali sentence
           v
+----------------------+
|      gTTS              |  <- Google's free Text-to-Speech, converts the
|  (speech synthesis)    |     sentence to spoken Nepali audio
+----------+------------+
           |
           v
      Spoken aloud to the user
```

---

## Tech stack

| Component | Technology |
|---|---|
| Object detection | Ultralytics **YOLOv8** (pretrained on COCO, 80 classes) |
| Web interface | **Streamlit** |
| Speech synthesis | **gTTS** (Google Text-to-Speech) |
| Audio playback (live mode) | **pygame** |
| Camera / image handling | **OpenCV** |
| Language | **Python 3** |
| Nepali object dictionary | Custom hand-built Python dictionary |

---

## Project structure

```
sarathi/
├── app.py            # Streamlit UI - camera/upload input + Live Mode launcher
├── live_mode.py        # Real-time continuous detection + speech loop
├── detector.py         # YOLO detection + sentence building + TTS pipeline
├── nepali_dict.py       # English-to-Nepali object dictionary + sentence templates
├── requirements.txt     # Python dependencies
└── README.md
```

---

## Setup

```bash
git clone https://github.com/purnimapant77/shayatri.git
cd sarathi

python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# Install PyTorch CPU build first (avoids GPU/CPU mismatch issues)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install the rest
pip install -r requirements.txt
```

> First run auto-downloads `yolov8n.pt` (~6MB) from Ultralytics — needs
> internet, happens once.

## Run

```bash
streamlit run app.py
```

Opens in your browser (usually `http://localhost:8501`). Use the photo
tabs, or click **"Start Live Mode"** to open the real-time detection window
(a separate window, not embedded in the browser — see technical notes below).

---

## Language support

Sarathi is currently built **specifically for Nepali speakers** — every
sentence, direction word, and distance word is in Nepali, and this was a
deliberate scope decision, not an oversight. Most visually-impaired
assistive tools are built with English as the default and treat other
languages as an afterthought; we wanted to build for a specific
underserved language community first, rather than a generic multilingual
tool that serves no one particularly well.

**Going forward**, the architecture is designed to make this extensible:
the object-detection layer is language-agnostic (YOLO doesn't care what
language you speak), and the Nepali dictionary + sentence templates are
isolated in a single file (`nepali_dict.py`). Adding another specific
language community — e.g. Newari, Maithili, or other underserved regional
languages — would mean writing a new dictionary + template file, not
rebuilding the whole pipeline.

---

## Limitations

- **Distance and direction are simple geometry estimates**, not true depth
  sensing or spatial audio. Distance is inferred from how large an object's
  bounding box appears (bigger = closer); direction is inferred from where
  its center falls in the frame. This is good enough for a working demo,
  but not precise enough for safety-critical, real-world navigation yet.
- **Limited object vocabulary** — YOLOv8 recognizes 80 general object
  categories (people, furniture, vehicles, electronics, animals, food,
  etc.), all from the COCO dataset. It has no concept of Nepal-specific
  obstacles that aren't in COCO (e.g. certain street infrastructure, local
  signage, uneven terrain).
- **Requires internet for speech synthesis.** Object detection is fully
  offline, but gTTS needs a connection the first time each sentence is
  spoken. Live mode caches spoken sentences to reduce repeated network
  calls, but a fully offline fallback (with real Nepali voice quality) is
  still an open problem — most offline TTS engines (like pyttsx3/espeak)
  have poor or missing Nepali support.
- **Single-language support**, by design for now — see "Language support"
  above.
- **Desktop/laptop dependent** — currently requires a laptop or phone
  running the app; not yet a standalone wearable device.

---

## Challenges we faced

- **Environment/dependency conflicts eating hackathon time** — mismatched
  `torch`/`torchvision` builds (CPU vs. CUDA versions installed separately)
  caused a `torchvision::nms does not exist` error that took real debugging
  time to trace back to a build mismatch, not a code bug.
- **Disk space constraints** — PyTorch and its dependencies are large
  (500MB+ for GPU builds), and low free disk space repeatedly interrupted
  installs mid-hackathon. Solved by switching to CPU-only PyTorch builds.
- **Offline speech synthesis doesn't support Nepali well** — we initially
  tried `pyttsx3` for real-time speech (so live mode wouldn't depend on
  internet), but its underlying engines (espeak on Linux, SAPI5 on Windows)
  have poor or missing Nepali voice support and failed silently with no
  error message, which was hard to diagnose. We switched to gTTS (same
  engine as our photo mode) with local caching as the fix.
- **Real-time video in-browser vs. reliability trade-off** — true
  browser-embedded live video (via `streamlit-webrtc`) requires the
  `aiortc` library, which is fragile to install under time pressure. We
  chose a simpler, more reliable approach: a native camera window launched
  from a button inside the web app, trading a bit of visual polish for
  guaranteed functionality during the live demo.
- **Directional/distance accuracy tuning** — getting the near/far and
  left/center/right thresholds to feel natural (not too sensitive, not too
  sluggish) took several rounds of testing with real camera input rather
  than just guessing at numbers.

---

## Future improvements

- **Wearable integration — blind stick or smart glasses.** The most
  important next step is moving Sarathi off a laptop/phone screen and onto
  a physical assistive device. Plan: mount a small camera module (e.g. an
  **ESP32-CAM**) onto a walking stick or a pair of glasses, stream its
  camera feed to a lightweight backend running the same detection
  pipeline, and output audio through a small onboard speaker or
  bone-conduction headphones. This would make the experience fully
  hands-free, with no phone or laptop needed at all.
- **True offline speech synthesis in Nepali**, removing the internet
  dependency entirely — likely requires a dedicated Nepali TTS model
  rather than general-purpose offline engines.
- **Better distance estimation** using stereo cameras or a depth sensor
  instead of bounding-box-size heuristics, for more reliable real-world
  navigation guidance.
- **Expanding language support** to other underserved regional languages
  (see "Language support" above), once the Nepali experience is solid.
- **Obstacle types beyond COCO's 80 classes** — training or fine-tuning on
  Nepal-specific obstacles (potholes, uneven pavement, specific signage)
  rather than relying solely on general pretrained categories.
- **Battery-efficient, low-power hardware** so a wearable version can run
  a full day without needing to be recharged mid-use.

---

## 🙏 Acknowledgements

This project was built in a short, intense span of time by Team STAR — Dristi Shakya, Sweta Aryal, and Purnima Pant. We learned a lot along the way, from debugging real-time systems to grounding an LLM's responses in truth. Our hope is that ideas like Sarathi can, in some small way, make the world a little more accessible.

Thank you to everyone who supported and mentored us throughout this hackathon.

**सारथी — तपाईंको साथी, हरेक कदममा।**
*(Sarathi — your companion, every step of the way.)*
