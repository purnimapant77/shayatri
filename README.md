# Khoji Lite 🎙️

A Nepali-language voice assistant that helps visually impaired users understand
what's around them. Take or upload a photo → objects are detected → a Nepali
sentence describing the scene is generated and spoken aloud.

Built for a hackathon in a time crunch — no LLM API keys, no Ollama, no fragile
network dependencies. Just a pretrained YOLOv8 model + simple rules + Google TTS.

## How it works

1. **YOLOv8n** (pretrained on COCO, 80 object classes) detects objects in the photo.
2. Each object is mapped to a **Nepali noun**, and a rough **distance zone**
   (नजिकै / अगाडि / टाढा) is estimated from how large the object's box is
   relative to the frame.
3. A template builds a natural Nepali sentence, e.g.
   *"अगाडि एउटा कुर्सी र नजिकै एउटा मानिस छ।"*
4. **gTTS** converts the sentence to speech, played back in the browser.

## Project structure

```
khoji-lite/
├── app.py            # Streamlit UI — camera/upload input, shows result + plays audio
├── detector.py        # YOLO detection + sentence building + TTS pipeline
├── nepali_dict.py      # English→Nepali object dictionary + sentence templates
├── requirements.txt
└── README.md
```

## Setup (run this first, ~5 minutes)

```bash
# 1. Clone your repo (after you've pushed it, or just cd into this folder)
cd khoji-lite

# 2. Create a virtual environment (recommended but optional)
python3 -m venv venv
source venv/bin/activate        # on Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

> First run will auto-download `yolov8n.pt` (~6MB) — needs internet, happens once.

## Run the app

```bash
streamlit run app.py
```

This single command gives you everything: photo capture, photo upload, AND
live real-time mode (via a button inside the app — see below). It'll open in
your browser automatically (usually `http://localhost:8501`).

## 🔴 Live mode — real-time continuous detection

Everything runs from **one command** — `streamlit run app.py` — including
live mode. Inside the app there are three tabs: photo capture, photo upload,
and a **"लाइभ मोड (Real-time)"** tab with a "Start Live Mode" button.

Clicking that button opens a **separate window** (not inside the browser)
showing your live camera feed with continuous detection. This window speaks
Nepali sentences describing what's around, including **direction** (बायाँतिर /
अगाडि / दायाँतिर — left / center / right) and **distance** (नजिकै / टाढा —
near / far), automatically, with no photo-taking needed.

**Why a separate window instead of embedded in the browser?** Browsers can't
run continuous Python-side video processing without a heavy extra dependency
(`streamlit-webrtc` + `aiortc`), which is notoriously fragile to install right
before a demo. Launching a plain OpenCV window from a button click gets you
the same real-time experience with zero extra risk — you still only ever run
one command to start everything.

- Press **`q`** inside that live window to close it.
- Uses **gTTS** — the same speech engine as photo mode — instead of an
  offline engine, because offline engines (pyttsx3/espeak on Linux,
  pyttsx3/SAPI5 on Windows) generally have poor or missing Nepali voice
  support and can fail to produce sound with no error at all. gTTS needs
  internet, but every spoken sentence is **cached to disk** (in a
  `tts_cache/` folder, keyed by a hash of the sentence text), so repeated
  announcements — very common in live mode, e.g. "chair ahead" said again a
  minute later — play back instantly with no new network call.
- Requires a stable internet connection the first time each distinct
  sentence is spoken. If your venue's wifi is unreliable, do a "priming run"
  before your demo: point the camera at all the objects you plan to show, so
  their sentences get cached in advance and play back instantly during the
  actual demo even if wifi hiccups.
- **Don't** use the photo-capture tab's camera at the same time as live mode
  — only one process can use the webcam at once. Close the camera tab (or
  don't grant it camera permission) before starting live mode.

### Tuning live mode
Open `live_mode.py` and adjust these constants near the top if needed:
- `CONF_THRESHOLD` — raise it if it's detecting too many false objects, lower
  it if it's missing real objects.
- `SPEAK_COOLDOWN_SECONDS` — minimum gap between spoken sentences (default 2s).
  Lower it for more frequent updates, raise it if it feels too chatty.

### Troubleshooting: "it detects but doesn't speak"
Check your terminal while live mode is running — every sentence it decides
to speak is printed as `[speech] speaking: ...`. If you see that line but
hear no audio, check:
- Internet connection (gTTS needs it for any sentence not already cached)
- Your system volume / correct audio output device is selected
- If you see `[speech error] ...`, the printed message will usually say why
  (most often a network issue)

## Testing the pipeline without the UI (useful for debugging)

```bash
python3 detector.py path/to/some_photo.jpg
```
This prints the Nepali sentence and saves `test_output.mp3` + `test_annotated.jpg`
so you can check detection quality before doing a live demo.

## Requirements for the demo

- Internet connection (needed for gTTS — used for both photo mode and live
  mode). Live mode caches sentences after their first use, so a brief wifi
  hiccup mid-demo won't break already-spoken phrases.
- A webcam (built-in laptop cam is fine) or just test images ready on disk.

## Known limitations (mention these honestly in your pitch — judges respect this)

- Distance estimation (नजिकै/अगाडि/टाढा) is a simple bounding-box-size heuristic,
  not real depth sensing — good enough for a demo, not production-grade.
- Only recognizes YOLOv8's 80 COCO classes (common objects: people, vehicles,
  furniture, animals, food, electronics). Won't recognize Nepal-specific objects
  not in COCO (e.g. specific street signs) — a good "future work" talking point.
- gTTS requires internet at runtime (no API key needed, but needs network access).

## Pushing to GitHub

```bash
git init
git add .
git commit -m "Khoji Lite: Nepali voice assistant for visually impaired users"
git branch -M main
git remote add origin https://github.com/<your-username>/khoji-lite.git
git push -u origin main
```

## Demo script (30-45 seconds, practice this)

1. "Visually impaired people in Nepal have almost no accessible tools in their
   own language. Khoji Lite gives them a voice-based description of their
   surroundings, entirely in Nepali."
2. Open camera tab, point at a few objects on the table (phone, cup, bottle).
3. Take the photo — sentence appears + Nepali audio plays automatically.
4. "This uses a pretrained YOLO model, so there's zero training data needed
   and zero risk of API failures during the demo — everything runs locally
   except the final speech synthesis."
