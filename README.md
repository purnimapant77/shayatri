# Sahayatri 🎙️

**Sahayatri** ("companion/guide" in Nepali) is a Nepali-language voice assistant 
that helps visually impaired users understand what's around them. Point a 
camera at a scene — via photo or live video — and it detects nearby objects, 
then describes them out loud in Nepali, including which **direction** they're 
in (left / center / right) and how **close** they are (near / far).

Built during a hackathon with a focus on reliability: every component is a 
pretrained model or well-tested library, so there's no custom training, no 
fragile LLM API dependency, and no risk of surprise failures during a live demo.

## Example

Point the camera at a room and hear:

> *"बायाँतिर नजिकै एउटा मानिस र दायाँतिर टाढा एउटा कुर्सी छ।"*
> ("To the left, nearby, a person, and to the right, far away, a chair.")

## How it works

1. **YOLOv8** (pretrained on COCO, 80 object classes) detects objects in the frame.
2. Each object's position and bounding-box size are converted into a 
   **direction** (बायाँतिर / अगाडि / दायाँतिर) and **distance** (नजिकै / टाढा).
3. A template builds a natural Nepali sentence describing the scene.
4. **gTTS** (Google Text-to-Speech) converts the sentence into spoken Nepali 
   audio. Live mode caches every spoken sentence to disk so repeated phrases 
   play back instantly without waiting on the network again.

## Features

- **📷 Photo capture / 🖼️ Photo upload** — take or upload a single image, get 
  an instant spoken Nepali description.
- **🔴 Live mode** — keep the camera running continuously; it automatically 
  announces updated descriptions as the scene changes, with direction and 
  distance for every object.
- Runs entirely from one command (`streamlit run app.py`) — live mode is 
  launched from a button inside the same app.

## Tech stack

Python · Streamlit (web UI) · Ultralytics YOLOv8 (object detection) · gTTS 
(speech synthesis) · OpenCV (camera/image handling) · pygame (audio playback)

## Project structure
