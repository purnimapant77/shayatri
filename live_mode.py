"""
Khoji Lite — LIVE MODE
Continuously reads webcam frames, detects objects with YOLOv8, and speaks
Nepali sentences describing what's around (direction: left/center/right,
distance: near/far), out loud.

Speech uses gTTS (same engine as photo mode) instead of pyttsx3, because
pyttsx3's offline engines (espeak on Linux, SAPI5 on Windows) generally have
poor or missing Nepali voice support and can fail silently. gTTS needs
internet, but generated audio is cached to disk by sentence text, so
repeated phrases (very common in live mode) play back instantly without
re-hitting the network.

Run with: python3 live_mode.py
Press 'q' in the video window to quit.
"""
import os
import time
import hashlib
import threading
import queue

import cv2
from ultralytics import YOLO
from gtts import gTTS
import pygame

from nepali_dict import direction_zone, distance_zone, build_sentence_with_direction

# English glosses for on-screen display only (OpenCV can't render Devanagari)
_DIRECTION_EN = {"बायाँतिर": "left", "अगाडि": "center", "दायाँतिर": "right"}
_DISTANCE_EN = {"नजिकै": "near", "टाढा": "far"}

# ---------------------------------------------------------------------------
# Config — tune these if the demo feels too chatty/slow on your machine
# ---------------------------------------------------------------------------
CONF_THRESHOLD = 0.45       # higher = fewer false positives, good for live use
SPEAK_COOLDOWN_SECONDS = 2.0  # minimum gap between spoken sentences
CAMERA_INDEX = 0             # change to 1 if you have multiple cameras

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tts_cache")
os.makedirs(CACHE_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Background speech worker — runs in its own thread so speaking never
# freezes the video feed. Only ever speaks the MOST RECENT sentence queued;
# older pending sentences are dropped so audio never lags behind reality.
#
# Uses gTTS to synthesize Nepali speech (needs internet) and pygame to play
# it back. Each sentence is cached to disk by an md5 hash of its text, so
# repeated announcements (the common case in live mode -- e.g. "chair ahead"
# said again a minute later) play back instantly without a new network call.
# ---------------------------------------------------------------------------
class SpeechWorker:
    def __init__(self):
        self._queue = queue.Queue()
        pygame.mixer.init()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        print("[speech] gTTS + pygame speech worker ready.")

    def say(self, text: str):
        # Drop any stale pending sentence, keep only the newest
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break
        self._queue.put(text)

    def _cache_path(self, text: str) -> str:
        digest = hashlib.md5(text.encode("utf-8")).hexdigest()
        return os.path.join(CACHE_DIR, f"{digest}.mp3")

    def _synthesize(self, text: str) -> str:
        """Return path to an mp3 file for this text, generating + caching if needed."""
        path = self._cache_path(text)
        if not os.path.exists(path):
            tts = gTTS(text=text, lang="ne")
            tts.save(path)
        return path

    def _run(self):
        while True:
            text = self._queue.get()  # blocks until something is queued
            try:
                path = self._synthesize(text)
                print(f"[speech] speaking: {text}")
                pygame.mixer.music.load(path)
                pygame.mixer.music.play()
                while pygame.mixer.music.get_busy():
                    time.sleep(0.1)
            except Exception as e:
                print(f"[speech error] {e} -- check internet connection (gTTS needs it)")


def get_detections(result, frame_width, frame_height, model_names):
    """Extract (label, direction, distance) for every box in a YOLO result."""
    detections = []
    for box in result.boxes:
        cls_id = int(box.cls[0])
        label = model_names[cls_id]

        x1, y1, x2, y2 = box.xyxy[0].tolist()
        center_x_ratio = ((x1 + x2) / 2) / frame_width
        height_ratio = (y2 - y1) / frame_height

        detections.append({
            "label": label,
            "direction": direction_zone(center_x_ratio),
            "distance": distance_zone(height_ratio),
        })
    return detections


def main():
    print("Loading YOLOv8n model...")
    model = YOLO("yolov8n.pt")
    print("Model loaded. Opening camera...")

    speaker = SpeechWorker()

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("ERROR: Could not open camera. Check CAMERA_INDEX or that no other")
        print("app (like the Streamlit app) is currently using the webcam.")
        return

    last_spoken_signature = None
    last_speak_time = 0.0
    last_status_line = "Scanning..."

    print("Live mode running. Press 'q' in the video window to quit.")

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Failed to read frame from camera.")
            break

        frame_h, frame_w = frame.shape[:2]
        results = model(frame, conf=CONF_THRESHOLD, verbose=False)
        result = results[0]

        detections = get_detections(result, frame_w, frame_h, model.names)

        # Build a simple signature of the current scene to detect real change
        signature = tuple(sorted(
            (d["label"], d["direction"], d["distance"]) for d in detections
        ))

        now = time.time()
        scene_changed = signature != last_spoken_signature
        cooldown_passed = (now - last_speak_time) >= SPEAK_COOLDOWN_SECONDS

        if detections and scene_changed and cooldown_passed:
            sentence = build_sentence_with_direction(detections)
            print("Speaking:", sentence)
            speaker.say(sentence)
            last_spoken_signature = signature
            last_speak_time = now
            # English summary for the on-screen overlay (OpenCV can't render
            # Devanagari script -- full Nepali is spoken aloud + printed above)
            last_status_line = ", ".join(
                f"{d['label']} ({_DIRECTION_EN.get(d['direction'], '?')}, "
                f"{_DISTANCE_EN.get(d['distance'], '?')})"
                for d in detections
            )
        elif not detections:
            last_status_line = "Nothing detected"

        # Draw boxes + labels for the live preview window (visual proof for judges)
        annotated = result.plot()
        cv2.putText(
            annotated, "Khoji Lite - Live Mode (press q to quit)",
            (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
        )
        # Status bar at the bottom showing the last announced scene in English
        # (audio + terminal carry the actual Nepali sentence)
        bar_y = frame_h - 20
        cv2.rectangle(annotated, (0, frame_h - 40), (frame_w, frame_h), (0, 0, 0), -1)
        cv2.putText(
            annotated, f"Last spoken: {last_status_line}",
            (10, bar_y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1
        )
        cv2.imshow("Khoji Lite - Live", annotated)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
