"""
Core logic: image -> YOLO detections -> Nepali sentence -> speech audio file.
Kept separate from app.py so it can be tested standalone with:
    python3 detector.py test.jpg
"""
import io
from collections import defaultdict, Counter

from ultralytics import YOLO
from gtts import gTTS

from nepali_dict import distance_zone, build_sentence

_model = None


def get_model():
    """Load YOLOv8n once and reuse (loading is slow, ~1-2s)."""
    global _model
    if _model is None:
        _model = YOLO("yolov8n.pt")
    return _model


def detect_objects(image_path_or_array, conf_threshold: float = 0.35):
    """
    Runs YOLO on an image and returns:
      object_counts: {"chair": 2, "person": 1}
      zones: {"chair": "नजिकै", "person": "अगाडि"}
      annotated_image: numpy array (BGR) with boxes drawn, for display
    """
    model = get_model()
    results = model(image_path_or_array, conf=conf_threshold, verbose=False)
    result = results[0]

    frame_h = result.orig_shape[0]

    counts = Counter()
    # track the largest box-height-ratio seen per object, to pick nearest zone
    best_ratio = defaultdict(float)

    for box in result.boxes:
        cls_id = int(box.cls[0])
        label = model.names[cls_id]
        counts[label] += 1

        y1, y2 = box.xyxy[0][1].item(), box.xyxy[0][3].item()
        box_height = y2 - y1
        ratio = box_height / frame_h
        if ratio > best_ratio[label]:
            best_ratio[label] = ratio

    zones = {label: distance_zone(r) for label, r in best_ratio.items()}
    annotated = result.plot()  # numpy BGR image with boxes + labels drawn

    return dict(counts), zones, annotated


def text_to_speech_bytes(nepali_text: str) -> bytes:
    """Convert Nepali text to MP3 audio bytes using gTTS (needs internet)."""
    tts = gTTS(text=nepali_text, lang="ne")
    buf = io.BytesIO()
    tts.write_to_fp(buf)
    buf.seek(0)
    return buf.read()


def process_image(image_path_or_array, conf_threshold: float = 0.35):
    """
    Full pipeline: detect -> build sentence -> synthesize speech.
    Returns (nepali_sentence, audio_bytes, annotated_image_bgr)
    """
    counts, zones, annotated = detect_objects(image_path_or_array, conf_threshold)
    sentence = build_sentence(counts, zones)
    audio_bytes = text_to_speech_bytes(sentence)
    return sentence, audio_bytes, annotated


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python3 detector.py <image_path>")
        sys.exit(1)

    sentence, audio_bytes, annotated = process_image(sys.argv[1])
    print("Nepali sentence:", sentence)
    with open("test_output.mp3", "wb") as f:
        f.write(audio_bytes)
    print("Saved audio to test_output.mp3")

    import cv2
    cv2.imwrite("test_annotated.jpg", annotated)
    print("Saved annotated image to test_annotated.jpg")
