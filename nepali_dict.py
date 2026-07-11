# Nepali nouns for all 80 COCO classes detected by YOLOv8.
# Used to build spoken sentences like: "अगाडि एउटा कुर्सी छ"

NEPALI_OBJECTS = {
    "person": "मानिस",
    "bicycle": "साइकल",
    "car": "कार",
    "motorcycle": "मोटरसाइकल",
    "airplane": "हवाइजहाज",
    "bus": "बस",
    "train": "रेल",
    "truck": "ट्रक",
    "boat": "डुङ्गा",
    "traffic light": "ट्राफिक बत्ती",
    "fire hydrant": "फायर हाइड्रेन्ट",
    "stop sign": "रोक चिन्ह",
    "parking meter": "पार्किङ मिटर",
    "bench": "बेन्च",
    "bird": "चरा",
    "cat": "बिरालो",
    "dog": "कुकुर",
    "horse": "घोडा",
    "sheep": "भेडा",
    "cow": "गाई",
    "elephant": "हात्ती",
    "bear": "भालु",
    "zebra": "जेब्रा",
    "giraffe": "जिराफ",
    "backpack": "झोला",
    "umbrella": "छाता",
    "handbag": "ह्यान्डब्याग",
    "tie": "टाई",
    "suitcase": "सुटकेस",
    "frisbee": "फ्रिस्बी",
    "skis": "स्की",
    "snowboard": "स्नोबोर्ड",
    "sports ball": "बल",
    "kite": "चङ्गा",
    "baseball bat": "ब्याट",
    "baseball glove": "पन्जा",
    "skateboard": "स्केटबोर्ड",
    "surfboard": "सर्फबोर्ड",
    "tennis racket": "र्याकेट",
    "bottle": "बोतल",
    "wine glass": "गिलास",
    "cup": "कप",
    "fork": "काँटा",
    "knife": "चक्कु",
    "spoon": "चम्चा",
    "bowl": "बाटा",
    "banana": "केरा",
    "apple": "स्याउ",
    "sandwich": "स्यान्डविच",
    "orange": "सुन्तला",
    "broccoli": "ब्रोकाउली",
    "carrot": "गाजर",
    "hot dog": "हट डग",
    "pizza": "पिज्जा",
    "donut": "डोनट",
    "cake": "केक",
    "chair": "कुर्सी",
    "couch": "सोफा",
    "potted plant": "गमला",
    "bed": "ओछ्यान",
    "dining table": "टेबल",
    "toilet": "शौचालय",
    "tv": "टिभी",
    "laptop": "ल्यापटप",
    "mouse": "माउस",
    "remote": "रिमोट",
    "keyboard": "किबोर्ड",
    "cell phone": "मोबाइल फोन",
    "microwave": "माइक्रोवेभ",
    "oven": "ओभन",
    "toaster": "टोस्टर",
    "sink": "सिंक",
    "refrigerator": "फ्रिज",
    "book": "किताब",
    "clock": "घडी",
    "vase": "फूलदान",
    "scissors": "कैंची",
    "teddy bear": "टेडी बियर",
    "hair drier": "हेयर ड्रायर",
    "toothbrush": "टुथब्रस",
}

# Rough distance zone based on bounding-box height relative to frame height.
# Bigger box = closer object. Tune thresholds after testing with real photos.
def distance_zone(box_height_ratio: float) -> str:
    if box_height_ratio > 0.35:
        return "नजिकै"       # near
    else:
        return "टाढा"        # far


# Direction based on horizontal position of the box's center in the frame.
def direction_zone(box_center_x_ratio: float) -> str:
    """box_center_x_ratio: box center x / frame width, range 0.0-1.0"""
    if box_center_x_ratio < 0.35:
        return "बायाँतिर"    # to the left
    elif box_center_x_ratio > 0.65:
        return "दायाँतिर"    # to the right
    else:
        return "अगाडि"       # straight ahead / center


def build_sentence(object_counts: dict, zones: dict) -> str:
    """
    object_counts: {"chair": 2, "person": 1}
    zones: {"chair": "नजिकै", "person": "अगाडि"} (most common zone per object)
    Returns one combined Nepali sentence. (Used by the photo-upload app.py flow.)
    """
    if not object_counts:
        return "अगाडि केही देखिएन।"  # nothing detected ahead

    parts = []
    for obj, count in object_counts.items():
        name = NEPALI_OBJECTS.get(obj, obj)
        zone = zones.get(obj, "अगाडि")
        if count > 1:
            parts.append(f"{zone} {count} वटा {name}")
        else:
            parts.append(f"{zone} एउटा {name}")

    return " र ".join(parts) + " छ।"


def build_sentence_with_direction(detections: list) -> str:
    """
    detections: list of dicts like
        {"label": "chair", "direction": "बायाँतिर", "distance": "नजिकै"}
    Groups identical (label, direction, distance) combos and speaks counts.
    Used by the real-time live_mode.py loop.
    """
    if not detections:
        return "अगाडि केही देखिएन।"

    groups = {}
    for d in detections:
        key = (d["direction"], d["distance"], d["label"])
        groups[key] = groups.get(key, 0) + 1

    parts = []
    for (direction, distance, label), count in groups.items():
        name = NEPALI_OBJECTS.get(label, label)
        if count > 1:
            parts.append(f"{direction} {distance} {count} वटा {name}")
        else:
            parts.append(f"{direction} {distance} एउटा {name}")

    return " र ".join(parts) + " छ।"
