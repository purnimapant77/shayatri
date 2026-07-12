"""
Sarathi — Nepali voice assistant for visually impaired users.
Take/upload a photo -> YOLO detects objects -> Nepali sentence is built ->
sentence is spoken aloud + shown on screen.

Also includes a Live Mode tab that launches continuous real-time detection
in its own window (a separate OpenCV window, not embedded in the browser --
browsers can't run continuous Python-side video processing without a heavy
extra dependency, so this keeps things simple and reliable for a live demo).

Run with: streamlit run app.py
"""
import subprocess
import sys

import cv2
import streamlit as st
from PIL import Image
import numpy as np

from detector import process_image

st.set_page_config(page_title="Sarathi", page_icon="🎙️", layout="centered")

st.title("🎙️ Sarathi — दृष्टिविहीनका लागि आवाज सहायक")
st.caption("फोटो लिनुहोस् वा अपलोड गर्नुहोस् — अगाडि के छ भनेर नेपालीमा सुन्नुहोस्।")

st.divider()

tab1, tab2, tab3 = st.tabs([
    "📷 क्यामेराबाट फोटो लिनुहोस्",
    "🖼️ फोटो अपलोड गर्नुहोस्",
    "🔴 लाइभ मोड (Real-time)",
])

image_input = None

with tab1:
    camera_photo = st.camera_input("क्यामेरा खोल्नुहोस् र फोटो खिच्नुहोस्")
    if camera_photo is not None:
        image_input = Image.open(camera_photo)

with tab2:
    uploaded_file = st.file_uploader("फोटो छान्नुहोस्", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image_input = Image.open(uploaded_file)

with tab3:
    st.markdown("""
    **लाइभ मोडले** तपाईंको क्यामेरा लगातार खुला राखेर वस्तुहरू पहिचान गर्छ
    र दिशा (बायाँ/अगाडि/दायाँ) र दूरी (नजिकै/टाढा) सहित नेपालीमा लगातार बोल्छ।

    यो एउटा छुट्टै विन्डोमा खुल्छ (ब्राउजरभित्र होइन), किनभने ब्राउजरमा
    लगातार भिडियो प्रोसेस गर्न धेरै भारी अतिरिक्त लाइब्रेरी चाहिन्छ जुन
    डेमोको बेला जोखिमपूर्ण हुन सक्छ। तर तपाईंले यहीं बटन थिचेर सुरु गर्न सक्नुहुन्छ।
    """)

    st.info(
        "⚠️ सुरु गर्नु अघि: माथिको क्यामेरा ट्याब बन्द गर्नुहोस् (एउटै समयमा "
        "दुई ठाउँबाट क्यामेरा प्रयोग गर्न मिल्दैन)।"
    )

    if st.button("🔴 लाइभ मोड सुरु गर्नुहोस् (Start Live Mode)", type="primary"):
        st.warning(
            "एउटा नयाँ विन्डो खुल्दैछ — यो Streamlit ब्राउजर ट्याबमा देखिँदैन। "
            "बन्द गर्न त्यो विन्डोमा 'q' थिच्नुहोस्।\n\n"
            "(A new window is opening — it won't appear in this browser tab. "
            "Press 'q' inside that window to close it.)"
        )
        try:
            # Launch live_mode.py as its own process using the SAME python
            # interpreter (and therefore same venv) that's running Streamlit.
            subprocess.Popen([sys.executable, "live_mode.py"])
            st.success("लाइभ मोड सुरु भयो। नयाँ विन्डो हेर्नुहोस्।")
        except Exception as e:
            st.error(f"लाइभ मोड सुरु गर्न सकिएन: {e}")

if image_input is not None:
    with st.spinner("वस्तुहरू पहिचान गर्दै... (detecting objects...)"):
        img_array = np.array(image_input.convert("RGB"))
        # ultralytics expects BGR-ish handling internally; RGB numpy works fine too
        try:
            sentence, audio_bytes, annotated_bgr = process_image(img_array)
            annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)

            st.image(annotated_rgb, caption="पहिचान गरिएका वस्तुहरू", use_container_width=True)

            st.subheader("नतिजा:")
            st.success(sentence)

            st.audio(audio_bytes, format="audio/mp3", autoplay=True)

        except Exception as e:
            st.error(f"समस्या आयो: {e}")
            st.info(
                "यदि यो इन्टरनेट/gTTS समस्या हो भने, इन्टरनेट जडान जाँच्नुहोस्। "
                "(If this is a gTTS/network error, check your internet connection.)"
            )

st.divider()
with st.expander("ℹ️ यो कसरी काम गर्छ (How it works)"):
    st.markdown("""
    1. **YOLOv8** (pretrained on COCO) detects 80 common object classes in the photo.
    2. Each detected object is mapped to a **Nepali noun** and a rough **distance zone**
       (नजिकै / टाढा) plus **direction** (बायाँतिर / अगाडि / दायाँतिर) based on the
       object's position and size in the frame.
    3. A simple template builds a natural Nepali sentence describing the scene.
    4. Both modes use **gTTS** to convert the sentence to spoken Nepali audio.
       Live Mode caches each spoken sentence to disk so repeated phrases
       (very common when scanning a room) play back instantly.

    """)

