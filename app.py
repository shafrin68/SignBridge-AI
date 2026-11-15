
import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import urllib.request
import os

st.set_page_config(
    page_title="SignBridge AI",
    page_icon="🤟"
)

st.title("🤟 SignBridge AI")
st.subheader("AI-Based Sign Language Hand Detection")

st.write(
    "Upload a hand gesture image "
    "to detect hand landmarks using AI."
)

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/"
    "hand_landmarker.task"
)

MODEL_PATH = "hand_landmarker.task"

@st.cache_resource
def load_detector():
    if not os.path.exists(MODEL_PATH):
        urllib.request.urlretrieve(
            MODEL_URL, MODEL_PATH
        )

    options = mp.tasks.vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(
            model_asset_path=MODEL_PATH
        ),
        running_mode=mp.tasks.vision.RunningMode.IMAGE,
        num_hands=2
    )

    return mp.tasks.vision.HandLandmarker.create_from_options(
        options
    )

uploaded_file = st.file_uploader(
    "Choose Hand Gesture Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    data = np.frombuffer(
        uploaded_file.getvalue(),
        np.uint8
    )

    image = cv2.imdecode(data, cv2.IMREAD_COLOR)

    if image is not None:
        rgb = cv2.cvtColor(
            image, cv2.COLOR_BGR2RGB
        )

        try:
            detector = load_detector()

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb
            )

            result = detector.detect(mp_image)

            if result.hand_landmarks:
                for hand in result.hand_landmarks:
                    for point in hand:
                        x = int(point.x * image.shape[1])
                        y = int(point.y * image.shape[0])

                        cv2.circle(
                            image, (x, y), 5,
                            (0, 255, 0), -1
                        )

                st.success("Hand Detected Successfully!")

                st.write(
                    "Number of hands:",
                    len(result.hand_landmarks)
                )
            else:
                st.warning("No hand detected!")

            st.image(
                cv2.cvtColor(
                    image, cv2.COLOR_BGR2RGB
                ),
                caption="SignBridge AI Result"
            )

        except Exception as e:
            st.error(f"Detection Error: {e}")
          
