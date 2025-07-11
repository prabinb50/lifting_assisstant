import cv2
import streamlit as st
import numpy as np
import pandas as pd
import mediapipe as mp
import torch

st.set_page_config(
    page_title="Image Joint Points and YOLOv5 Object Detection",
    layout="centered",
    initial_sidebar_state="auto",
    menu_items=None,
)

# Load YOLOv5 model
model_weights_path = "./models/best_big_bounding.pt"
model = torch.hub.load("ultralytics/yolov5", "custom", path=model_weights_path)
model.to("mps")
model.eval()

# Load image
image_path = "bench2.jpg"
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


# Function to detect objects using YOLOv5
def detect_objects(frame):
    results = model(frame)
    pred = results.pred[0]
    return pred


# Initialize Mediapipe Pose model
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    min_detection_confidence=0.3, min_tracking_confidence=0.7, model_complexity=2
)

# Detect objects in the image
results_yolo = detect_objects(image)

# Display YOLOv5 results on the screen
if results_yolo is not None:
    for det in results_yolo:
        c1, c2 = det[:2].int(), det[2:4].int()
        cls, conf, *_ = det
        label = f"person {conf:.2f}"

        if conf >= 0.7:  # Display objects only if confidence is 0.7 or higher
            # Convert c1 and c2 to tuples
            c1 = (c1[0].item(), c1[1].item())
            c2 = (c2[0].item(), c2[1].item())

            # Extract the frame of the object detected by YOLOv5
            object_frame = image[c1[1] : c2[1], c1[0] : c2[0]]

            # Process the object frame for pose estimation
            object_frame_rgb = cv2.cvtColor(object_frame, cv2.COLOR_BGR2RGB)
            results_pose = pose.process(object_frame_rgb)

            if results_pose.pose_landmarks is not None:
                landmarks = results_pose.pose_landmarks.landmark

                # Display landmarks
                for landmark in mp_pose.PoseLandmark:
                    if landmarks[landmark.value].visibility >= 0.3:
                        mp.solutions.drawing_utils.draw_landmarks(
                            object_frame,
                            results_pose.pose_landmarks,
                            mp_pose.POSE_CONNECTIONS,
                            mp.solutions.drawing_styles.get_default_pose_landmarks_style(),
                        )

            # Draw the object frame back onto the original frame
            image[c1[1] : c2[1], c1[0] : c2[0]] = object_frame

            image = cv2.rectangle(image, c1, c2, (0, 255, 0), 2)
            image = cv2.putText(
                image,
                label,
                (c1[0], c1[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
            )

# Display the image
st.image(image, caption="YOLOv5 Object Detection and Joint Points", use_column_width=True)
