import cv2
import streamlit as st
import numpy as np
import pandas as pd
import mediapipe as mp
import datetime
import time
import pygame
import torch
import pickle

import sys

device = "cuda" if torch.cuda.is_available() else "cpu"
st.set_page_config(
    page_title="Real-time AI Posture Correction for Big 3 Exercises",
    layout="centered",
    initial_sidebar_state="auto",
    menu_items=None,
)

# Load YOLOv5 model
model_weights_path = "./models/best_big_bounding.pt"
model = torch.hub.load("./yolov5-master/yolov5-master", "custom", path=model_weights_path, source="local")
model.to(device)
model.eval()

# Record previous alert time
previous_alert_time = 0


def most_frequent(data):
    return max(data, key=data.count)


# Angle calculation function
def calculateAngle(a, b, c):
    a = np.array(a)  # First point
    b = np.array(b)  # Middle point
    c = np.array(c)  # End point

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(
        a[1] - b[1], a[0] - b[0]
    )
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle


# Object detection function using YOLOv5
def detect_objects(frame):
    results = model(frame)
    pred = results.pred[0]

    return pred


# Streamlit app initialization
st.title("Real-time AI Posture Correction for Big 3 Exercises")

pygame.mixer.init()

counter = 0
# Add menu to sidebar
menu_selection = st.selectbox("Select Exercise", ("Bench Press", "Squat", "Deadlift"))
counter_display = st.sidebar.empty()
counter_display.header(f"Current Counter: {counter} times")

# Load different models based on the selected exercise
current_stage = ""
posture_status = [None]

model_weights_path = "./models/benchpress/benchpress.pkl"
with open(model_weights_path, "rb") as f:
    model_e = pickle.load(f)

if menu_selection == "Bench Press":
    model_weights_path = "./models/benchpress/benchpress.pkl"
    with open(model_weights_path, "rb") as f:
        model_e = pickle.load(f)
elif menu_selection == "Squat":
    model_weights_path = "./models/squat/squat.pkl"
    with open(model_weights_path, "rb") as f:
        model_e = pickle.load(f)
elif menu_selection == "Deadlift":
    model_weights_path = "./models/deadlift/deadlift.pkl"
    with open(model_weights_path, "rb") as f:
        model_e = pickle.load(f)

FRAME_WINDOW = st.image([])
camera = cv2.VideoCapture(0)

# Initialize Mediapipe Pose model: min detection confidence=0.5, min tracking confidence=0.7, model complexity=2
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    min_detection_confidence=0.5, min_tracking_confidence=0.7, model_complexity=2
)

# Confidence threshold slider
confidence_threshold = st.sidebar.slider("Joint Tracking Confidence Threshold", 0.0, 1.0, 0.7)

# Initialize empty areas for angle display
neck_angle_display = st.sidebar.empty()
left_shoulder_angle_display = st.sidebar.empty()
right_shoulder_angle_display = st.sidebar.empty()
left_elbow_angle_display = st.sidebar.empty()
right_elbow_angle_display = st.sidebar.empty()
left_hip_angle_display = st.sidebar.empty()
right_hip_angle_display = st.sidebar.empty()
left_knee_angle_display = st.sidebar.empty()
right_knee_angle_display = st.sidebar.empty()
left_ankle_angle_display = st.sidebar.empty()
right_ankle_angle_display = st.sidebar.empty()

while True:
    ret, frame = camera.read()
    if not ret or frame is None:
        st.warning("Failed to grab frame from camera.")
        continue # Skip this iteration
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.flip(frame, 1)  # Flip frame horizontally

    # Object detection using YOLOv5
    results_yolo = detect_objects(frame)

    # Display YOLOv5 results on screen
    try:
        if results_yolo is not None:
            for det in results_yolo:
                c1, c2 = det[:2].int(), det[2:4].int()
                cls, conf, *_ = det
                label = f"person {conf:.2f}"

                if conf >= 0.7:  # Only display objects with confidence >= 0.7
                    # Convert c1 and c2 to tuples
                    c1 = (c1[0].item(), c1[1].item())
                    c2 = (c2[0].item(), c2[1].item())

                    # Extract object frame detected by YOLOv5
                    object_frame = frame[c1[1] : c2[1], c1[0] : c2[0]]

                    # Process object frame for pose estimation
                    object_frame_rgb = cv2.cvtColor(object_frame, cv2.COLOR_BGR2RGB)
                    results_pose = pose.process(object_frame_rgb)

                    if results_pose.pose_landmarks is not None:
                        landmarks = results_pose.pose_landmarks.landmark
                        nose = [
                            landmarks[mp_pose.PoseLandmark.NOSE].x,
                            landmarks[mp_pose.PoseLandmark.NOSE].y,
                        ]  # Nose
                        left_shoulder = [
                            landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x,
                            landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y,
                        ]  # Left shoulder
                        left_elbow = [
                            landmarks[mp_pose.PoseLandmark.LEFT_ELBOW].x,
                            landmarks[mp_pose.PoseLandmark.LEFT_ELBOW].y,
                        ]  # Left elbow
                        left_wrist = [
                            landmarks[mp_pose.PoseLandmark.LEFT_WRIST].x,
                            landmarks[mp_pose.PoseLandmark.LEFT_WRIST].y,
                        ]  # Left wrist
                        left_hip = [
                            landmarks[mp_pose.PoseLandmark.LEFT_HIP].x,
                            landmarks[mp_pose.PoseLandmark.LEFT_HIP].y,
                        ]  # Left hip
                        left_knee = [
                            landmarks[mp_pose.PoseLandmark.LEFT_KNEE].x,
                            landmarks[mp_pose.PoseLandmark.LEFT_KNEE].y,
                        ]  # Left knee
                        left_ankle = [
                            landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].x,
                            landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].y,
                        ]  # Left ankle
                        left_heel = [
                            landmarks[mp_pose.PoseLandmark.LEFT_HEEL].x,
                            landmarks[mp_pose.PoseLandmark.LEFT_HEEL].y,
                        ]  # Left heel
                        right_shoulder = [
                            landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].x,
                            landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y,
                        ]  # Right shoulder
                        right_elbow = [
                            landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW].x,
                            landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW].y,
                        ]  # Right elbow
                        right_wrist = [
                            landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].x,
                            landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].y,
                        ]  # Right wrist
                        right_hip = [
                            landmarks[mp_pose.PoseLandmark.RIGHT_HIP].x,
                            landmarks[mp_pose.PoseLandmark.RIGHT_HIP].y,
                        ]  # Right hip
                        right_knee = [
                            landmarks[mp_pose.PoseLandmark.RIGHT_KNEE].x,
                            landmarks[mp_pose.PoseLandmark.RIGHT_KNEE].y,
                        ]  # Right knee
                        right_ankle = [
                            landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE].x,
                            landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE].y,
                        ]  # Right ankle
                        right_heel = [
                            landmarks[mp_pose.PoseLandmark.RIGHT_HEEL].x,
                            landmarks[mp_pose.PoseLandmark.RIGHT_HEEL].y,
                        ]  # Right heel

                        # Calculate angles
                        neck_angle = (
                            calculateAngle(left_shoulder, nose, left_hip)
                            + calculateAngle(right_shoulder, nose, right_hip) / 2
                        )
                        left_elbow_angle = calculateAngle(
                            left_shoulder, left_elbow, left_wrist
                        )
                        right_elbow_angle = calculateAngle(
                            right_shoulder, right_elbow, right_wrist
                        )
                        left_shoulder_angle = calculateAngle(
                            left_elbow, left_shoulder, left_hip
                        )
                        right_shoulder_angle = calculateAngle(
                            right_elbow, right_shoulder, right_hip
                        )
                        left_hip_angle = calculateAngle(
                            left_shoulder, left_hip, left_knee
                        )
                        right_hip_angle = calculateAngle(
                            right_shoulder, right_hip, right_knee
                        )
                        left_knee_angle = calculateAngle(
                            left_hip, left_knee, left_ankle
                        )
                        right_knee_angle = calculateAngle(
                            right_hip, right_knee, right_ankle
                        )
                        left_ankle_angle = calculateAngle(
                            left_knee, left_ankle, left_heel
                        )
                        right_ankle_angle = calculateAngle(
                            right_knee, right_ankle, right_heel
                        )

                        # Update angle displays
                        neck_angle_display.text(f"Neck Angle: {neck_angle:.2f}Â°")
                        left_shoulder_angle_display.text(
                            f"Left Shoulder Angle: {left_shoulder_angle:.2f}Â°"
                        )
                        right_shoulder_angle_display.text(
                            f"Right Shoulder Angle: {right_shoulder_angle:.2f}Â°"
                        )
                        left_elbow_angle_display.text(
                            f"Left Elbow Angle: {left_elbow_angle:.2f}Â°"
                        )
                        right_elbow_angle_display.text(
                            f"Right Elbow Angle: {right_elbow_angle:.2f}Â°"
                        )
                        left_hip_angle_display.text(
                            f"Left Hip Angle: {left_hip_angle:.2f}Â°"
                        )
                        right_hip_angle_display.text(
                            f"Right Hip Angle: {right_hip_angle:.2f}Â°"
                        )
                        left_knee_angle_display.text(
                            f"Left Knee Angle: {left_knee_angle:.2f}Â°"
                        )
                        right_knee_angle_display.text(
                            f"Right Knee Angle: {right_knee_angle:.2f}Â°"
                        )
                        left_ankle_angle_display.text(
                            f"Left Ankle Angle: {left_ankle_angle:.2f}Â°"
                        )
                        right_ankle_angle_display.text(
                            f"Right Ankle Angle: {right_ankle_angle:.2f}Â°"
                        )

                # Repetition counting algorithm
                try:
                    row = [
                        coord
                        for res in results_pose.pose_landmarks.landmark
                        for coord in [res.x, res.y, res.z, res.visibility]
                    ]
                    X = pd.DataFrame([row])
                    exercise_class = model_e.predict(X)[0]
                    exercise_class_prob = model_e.predict_proba(X)[0]
                    print(exercise_class, exercise_class_prob)
                    if "down" in exercise_class:
                        current_stage = "down"
                        posture_status.append(exercise_class)
                        print(f"posture of exercise performer: {posture_status}")
                    elif current_stage == "down" and "up" in exercise_class:
                        # and exercise_class_prob[exercise_class_prob.argmax()] >= 0.3
                        current_stage = "up"
                        counter += 1
                        posture_status.append(exercise_class)
                        print(f"posture of exercise performer: {posture_status}")
                        counter_display.header(f"Current count: {counter} times")
                        if "correct" not in most_frequent(posture_status):
                            current_time = time.time()
                            if current_time - previous_alert_time >= 3:
                                now = datetime.datetime.now()
                                if "excessive_arch" in most_frequent(posture_status):
                                    options = [
                                        (
                                            "Avoid arching your lower back too much; try to keep it natural.",
                                            "./resources/sounds/excessive_arch_1.mp3",
                                        ),
                                        (
                                            "Lift your pelvis a bit and tighten your abs to keep your back flat.",
                                            "./resources/sounds/excessive_arch_2.mp3",
                                        ),
                                    ]
                                    selected_option = random.choice(options)
                                    selected_message = selected_option[0]
                                    selected_music = selected_option[1]
                                    st.error(selected_message)
                                    pygame.mixer.music.load(selected_music)
                                    pygame.mixer.music.play()
                                    posture_status = []
                                    previous_alert_time = current_time
                                elif "arms_spread" in most_frequent(posture_status):
                                    options = [
                                        (
                                            "Your grip is too wide. Hold the bar a bit narrower.",
                                            "./resources/sounds/arms_spread_1.mp3",
                                        ),
                                        (
                                            "When gripping the bar, hold it slightly wider than shoulder width.",
                                            "./resources/sounds/arms_spread_2.mp3",
                                        ),
                                    ]
                                    selected_option = random.choice(options)
                                    selected_message = selected_option[0]
                                    selected_music = selected_option[1]
                                    st.error(selected_message)
                                    pygame.mixer.music.load(selected_music)
                                    pygame.mixer.music.play()
                                    posture_status = []
                                    previous_alert_time = current_time
                                elif "spine_neutral" in most_frequent(posture_status):
                                    options = [
                                        (
                                            "Avoid excessive curvature of the spine.",
                                            "./resources/sounds/spine_neutral_feedback_1.mp3",
                                        ),
                                        (
                                            "Lift your chest and push your shoulders back.",
                                            "./resources/sounds/spine_neutral_feedback_2.mp3",
                                        ),
                                    ]
                                    selected_option = random.choice(options)
                                    selected_message = selected_option[0]
                                    selected_music = selected_option[1]
                                    st.error(selected_message)
                                    pygame.mixer.music.load(selected_music)
                                    pygame.mixer.music.play()
                                    posture_status = []
                                    previous_alert_time = current_time
                                elif "caved_in_knees" in most_frequent(posture_status):
                                    options = [
                                        (
                                            "Be cautious not to let your knees cave in during the squat.",
                                            "./resources/sounds/caved_in_knees_feedback_1.mp3",
                                        ),
                                        (
                                            "Push your hips back to keep your knees and toes in a straight line.",
                                            "./resources/sounds/caved_in_knees_feedback_2.mp3",
                                        ),
                                    ]
                                    selected_option = random.choice(options)
                                    selected_message = selected_option[0]
                                    selected_music = selected_option[1]
                                    st.error(selected_message)
                                    pygame.mixer.music.load(selected_music)
                                    pygame.mixer.music.play()
                                    posture_status = []
                                    previous_alert_time = current_time
                                elif "feet_spread" in most_frequent(posture_status):
                                    st.error(
                                        "Narrow your stance to about shoulder width."
                                    )
                                    pygame.mixer.music.load(
                                        "./resources/sounds/feet_spread.mp3"
                                    )
                                    pygame.mixer.music.play()
                                    posture_status = []
                                    previous_alert_time = current_time
                                elif "arms_narrow" in most_frequent(posture_status):
                                    st.error(
                                        "Your grip is too wide. Hold the bar a bit narrower."
                                    )
                                    pygame.mixer.music.load(
                                        "./resources/sounds/arms_narrow.mp3"
                                    )
                                    pygame.mixer.music.play()
                                    posture_status = []
                                    previous_alert_time = current_time
                        elif "correct" in most_frequent(posture_status):
                            pygame.mixer.music.load("./resources/sounds/correct.mp3")
                            pygame.mixer.music.play()
                            st.info(
                                "You are performing the exercise with the correct posture."
                            )
                            posture_status = []
                except Exception as e:
                    pass

                # Draw landmarks
                for landmark in mp_pose.PoseLandmark:
                    if landmarks[landmark.value].visibility >= confidence_threshold:
                        mp.solutions.drawing_utils.draw_landmarks(
                            object_frame,
                            results_pose.pose_landmarks,
                            mp_pose.POSE_CONNECTIONS,
                            mp.solutions.drawing_styles.get_default_pose_landmarks_style(),
                        )

            # Draw object frame back to original frame
            frame = object_frame

        # Output original frame
        FRAME_WINDOW.image(frame)
    except Exception as e:
        pass