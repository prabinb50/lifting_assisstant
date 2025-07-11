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
import random
import tempfile
from PIL import Image

# Page configuration with custom theme
st.set_page_config(
    page_title="AI Posture Coach for Powerlifting",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": "https://github.com/yourusername/powerlifter",
        "Report a bug": "https://github.com/yourusername/powerlifter/issues",
        "About": "# AI Posture Coach\nReal-time posture analysis for powerlifting exercises."
    }
)

# Custom CSS for better UI
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #FF4B4B;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        margin-top: 1rem;
        margin-bottom: 1rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
        margin-bottom: 2rem;
    }
    .stTabs [data-baseweb="tab"] {
        height: 4rem;
        white-space: pre-wrap;
        border-radius: 4px;
        padding: 0.5rem 1rem;
        background-color: #f0f0f0;
    }
    .stTabs [aria-selected="true"] {
        background-color: #FF4B4B !important;
        color: white !important;
    }
    .exercise-selector {
        background-color: #f9f9f9;
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1.5rem;
    }
    .stButton button {
        width: 100%;
        border-radius: 5px;
        background-color: #FF4B4B;
        color: white;
    }
    .feedback-box {
        padding: 1rem;
        border-radius: 10px;
        margin-top: 1rem;
    }
    .positive {
        background-color: #d1f0d1;
        border-left: 5px solid #4CAF50;
    }
    .negative {
        background-color: #f8d7da;
        border-left: 5px solid #dc3545;
    }
    .info-box {
        background-color: #e2f0fb;
        padding: 1rem;
        border-radius: 10px;
        margin-top: 1rem;
        border-left: 5px solid #0dcaf0;
    }
    .upload-box {
        border: 2px dashed #aaa;
        border-radius: 10px;
        padding: 2rem;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 1rem;
        margin-bottom: 1rem;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# App header
st.markdown('<h1 class="main-header">AI Posture Coach for Powerlifting</h1>', unsafe_allow_html=True)

# Initialize session state variables if they don't exist
if 'counter' not in st.session_state:
    st.session_state.counter = 0
if 'current_stage' not in st.session_state:
    st.session_state.current_stage = ""
if 'posture_status' not in st.session_state:
    st.session_state.posture_status = [None]
if 'previous_alert_time' not in st.session_state:
    st.session_state.previous_alert_time = 0
if 'processing_video' not in st.session_state:
    st.session_state.processing_video = False
if 'video_processed' not in st.session_state:
    st.session_state.video_processed = False
if 'video_file_buffer' not in st.session_state:
    st.session_state.video_file_buffer = None
if 'video_results' not in st.session_state:
    st.session_state.video_results = []

# Initialize Pygame mixer for audio feedback
pygame.mixer.init()


def most_frequent(data):
    return max(data, key=data.count) if data else None


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


# Function to load exercise model
def load_exercise_model(exercise_type):
    if exercise_type == "Bench Press":
        model_weights_path = "./models/benchpress/benchpress.pkl"
    elif exercise_type == "Squat":
        model_weights_path = "./models/squat/squat.pkl"
    elif exercise_type == "Deadlift":
        model_weights_path = "./models/deadlift/deadlift.pkl"
    else:
        return None
    
    with open(model_weights_path, "rb") as f:
        return pickle.load(f)


# Function to provide feedback
def provide_feedback(posture_issue, exercise_class_prob=None):
    current_time = time.time()
    
    if current_time - st.session_state.previous_alert_time < 3:
        return
    
    if posture_issue == "excessive_arch":
        options = [
            ("Don't arch your back too much. Try to focus on expanding your chest.",
             "./resources/sounds/excessive_arch_1.mp3"),
            ("Lift your pelvis slightly more and tighten your abs to keep your back flat.",
             "./resources/sounds/excessive_arch_2.mp3"),
        ]
    elif posture_issue == "arms_spread":
        options = [
            ("You're gripping the bar too wide. Narrow your grip a bit.",
             "./resources/sounds/arms_spread_1.mp3"),
            ("When gripping the bar, it's better to hold it just slightly wider than shoulder width.",
             "./resources/sounds/arms_spread_2.mp3"),
        ]
    elif posture_issue == "spine_neutral":
        options = [
            ("Try not to excessively bend your spine.",
             "./resources/sounds/spine_neutral_feedback_1.mp3"),
            ("Lift your chest and pull your shoulders back.",
             "./resources/sounds/spine_neutral_feedback_2.mp3"),
        ]
    elif posture_issue == "caved_in_knees":
        options = [
            ("Be careful not to let your knees cave inward.",
             "./resources/sounds/caved_in_knees_feedback_1.mp3"),
            ("Push your hips back to maintain alignment between your knees and toes.",
             "./resources/sounds/caved_in_knees_feedback_2.mp3"),
        ]
    elif posture_issue == "feet_spread":
        return ("Narrow your stance to keep your feet about shoulder-width apart.",
                "./resources/sounds/feet_spread.mp3")
    elif posture_issue == "arms_narrow":
        return ("It's better to grip the bar slightly wider than shoulder width.",
                "./resources/sounds/arms_narrow.mp3")
    elif posture_issue == "correct":
        return ("You are performing the exercise with correct posture.",
                "./resources/sounds/correct.mp3")
    else:
        return None
    
    selected_option = random.choice(options)
    return selected_option


# Function to process frames (used for both webcam and video upload)
def process_frame(frame, model_e, pose, confidence_threshold):
    results = {
        'frame': frame,
        'landmarks_detected': False,
        'angles': {},
        'posture_issue': None,
        'rep_counted': False
    }
    
    # Convert frame to RGB for processing
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process frame with MediaPipe Pose
    results_pose = pose.process(frame_rgb)
    
    # If pose landmarks detected
    if results_pose.pose_landmarks:
        results['landmarks_detected'] = True
        landmarks = results_pose.pose_landmarks.landmark
        
        # Extract landmark coordinates
        landmark_positions = {
            'nose': [landmarks[mp_pose.PoseLandmark.NOSE].x, landmarks[mp_pose.PoseLandmark.NOSE].y],
            'left_shoulder': [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y],
            'left_elbow': [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW].y],
            'left_wrist': [landmarks[mp_pose.PoseLandmark.LEFT_WRIST].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST].y],
            'left_hip': [landmarks[mp_pose.PoseLandmark.LEFT_HIP].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP].y],
            'left_knee': [landmarks[mp_pose.PoseLandmark.LEFT_KNEE].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE].y],
            'left_ankle': [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].y],
            'left_heel': [landmarks[mp_pose.PoseLandmark.LEFT_HEEL].x, landmarks[mp_pose.PoseLandmark.LEFT_HEEL].y],
            'right_shoulder': [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y],
            'right_elbow': [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW].x, landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW].y],
            'right_wrist': [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].x, landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].y],
            'right_hip': [landmarks[mp_pose.PoseLandmark.RIGHT_HIP].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP].y],
            'right_knee': [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE].x, landmarks[mp_pose.PoseLandmark.RIGHT_KNEE].y],
            'right_ankle': [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE].x, landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE].y],
            'right_heel': [landmarks[mp_pose.PoseLandmark.RIGHT_HEEL].x, landmarks[mp_pose.PoseLandmark.RIGHT_HEEL].y],
        }
        
        # Calculate angles
        angles = {
            'neck': (calculateAngle(landmark_positions['left_shoulder'], landmark_positions['nose'], 
                                    landmark_positions['left_hip']) + 
                     calculateAngle(landmark_positions['right_shoulder'], landmark_positions['nose'], 
                                    landmark_positions['right_hip'])) / 2,
            'left_elbow': calculateAngle(landmark_positions['left_shoulder'], landmark_positions['left_elbow'], 
                                        landmark_positions['left_wrist']),
            'right_elbow': calculateAngle(landmark_positions['right_shoulder'], landmark_positions['right_elbow'], 
                                         landmark_positions['right_wrist']),
            'left_shoulder': calculateAngle(landmark_positions['left_elbow'], landmark_positions['left_shoulder'], 
                                           landmark_positions['left_hip']),
            'right_shoulder': calculateAngle(landmark_positions['right_elbow'], landmark_positions['right_shoulder'], 
                                            landmark_positions['right_hip']),
            'left_hip': calculateAngle(landmark_positions['left_shoulder'], landmark_positions['left_hip'], 
                                      landmark_positions['left_knee']),
            'right_hip': calculateAngle(landmark_positions['right_shoulder'], landmark_positions['right_hip'], 
                                       landmark_positions['right_knee']),
            'left_knee': calculateAngle(landmark_positions['left_hip'], landmark_positions['left_knee'], 
                                       landmark_positions['left_ankle']),
            'right_knee': calculateAngle(landmark_positions['right_hip'], landmark_positions['right_knee'], 
                                        landmark_positions['right_ankle']),
            'left_ankle': calculateAngle(landmark_positions['left_knee'], landmark_positions['left_ankle'], 
                                        landmark_positions['left_heel']),
            'right_ankle': calculateAngle(landmark_positions['right_knee'], landmark_positions['right_ankle'], 
                                         landmark_positions['right_heel']),
        }
        
        results['angles'] = angles
        
        # Exercise classification
        try:
            row = [coord for res in results_pose.pose_landmarks.landmark 
                  for coord in [res.x, res.y, res.z, res.visibility]]
            X = pd.DataFrame([row])
            exercise_class = model_e.predict(X)[0]
            exercise_class_prob = model_e.predict_proba(X)[0]
            
            # Update posture status and rep count
            if "down" in exercise_class:
                st.session_state.current_stage = "down"
                st.session_state.posture_status.append(exercise_class)
            elif st.session_state.current_stage == "down" and "up" in exercise_class:
                st.session_state.current_stage = "up"
                st.session_state.counter += 1
                st.session_state.posture_status.append(exercise_class)
                results['rep_counted'] = True
                
                # Determine posture issue
                if st.session_state.posture_status:
                    most_frequent_status = most_frequent(st.session_state.posture_status)
                    if most_frequent_status and "correct" not in most_frequent_status:
                        for issue in ["excessive_arch", "arms_spread", "spine_neutral", 
                                    "caved_in_knees", "feet_spread", "arms_narrow"]:
                            if issue in most_frequent_status:
                                results['posture_issue'] = issue
                                break
                    elif most_frequent_status and "correct" in most_frequent_status:
                        results['posture_issue'] = "correct"
                        
        except Exception as e:
            pass
        
        # Draw landmarks on frame
        mp_drawing = mp.solutions.drawing_utils
        mp_drawing_styles = mp.solutions.drawing_styles
        
        for landmark in mp_pose.PoseLandmark:
            if landmarks[landmark.value].visibility >= confidence_threshold:
                mp_drawing.draw_landmarks(
                    frame,
                    results_pose.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    mp_drawing_styles.get_default_pose_landmarks_style()
                )
    
    results['frame'] = frame
    return results


# Process video file function
def process_video_file(video_file, model_e, pose, confidence_threshold, progress_bar):
    # Create a temporary file
    tfile = tempfile.NamedTemporaryFile(delete=False) 
    tfile.write(video_file.read())
    
    # Open the video file
    video = cv2.VideoCapture(tfile.name)
    
    # Get video properties
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = video.get(cv2.CAP_PROP_FPS)
    
    # Results storage
    results = []
    frame_count = 0
    
    # Reset counters
    st.session_state.counter = 0
    st.session_state.current_stage = ""
    st.session_state.posture_status = [None]
    
    # Process each frame
    while video.isOpened():
        ret, frame = video.read()
        if not ret:
            break
        
        # Flip the frame horizontally
        frame = cv2.flip(frame, 1)
        
        # Process the frame
        frame_result = process_frame(frame, model_e, pose, confidence_threshold)
        results.append(frame_result)
        
        # Update progress bar
        frame_count += 1
        progress_bar.progress(min(frame_count / total_frames, 1.0))
    
    # Clean up
    video.release()
    tfile.close()
    
    return results, fps


# Main content with tabs for webcam and video upload
tab1, tab2 = st.tabs(["🎥 Live Webcam Analysis", "📁 Video Analysis"])

# Initialize MediaPipe Pose model
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    min_detection_confidence=0.5, 
    min_tracking_confidence=0.7, 
    model_complexity=2
)

# Common sidebar elements
st.sidebar.markdown('<h2 class="sub-header">Exercise Configuration</h2>', unsafe_allow_html=True)

# Add class to the exercise selection box
st.sidebar.markdown('<div class="exercise-selector">', unsafe_allow_html=True)
menu_selection = st.sidebar.selectbox("Select Exercise", ("Bench Press", "Squat", "Deadlift"))
st.sidebar.markdown('</div>', unsafe_allow_html=True)

# Load model for selected exercise
model_e = load_exercise_model(menu_selection)

# Joint tracking confidence
confidence_threshold = st.sidebar.slider(
    "Joint Tracking Confidence", 
    0.0, 1.0, 0.7,
    help="Adjust this threshold to filter out low-confidence pose estimations"
)

# Add info about the exercise
st.sidebar.markdown('<div class="info-box">', unsafe_allow_html=True)
if menu_selection == "Bench Press":
    st.sidebar.markdown("""
    ### Bench Press Form Tips:
    - Keep a slight arch in your back
    - Grip the bar slightly wider than shoulder width
    - Keep your feet flat on the ground
    """)
elif menu_selection == "Squat":
    st.sidebar.markdown("""
    ### Squat Form Tips:
    - Keep your knees tracking over your toes
    - Maintain a neutral spine
    - Feet should be shoulder-width apart
    """)
else:  # Deadlift
    st.sidebar.markdown("""
    ### Deadlift Form Tips:
    - Keep your back straight
    - Bar should be over mid-foot
    - Grip slightly wider than shoulder width
    """)
st.sidebar.markdown('</div>', unsafe_allow_html=True)

# Tracking statistics in sidebar
st.sidebar.markdown('<h2 class="sub-header">Exercise Stats</h2>', unsafe_allow_html=True)
rep_counter = st.sidebar.empty()
rep_counter.markdown(f'<div class="metric-card"><h3>Rep Count: {st.session_state.counter}</h3></div>', unsafe_allow_html=True)

# Initialize angle displays
angle_displays = {
    'neck': st.sidebar.empty(),
    'left_shoulder': st.sidebar.empty(),
    'right_shoulder': st.sidebar.empty(),
    'left_elbow': st.sidebar.empty(),
    'right_elbow': st.sidebar.empty(),
    'left_hip': st.sidebar.empty(),
    'right_hip': st.sidebar.empty(),
    'left_knee': st.sidebar.empty(),
    'right_knee': st.sidebar.empty(),
    'left_ankle': st.sidebar.empty(),
    'right_ankle': st.sidebar.empty()
}

# TAB 1: WEBCAM MODE
with tab1:
    st.markdown('<h2 class="sub-header">Live Posture Analysis</h2>', unsafe_allow_html=True)
    
    # Instructions
    with st.expander("How to use webcam analysis"):
        st.write("""
        1. Make sure your webcam is connected and working
        2. Position yourself so your full body is visible
        3. Select your exercise type from the sidebar
        4. Begin performing the exercise
        5. You'll receive real-time feedback on your form
        """)
    
    # Start button for webcam
    start_webcam = st.button("Start Webcam Analysis")
    
    if start_webcam:
        # Video capture
        cap = cv2.VideoCapture(0)
        
        # Create placeholder for webcam feed
        frame_window = st.empty()
        feedback_placeholder = st.empty()
        
        try:
            while True:
                # Capture frame from webcam
                ret, frame = cap.read()
                if not ret:
                    st.error("Failed to access webcam. Please check your camera connection.")
                    break
                
                # Flip the frame horizontally for a more intuitive mirror view
                frame = cv2.flip(frame, 1)
                
                # Process the frame
                results = process_frame(frame, model_e, pose, confidence_threshold)
                
                # Update angles in sidebar if landmarks detected
                if results['landmarks_detected'] and results['angles']:
                    for joint, angle in results['angles'].items():
                        angle_displays[joint].markdown(
                            f'<div class="metric-card">{joint.replace("_", " ").title()} Angle: {angle:.2f}°</div>',
                            unsafe_allow_html=True
                        )
                
                # Update rep counter
                rep_counter.markdown(
                    f'<div class="metric-card"><h3>Rep Count: {st.session_state.counter}</h3></div>',
                    unsafe_allow_html=True
                )
                
                # Show feedback if a rep is counted and there's a posture issue
                if results['rep_counted'] and results['posture_issue']:
                    feedback = provide_feedback(results['posture_issue'])
                    if feedback:
                        message, sound_file = feedback
                        
                        if results['posture_issue'] == "correct":
                            feedback_placeholder.markdown(
                                f'<div class="feedback-box positive">{message}</div>',
                                unsafe_allow_html=True
                            )
                        else:
                            feedback_placeholder.markdown(
                                f'<div class="feedback-box negative">{message}</div>',
                                unsafe_allow_html=True
                            )
                        
                        pygame.mixer.music.load(sound_file)
                        pygame.mixer.music.play()
                        st.session_state.posture_status = []
                        st.session_state.previous_alert_time = time.time()
                
                # Display the processed frame
                frame_window.image(results['frame'], channels="RGB", use_column_width=True)
                
        except Exception as e:
            st.error(f"An error occurred: {e}")
        finally:
            cap.release()

# TAB 2: VIDEO UPLOAD MODE
with tab2:
    st.markdown('<h2 class="sub-header">Video Upload Analysis</h2>', unsafe_allow_html=True)
    
    # Instructions
    with st.expander("How to use video analysis"):
        st.write("""
        1. Upload a video file showing your exercise form
        2. Select your exercise type from the sidebar
        3. Click "Process Video" to analyze your form
        4. View the analysis results and feedback
        5. You can download the analysis report
        """)
    
    # File uploader
    st.markdown('<div class="upload-box">', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "mov", "avi"])
    st.markdown('</div>', unsafe_allow_html=True)
    
    if uploaded_file is not None:
        # Store the uploaded file in session state
        st.session_state.video_file_buffer = uploaded_file
        
        # Display a preview of the uploaded video
        st.video(uploaded_file)
        
        # Process button
        if st.button("Process Video"):
            st.session_state.processing_video = True
            
            # Create a progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()
            status_text.text("Processing video...")
            
            # Process the video
            video_results, video_fps = process_video_file(
                st.session_state.video_file_buffer, 
                model_e, 
                pose, 
                confidence_threshold,
                progress_bar
            )
            
            # Store results in session state
            st.session_state.video_results = video_results
            st.session_state.video_processed = True
            st.session_state.processing_video = False
            
            # Update status
            status_text.text("Video processing complete!")
            progress_bar.empty()
            
            # Automatically show results
            st.experimental_rerun()
    
    # Display results after processing
    if st.session_state.video_processed and st.session_state.video_results:
        st.markdown('<h3 class="sub-header">Analysis Results</h3>', unsafe_allow_html=True)
        
        # Results metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Reps", st.session_state.counter)
        
        # Posture issues summary
        posture_issues = [result['posture_issue'] for result in st.session_state.video_results 
                         if result['posture_issue'] is not None]
        
        # with col2:
        #     correct_percentage = sum(1 for issue in posture_issues if issue == "correct") / max(len(posture_issues), 1) * 100
        #     st.metric("Correct Form %", f"{correct_percentage:.1f}%")
        
        with col3:
            most_common_issue = most_frequent([i for i in posture_issues if i != "correct"]) if posture_issues else "None"
            st.metric("Most Common Issue", most_common_issue if most_common_issue else "None")
        
        # Display feedback on common issues
        if most_common_issue and most_common_issue != "None":
            feedback = provide_feedback(most_common_issue)
            if feedback:
                message, _ = feedback
                st.markdown(f'<div class="feedback-box negative"><h4>Key Form Issue:</h4>{message}</div>', 
                           unsafe_allow_html=True)
                
        # Option to play back analyzed video
        st.markdown('<h3 class="sub-header">Video Playback</h3>', unsafe_allow_html=True)
        
        # Display some analyzed frames as a gallery
        st.write("Key Frames:")
        
        # Select frames with issues for the gallery
        issue_frames = [result['frame'] for result in st.session_state.video_results 
                      if result['posture_issue'] is not None and result['posture_issue'] != "correct"]
        
        # If we have issue frames, show a gallery
        if issue_frames:
            # Select a sample of frames to show (to avoid showing too many)
            sample_size = min(5, len(issue_frames))
            sample_frames = [issue_frames[i] for i in range(0, len(issue_frames), max(1, len(issue_frames)//sample_size))][:sample_size]
            
            cols = st.columns(len(sample_frames))
            for i, col in enumerate(cols):
                with col:
                    st.image(sample_frames[i], use_column_width=True)
        else:
            st.info("No form issues detected in the video.")
        
        # Reset button
        if st.button("Upload Another Video"):
            st.session_state.video_processed = False
            st.session_state.video_file_buffer = None
            st.session_state.video_results = []
            st.session_state.counter = 0
            st.session_state.current_stage = ""
            st.session_state.posture_status = [None]
            st.experimental_rerun()