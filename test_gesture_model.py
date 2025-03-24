import streamlit as st
import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import tensorflow as tf
import gtts
import os
import threading
import playsound
import time
from PIL import Image
import tempfile

# Set page config
st.set_page_config(
    page_title="Hand Gesture to Speech",
    page_icon="👋",
    layout="wide"
)

# Add title and description
st.title("Hand Gesture Recognition & Speech")
st.markdown("""
    This application detects hand gestures to recognize American Sign Language (ASL) alphabets and speaks them out loud.
    Position your hand in front of the camera to see it in action!
""")

# Sidebar for controls
with st.sidebar:
    st.header("Settings")
    detection_confidence = st.slider("Detection Confidence", 0.5, 1.0, 0.8, 0.05)
    buffer_size = st.slider("Prediction Stability (frames)", 1, 10, 3, 1)
    speech_enabled = st.checkbox("Enable Speech", value=True)
    
    st.header("Instructions")
    st.markdown("""
    1. Allow camera access when prompted
    2. Show your hand gesture in the camera
    3. Hold steady for best results
    4. The app will recognize and speak the letter
    5. Press Stop to end the session
    """)
    
    # Display the ASL alphabet reference
    st.header("ASL Alphabet Reference")
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/3/3c/ASLdigits.jpg/600px-ASLdigits.jpg", 
             caption="ASL Alphabet Reference")

# Function to convert text to speech without blocking
@st.cache_resource
def get_speech_engine():
    def speak_text(text):
        def play_audio():
            try:
                tts = gtts.gTTS(text)
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp3')
                temp_filename = temp_file.name
                temp_file.close()
                
                tts.save(temp_filename)
                playsound.playsound(temp_filename, block=True)
                
                # Clean up temp file
                time.sleep(0.5)
                try:
                    os.remove(temp_filename)
                except Exception as e:
                    pass
            except Exception as e:
                st.error(f"Error in speech: {e}")
        
        if speech_enabled:
            threading.Thread(target=play_audio, daemon=True).start()
    
    return speak_text

# Load model
@st.cache_resource
def load_gesture_model():
    try:
        model = tf.keras.models.load_model("gesture_model_finetuned2.keras")
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Initialize components
detector = None
model = None
gesture_labels = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
speak_text = get_speech_engine()

# Initialize session state variables
if 'previous_prediction' not in st.session_state:
    st.session_state.previous_prediction = None
if 'prediction_buffer' not in st.session_state:
    st.session_state.prediction_buffer = []
if 'predicted_label' not in st.session_state:
    st.session_state.predicted_label = ""
if 'confidence' not in st.session_state:
    st.session_state.confidence = 0.0
if 'running' not in st.session_state:
    st.session_state.running = False

# Main interface layout with columns
col1, col2 = st.columns([2, 1])

# Create placeholders for video and skeleton
with col1:
    video_placeholder = st.empty()
with col2:
    skeleton_placeholder = st.empty()
    prediction_text = st.empty()
    confidence_text = st.empty()
    
    # Add prediction history
    st.subheader("Prediction History")
    history_placeholder = st.empty()
    prediction_history = []

# Button columns
button_col1, button_col2 = st.columns(2)

# Start and stop buttons
with button_col1:
    start_button = st.button("Start Camera", key="start_button", disabled=st.session_state.running)

with button_col2:
    stop_button = st.button("Stop Camera", key="stop_button", disabled=not st.session_state.running)

# Main app logic
if start_button:
    st.session_state.running = True
    
    # Load the model
    model = load_gesture_model()
    if model is None:
        st.error("Failed to load the gesture recognition model. Please check if the model file exists.")
        st.session_state.running = False
        st.experimental_rerun()
    
    # Initialize detector
    detector = HandDetector(maxHands=1, detectionCon=detection_confidence)

# Use a placeholder for the camera feed container
camera_container = st.empty()

# Function to run the camera processing
def process_camera():
    cap = cv2.VideoCapture(0)
    
    while st.session_state.running and cap.isOpened():
        success, frame = cap.read()
        if not success:
            st.error("Failed to capture image from camera")
            break
            
        # Process frame
        frame = cv2.flip(frame, 1)  # Mirror effect
        display_frame = frame.copy()
        
        # Detect hands
        hands, img = detector.findHands(frame, draw=False, flipType=True)
        
        # Create skeleton canvas
        skeleton_canvas = np.ones((224, 224, 3), dtype=np.uint8) * 255
        
        if hands:
            hand = hands[0]
            
            # Draw landmarks on skeleton canvas
            if 'lmList' in hand:
                landmarks = hand['lmList']
                
                # Define connections for hand skeleton
                connections = [
                    (0, 1), (1, 2), (2, 3), (3, 4),  # Thumb
                    (0, 5), (5, 6), (6, 7), (7, 8),  # Index finger
                    (0, 9), (9, 10), (10, 11), (11, 12),  # Middle finger
                    (0, 13), (13, 14), (14, 15), (15, 16),  # Ring finger
                    (0, 17), (17, 18), (18, 19), (19, 20),  # Pinky
                    (5, 9), (9, 13), (13, 17)  # Palm connections
                ]
                
                # Scale landmarks
                x, y, w, h = hand['bbox']
                scaled_landmarks = []
                
                for lm in landmarks:
                    x_point, y_point, _ = lm
                    scaled_x = int((x_point - x) * 224 / max(w, 1))
                    scaled_y = int((y_point - y) * 224 / max(h, 1))
                    scaled_landmarks.append((scaled_x, scaled_y))
                
                # Draw connections on skeleton
                for connection in connections:
                    start_idx, end_idx = connection
                    if start_idx < len(scaled_landmarks) and end_idx < len(scaled_landmarks):
                        pt1 = scaled_landmarks[start_idx]
                        pt2 = scaled_landmarks[end_idx]
                        cv2.line(skeleton_canvas, pt1, pt2, (0, 255, 0), 2)
                
                # Draw landmark points
                for point in scaled_landmarks:
                    cv2.circle(skeleton_canvas, point, 3, (0, 0, 255), -1)
                
                # Make prediction
                skeleton_input = np.expand_dims(skeleton_canvas, axis=0)
                skeleton_input = skeleton_input / 255.0
                
                prediction = model.predict(skeleton_input, verbose=0)
                predicted_idx = np.argmax(prediction)
                st.session_state.confidence = float(np.max(prediction))
                
                # Add to prediction buffer
                st.session_state.prediction_buffer.append(predicted_idx)
                
                # Stabilize prediction
                if len(st.session_state.prediction_buffer) >= buffer_size:
                    most_common = max(set(st.session_state.prediction_buffer), key=st.session_state.prediction_buffer.count)
                    st.session_state.predicted_label = gesture_labels[most_common]
                    
                    # Speak only if prediction changes
                    if st.session_state.predicted_label != st.session_state.previous_prediction:
                        speak_text(st.session_state.predicted_label)
                        st.session_state.previous_prediction = st.session_state.predicted_label
                        
                        # Add to history
                        if len(prediction_history) > 15:  # Keep last 15 predictions
                            prediction_history.pop(0)
                        prediction_history.append(st.session_state.predicted_label)
                    
                    # Keep buffer at fixed size
                    st.session_state.prediction_buffer = st.session_state.prediction_buffer[-buffer_size:]
                
                # Draw hand landmarks on display frame
                for connection in connections:
                    start_idx, end_idx = connection
                    if start_idx < len(landmarks) and end_idx < len(landmarks):
                        start_point = (landmarks[start_idx][0], landmarks[start_idx][1])
                        end_point = (landmarks[end_idx][0], landmarks[end_idx][1])
                        cv2.line(display_frame, start_point, end_point, (0, 255, 0), 2)
                
                # Draw points
                for lm in landmarks:
                    cv2.circle(display_frame, (lm[0], lm[1]), 3, (0, 0, 255), -1)
        else:
            # Clear prediction buffer when no hand is detected
            st.session_state.prediction_buffer = []
        
        # Display the predicted alphabet on frame
        cv2.putText(display_frame, f"Detected: {st.session_state.predicted_label}", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Convert frames to RGB for Streamlit
        display_frame_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
        skeleton_rgb = cv2.cvtColor(skeleton_canvas, cv2.COLOR_BGR2RGB)
        
        # Display video and skeleton
        video_placeholder.image(display_frame_rgb, channels="RGB", use_container_width=True)
        skeleton_placeholder.image(skeleton_rgb, channels="RGB", use_container_width=True, caption="Hand Skeleton")
        
        # Update prediction display
        prediction_text.markdown(f"<h1 style='text-align: center; font-size: 80px;'>{st.session_state.predicted_label}</h1>", unsafe_allow_html=True)
        confidence_text.progress(st.session_state.confidence)
        
        # Update prediction history
        history_placeholder.markdown(" ".join([f"<span style='font-size: 24px; margin: 5px;'>{letter}</span>" for letter in prediction_history]), unsafe_allow_html=True)
        
        # Check if stop button clicked (handled externally)
        if not st.session_state.running:
            break
            
        # Wait to reduce CPU usage
        time.sleep(0.03)
    
    # Release resources
    cap.release()
    st.session_state.running = False
    st.experimental_rerun()  # Refresh the UI

# Handle stop button
if stop_button:
    st.session_state.running = False
    st.success("Camera stopped")
    st.experimental_rerun()

# Run camera processing if in running state
if st.session_state.running:
    with st.spinner('Starting camera...'):
        process_camera()
else:
    # Display placeholder image when not started
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/3/3c/ASLdigits.jpg/600px-ASLdigits.jpg", 
             caption="Start the camera to begin gesture recognition", use_container_width=True)