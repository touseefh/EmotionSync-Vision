import streamlit as st
import cv2
import mediapipe as mp
import tempfile
import yaml
import os
import numpy as np
import ffmpeg
from deepface import DeepFace
from gtts import gTTS
import threading
import queue
from collections import Counter
from moviepy.editor import VideoFileClip, AudioFileClip
import matplotlib.pyplot as plt

# Emotion-based suggestions
emotion_suggestions = {
    "angry": "You look angry! Take deep breaths and relax.",
    "happy": "You seem happy! Keep smiling and enjoy your day.",
    "sad": "You look sad. Try talking to someone or doing something fun.",
    "surprise": "You look surprised! Hope it's something good.",
    "neutral": "You seem neutral. Stay positive!",
    "fear": "You seem fearful. Stay strong, everything will be fine.",
    "disgust": "You look disgusted. Try focusing on something pleasant."
}

class HandFaceMeshApp:
    def __init__(self, config_path):
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)

        # Hand tracking setup
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            min_detection_confidence=self.config['min_detection_confidence'],
            min_tracking_confidence=self.config['min_tracking_confidence'])
        self.mp_drawing = mp.solutions.drawing_utils

        # Face Mesh setup
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            min_detection_confidence=self.config['face_min_detection_confidence'],
            min_tracking_confidence=self.config['face_min_tracking_confidence'])

        self.hand_drawing_spec = self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3)
        self.face_drawing_spec = self.mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=1, circle_radius=2)  # Blue in BGR

    def speak_text(self, text, output_file="suggestion.mp3"):
        tts = gTTS(text=text, lang='en')
        tts.save(output_file)
        return output_file

    def process_frame(self, image):
        # Rotate and resize to vertical (480x640)
        image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        image = cv2.resize(image, (480, 640))
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        return image, rgb_image

    def draw_hand_landmarks(self, image, hand_results):
        gesture_detected = False
        if hand_results.multi_hand_landmarks:
            gesture_detected = True
            for hand_landmarks in hand_results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    image, hand_landmarks, self.mp_hands.HAND_CONNECTIONS,
                    self.hand_drawing_spec, self.hand_drawing_spec)
        return gesture_detected

    def draw_face_landmarks(self, image, face_results):
        if face_results.multi_face_landmarks:
            for face_landmarks in face_results.multi_face_landmarks:
                self.mp_drawing.draw_landmarks(
                    image, face_landmarks, self.mp_face_mesh.FACEMESH_CONTOURS,
                    self.face_drawing_spec, self.face_drawing_spec)

    def process_video(self, input_path, output_path, result_queue):
        # Extract original audio
        video_clip = VideoFileClip(input_path)
        original_audio = video_clip.audio
        audio_path = "original_audio.mp3"
        original_audio.write_audiofile(audio_path, verbose=False, logger=None)
        original_audio.close()
        video_clip.close()

        cap = cv2.VideoCapture(input_path)
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps if fps > 0 else 0

        # Video dimensions for vertical output (480x640)
        temp_avi = output_path.replace(".mp4", ".avi")
        out = cv2.VideoWriter(temp_avi, fourcc, fps, (480, 640))

        emotions_detected = []
        emotion_confidences = []
        gesture_detected = False
        processed_frames = []

        frame_idx = 0
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            # Process frame for vertical orientation
            frame, rgb_image = self.process_frame(frame)

            # Detect hands
            hand_results = self.hands.process(rgb_image)
            if self.draw_hand_landmarks(frame, hand_results):
                gesture_detected = True
                cv2.putText(frame, "Gesture Detected", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Detect face mesh
            face_results = self.face_mesh.process(rgb_image)
            self.draw_face_landmarks(frame, face_results)

            # Detect emotions (every 10th frame for efficiency)
            if frame_idx % 10 == 0:
                try:
                    result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)[0]
                    emotion = result['dominant_emotion']
                    emotions_detected.append(emotion)
                    # Collect confidence scores for the dominant emotion
                    confidence = result['emotion'][emotion]  # Confidence score for the dominant emotion
                    emotion_confidences.append((emotion, confidence))
                    cv2.putText(frame, f"Emotion: {emotion} ({confidence:.1f}%)", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                except Exception:
                    pass

            processed_frames.append(frame)
            out.write(frame)
            frame_idx += 1

        cap.release()
        out.release()

        # Convert to MP4 and add original audio
        processed_clip = VideoFileClip(temp_avi)
        audio_clip = AudioFileClip(audio_path)
        final_clip = processed_clip.set_audio(audio_clip)
        final_clip.write_videofile(output_path, codec='libx264', audio_codec='aac', verbose=False, logger=None)
        processed_clip.close()
        audio_clip.close()
        final_clip.close()
        os.remove(temp_avi)
        os.remove(audio_path)

        # Analyze emotions and confidence
        if emotions_detected:
            emotion_counts = Counter(emotions_detected)
            total_frames_analyzed = len(emotions_detected)
            # Calculate average confidence for each emotion
            confidence_dict = {}
            for emotion, confidence in emotion_confidences:
                if emotion not in confidence_dict:
                    confidence_dict[emotion] = []
                confidence_dict[emotion].append(confidence)

            # Summarize emotions with percentages and average confidence
            emotion_summary = ", ".join([
                f"{emotion}: {count/total_frames_analyzed*100:.1f}% (Avg Confidence: {sum(confidence_dict[emotion])/len(confidence_dict[emotion]):.1f}%)"
                for emotion, count in emotion_counts.items()
            ])
            dominant_emotion = emotion_counts.most_common(1)[0][0]
            suggestion = emotion_suggestions.get(dominant_emotion, "Stay positive!")
        else:
            emotion_summary = "No emotions detected."
            suggestion = "Stay positive!"
            dominant_emotion = "neutral"

        # Generate AI voice feedback
        length_message = "The video is "
        if duration < 30:
            length_message += "less than 30 seconds."
        else:
            length_message += "30 seconds or more."
        gesture_message = "Hand gestures were detected." if gesture_detected else "No hand gestures detected."
        voice_text = f"{length_message} The emotions detected are: {emotion_summary}. {suggestion} {gesture_message}"
        audio_path = self.speak_text(voice_text)

        result_queue.put({
            "frames": processed_frames,
            "emotion_summary": emotion_summary,
            "suggestion": suggestion,
            "gesture": gesture_message,
            "audio_path": audio_path,
            "output_video_path": output_path,
            "emotion_counts": emotion_counts,  # Add emotion counts for visualization
            "confidence_dict": confidence_dict  # Add confidence data for visualization
        })

# Streamlit UI
st.set_page_config(layout="wide")
st.title("🎥 EmotionSync Vision")
st.write("AI Powered Emotion Detection System")

uploaded_file = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])
if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_input_file:
        temp_input_file.write(uploaded_file.read())
        input_video_path = temp_input_file.name

    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_output_file:
        output_video_path = temp_output_file.name

    # Load Config
    config_yaml = """
video_source: 0
min_detection_confidence: 0.5
min_tracking_confidence: 0.5
face_min_detection_confidence: 0.5
face_min_tracking_confidence: 0.5
    """
    config_file = "config.yaml"
    with open(config_file, "w") as file:
        file.write(config_yaml)

    app = HandFaceMeshApp(config_file)

    # Create a queue for threading
    result_queue = queue.Queue()

    # Start video processing in a separate thread
    st.write("Processing video, please wait...")
    processing_thread = threading.Thread(target=app.process_video, args=(input_video_path, output_video_path, result_queue))
    processing_thread.start()

    # Wait for processing to complete
    processing_thread.join()
    result = result_queue.get()

    # Display results
    st.success("Video processing complete!")

    # UI layout
    col1, col2 = st.columns([1, 2])
    with col1:
        st.write("### Processed Video")
        st.video(result["output_video_path"])  # Video will play with original audio

    with col2:
        st.write("### Analysis Results")
        st.markdown(f"**Emotions Detected:** {result['emotion_summary']}")
        st.markdown(f"**Suggestion:** {result['suggestion']}")
        st.markdown(f"**Gestures:** {result['gesture']}")

    # Play and allow download of AI voice
    st.write("### AI Voice Feedback")
    st.audio(result["audio_path"])
    with open(result["audio_path"], "rb") as audio_file:
        st.download_button("🔊 Download AI Voice", data=audio_file, file_name="analysis_suggestion.mp3", mime="audio/mp3")

    # Candidate Analytics Report with Visualization
    st.write("### Candidate Analytics Report")
    if result['emotion_counts']:
        # Prepare data for visualization
        emotions = list(result['emotion_counts'].keys())
        percentages = [count / sum(result['emotion_counts'].values()) * 100 for count in result['emotion_counts'].values()]
        avg_confidences = [sum(result['confidence_dict'][emotion]) / len(result['confidence_dict'][emotion]) if emotion in result['confidence_dict'] else 0 for emotion in emotions]

        # Create a bar chart for emotion percentages with confidence overlay
        fig, ax1 = plt.subplots(figsize=(6, 4))  # Reduced size from (10, 6) to (6, 4)

        # Bar chart for emotion percentages
        ax1.bar(emotions, percentages, color='skyblue', label='Percentage')
        ax1.set_xlabel('Emotions')
        ax1.set_ylabel('Percentage (%)', color='black')
        ax1.tick_params(axis='y', labelcolor='black')

        # Overlay line chart for average confidence
        ax2 = ax1.twinx()
        ax2.plot(emotions, avg_confidences, color='red', marker='o', label='Avg Confidence (%)')
        ax2.set_ylabel('Average Confidence (%)', color='red')
        ax2.tick_params(axis='y', labelcolor='red')

        # Title and legend
        plt.title('Candidate Emotion Analytics')
        fig.legend(loc='upper right', bbox_to_anchor=(0.9, 0.9))

        # Display the plot in Streamlit
        st.pyplot(fig)

    else:
        st.write("No emotion data available for visualization.")

    # Cleanup
    os.remove(input_video_path)
    os.remove(result["output_video_path"])
    os.remove(result["audio_path"])
    os.remove(config_file)