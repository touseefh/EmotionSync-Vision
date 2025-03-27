# EmotionSync Vision - AI Powered Emotion Detection System

## Overview
EmotionSync Vision is an advanced AI-powered system designed for real-time emotion and hand gesture detection in videos. Using DeepFace for facial emotion recognition and MediaPipe for hand tracking, the application provides insightful analysis and AI-generated voice feedback based on detected emotions and gestures.

This tool is ideal for applications such as:
- Interview analysis
- Behavioral assessment
- Emotion tracking in therapy sessions
- AI-assisted video analytics

## Features
- **Emotion Detection**: Uses DeepFace to analyze emotions (angry, happy, sad, surprise, neutral, fear, disgust).
- **Hand Gesture Recognition**: Uses MediaPipe to detect and track hand gestures.
- **Real-time Video Processing**: Processes frames efficiently to ensure accurate emotion and gesture analysis.
- **Audio Feedback**: Uses gTTS to generate AI voice feedback based on detected emotions.
- **Video Enhancement**: Processes videos in a vertical (portrait) format with annotations.
- **Interactive Streamlit UI**: Provides an easy-to-use web-based interface for users.
- **Analytics Visualization**: Displays emotion distributions and confidence levels graphically.
- **Audio-Preserved Video Output**: Ensures original audio remains hearable while processing the video.
- **Automatic Cleanup**: Removes temporary files after processing to save storage space.
- **Video Upload Limit**: Supports uploading videos of up to 20 seconds for processing.


## Installation

### Prerequisites
Ensure you have the following installed:
- Python 3.8+
- pip (Python package manager)

### Step 1: Clone the Repository
```sh
git clone https://github.com/touseefh/EmotionSync-Vision.git
cd EmotionSync-Vision
```

### Step 2: Install Dependencies
```sh
pip install -r requirements.txt
```

### Step 3: FFmpeg Setup
FFmpeg is required for processing videos and handling audio. Install it using:
- **Windows**: Download from https://ffmpeg.org/download.html and add it to the system PATH.
- **Linux/macOS**:
  ```bash
  sudo apt update && sudo apt install ffmpeg
  ```

## Running the Application
After installation, run the Streamlit app with:

```bash
streamlit run app.py
```

This will open a web browser with the application interface.

## How to Use
1. **Upload Video**: Click on "Upload a video" and select an MP4, AVI, or MOV file (max 20 seconds).
2. **Processing**: The system will analyze emotions and hand gestures in the video.
3. **View Results**:
   - Processed video will be displayed with annotations.
   - Detected emotions and hand gestures will be shown as text.
   - AI-generated audio feedback will be available for playback and download.
   - A graphical report of emotions and confidence levels will be displayed.
4. **Download**: Save the AI-generated audio feedback if needed.

## Configuration
A `config.yaml` file is automatically created with the following parameters:

```yaml
video_source: 0
min_detection_confidence: 0.5
min_tracking_confidence: 0.5
face_min_detection_confidence: 0.5
face_min_tracking_confidence: 0.5
```

Modify these parameters to adjust the confidence thresholds for hand and face tracking.



## Technologies Used
- **Streamlit**: Web UI framework.
- **OpenCV**: Image and video processing.
- **MediaPipe**: Hand and face landmark detection.
- **DeepFace**: Emotion recognition.
- **gTTS**: Text-to-speech for AI-generated audio feedback.
- **FFmpeg & MoviePy**: Video and audio processing.
- **Matplotlib**: Data visualization.
- **PyYAML**: Configuration management.
- **NumPy**: Numerical processing.

## Troubleshooting
- **DeepFace Not Detecting Faces**: Ensure faces are clearly visible in the video.
- **Audio Not Playing**: Check FFmpeg installation and ensure proper audio codecs.
- **Video Processing Slow**: Reduce the `frame_idx % 10 == 0` condition to `frame_idx % 20 == 0` in `process_video`.

## Future Improvements
- **Live Video Streaming**: Support for real-time analysis via webcam.
- **Multi-Person Emotion Detection**: Extend detection for multiple faces.
- **Expanded Gesture Recognition**: Identify specific hand signs and gestures.
- **Enhanced Voice Feedback**: More natural and expressive AI-generated voices.

## Author
Developed by **HireSync.ai** 🚀

