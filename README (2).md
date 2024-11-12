
# Video to Sign Language Translator

This project converts video input with spoken language into sign language representations using Python. 
The code extracts audio from video, translates the audio to text, and then displays corresponding sign language images in a slideshow format.

## Features
- Audio extraction from video files
- Speech-to-text translation
- Sign language display for recognized words

## Dependencies
The following Python packages are required:
- `moviepy`
- `SpeechRecognition`
- `gTTS` (Google Text-to-Speech)
- `pydub`
- `Pillow` (for displaying images)

To install them, run:
```bash
pip install moviepy SpeechRecognition gTTS pydub Pillow
```

## Getting Started

1. **Upload Video**: Place your video file in the directory or specify its path when prompted.
2. **Run the Script**: Execute the `t2s.py` script.

## Usage
Run the following command to execute the code:
```bash
python t2s.py
```

## Functionality
1. The script extracts audio from the uploaded video file.
2. Uses speech recognition to convert the audio to text.
3. Maps the text to corresponding sign language images and displays them in a slideshow.

## Notes
- This script runs in a Jupyter Notebook format originally but is converted here for Python script execution.
- Ensure that the video file is accessible for the script.

