# Required installations (uncomment and run if needed):
# !pip install moviepy SpeechRecognition gTTS pydub

!pip install moviepy SpeechRecognition gTTS pydub


import moviepy.editor as mp
import speech_recognition as sr
import os
from gtts import gTTS
from IPython.display import Audio, display
import matplotlib.pyplot as plt
import glob
from google.colab import files


# Upload your video
uploaded = files.upload()
video_path = list(uploaded.keys())[0]

# Extract audio
video = mp.VideoFileClip(video_path)
audio_path = "extracted_audio.wav"
video.audio.write_audiofile(audio_path)


# Initialize recognizer
recognizer = sr.Recognizer()

# Load the audio
with sr.AudioFile(audio_path) as source:
    audio_data = recognizer.record(source)

# Convert audio to text
try:
    text = recognizer.recognize_google(audio_data)
    print("Text:", text)
except sr.UnknownValueError:
    print("Could not understand audio")
except sr.RequestError:
    print("Could not request results; check your internet connection")

output_text = text


import time
import IPython.display as display
from PIL import Image
import os

# Dictionary mapping each letter to the corresponding sign language image path
letter_to_sign = {
    'A': '/content/drive/MyDrive/bfy/A.jpg',
    'B': '/content/drive/MyDrive/bfy/B.jpeg',
    'C': '/content/drive/MyDrive/bfy/C.jpeg',
    'D': '/content/drive/MyDrive/bfy/D.jpeg',
    'E': '/content/drive/MyDrive/bfy/E.jpeg',
    'F': '/content/drive/MyDrive/bfy/F.jpeg',
    'G': '/content/drive/MyDrive/bfy/G.jpeg',
    'H': '/content/drive/MyDrive/bfy/H.jpeg',
    'I': '/content/drive/MyDrive/bfy/I.jpeg',
    'J': '/content/drive/MyDrive/bfy/J.jpeg',
    'K': '/content/drive/MyDrive/bfy/K.jpeg',
    'L': '/content/drive/MyDrive/bfy/L.jpeg',
    'M': '/content/drive/MyDrive/bfy/M.jpeg',
    'N': '/content/drive/MyDrive/bfy/N.jpeg',
    'O': '/content/drive/MyDrive/bfy/O.jpeg',
    'P': '/content/drive/MyDrive/bfy/P.jpeg',
    'Q': '/content/drive/MyDrive/bfy/Q.jpeg',
    'R': '/content/drive/MyDrive/bfy/R.jpeg',
    'S': '/content/drive/MyDrive/bfy/S.jpeg',
    'T': '/content/drive/MyDrive/bfy/T.jpeg',
    'U': '/content/drive/MyDrive/bfy/U.jpeg',
    'V': '/content/drive/MyDrive/bfy/V.jpeg',
    'W': '/content/drive/MyDrive/bfy/W.jpeg',
    'X': '/content/drive/MyDrive/bfy/X.jpeg',
    'Y': '/content/drive/MyDrive/bfy/Y.jpeg',
    'Z': '/content/drive/MyDrive/bfy/Z.jpeg',
}

# Function to display sign language images in a slideshow format
def display_sign_language_slideshow(text, delay=1.0):
    # Convert text to uppercase and retain only alphabetic characters and spaces
    filtered_text = ''.join([char.upper() for char in text if char.isalpha() or char == ' '])

    for char in filtered_text:
        display.clear_output(wait=True)

        if char == ' ':
            # Display "New Word" for spaces
            img = Image.new("RGB", (200, 100), color=(255, 255, 255))
            display.display(img)
            print("New Word")
        else:
            # Display the corresponding sign language image
            image_path = letter_to_sign.get(char)

            if image_path and os.path.exists(image_path):
                img = Image.open(image_path)
                display.display(img)
                print(f"Character: {char}")
            else:
                print(f"Image for '{char}' not found.")




display_sign_language_slideshow(output_text, delay=2)


