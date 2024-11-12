#!/usr/bin/env python
# coding: utf-8

# In[ ]:


pip install --upgrade youtube-dl


# In[ ]:


pip install yt-dlp


# In[ ]:


import yt_dlp as youtube_dl

def download_video(url, output_path="downloaded_video.mp4"):
    ydl_opts = {
        'outtmpl': output_path,
        'format': 'bestvideo+bestaudio/best',
    }

    try:
        with youtube_dl.YoutubeDL(ydl_opts) as ydl:
            print("Downloading video...")
            ydl.download([url])
        print(f"Video downloaded successfully as {output_path}")
    except Exception as e:
        print(f"Error downloading video: {e}")

video_url = input("Enter the YouTube video URL: ")
download_video(video_url)



# In[ ]:


import yt_dlp as youtube_dl
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import time
def download_video(url, output_path="downloaded_video.mp4"):
    ydl_opts = {
        'outtmpl': output_path,
        'format': 'bestvideo+bestaudio/best',
    }

    try:
        with youtube_dl.YoutubeDL(ydl_opts) as ydl:
            print("Downloading video...")
            ydl.download([url])
        print(f"Video downloaded successfully as {output_path}")
    except Exception as e:
        print(f"Error downloading video: {e}")

def extract_frames(video_path="downloaded_video.mp4"):
    cap = cv2.VideoCapture(video_path)
    frames = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)

    cap.release()
    return frames


def load_asl_model():
    try:
        model = load_model('sign_language_model.h5')
        print("Model loaded successfully.")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def predict_sign_language(frames, model):
    predictions = []
    for frame in frames:

        frame_resized = cv2.resize(frame, (224, 224))
        frame_array = img_to_array(frame_resized)
        frame_array = np.expand_dims(frame_array, axis=0)
        frame_array /= 255.0

        prediction = model.predict(frame_array)
        predicted_class = np.argmax(prediction)
        predictions.append(predicted_class)
    return predictions

def convert_predictions_to_text(predictions):
    sign_language_map = {0: "Hello", 1: "Thank you", 2: "Goodbye", 3: "Please", 4: "Sorry"}
    recognized_text = [sign_language_map.get(prediction, "Unknown") for prediction in predictions]
    return ' '.join(recognized_text)

def main():
    video_url = input("Enter the YouTube video URL: ")
    download_video(video_url)
    print("Extracting frames from the video...")
    frames = extract_frames()
    model = load_asl_model()

    if model:
        print("Predicting sign language from frames...")
        predictions = predict_sign_language(frames, model)
        recognized_text = convert_predictions_to_text(predictions)
        print(f"Recognized Text: {recognized_text}")
    else:
        print("Model not available for predictions.")

if __name__ == "__main__":
    main()


# In[ ]:





# In[ ]:


def mock_predict_sign_language(frames):

    predictions = ['Hello', 'Age', 'Name', 'Address', 'Native place']
    return predictions

def convert_predictions_to_text(predictions):
    return ' '.join(predictions)

def main():
    video_url = input("Enter the YouTube video URL: ")
    download_video(video_url)

    print("Extracting frames from the video...")
    frames = extract_frames()

    print("Predicting sign language from frames (mock prediction)...")
    predictions = mock_predict_sign_language(frames)

    recognized_text = convert_predictions_to_text(predictions)
    print(f"Recognized Text: {recognized_text}")

if __name__ == "__main__":
    main()


# In[ ]:


import yt_dlp as youtube_dl
import cv2
import numpy as np
import os

# Step 1: Download Video from YouTube
def download_video(url):
    ydl_opts = {
        'format': 'best',
        'outtmpl': 'downloaded_video.mp4',
        'verbose': True
    }

    try:
        print("Downloading video...")
        with youtube_dl.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        print("Video downloaded successfully.")
    except Exception as e:
        print(f"Error downloading video: {str(e)}")
        return None
    return 'downloaded_video.mp4'

# Step 2: Extract Frames from Video
def extract_frames(video_path):
    # Initialize video capture
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Could not open video.")
        return []

    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames


def mock_predict_sign_language(frames):

    predictions = ['Hello', 'Age', 'Name', 'Address', 'Native place']
    return predictions

def convert_predictions_to_text(predictions):
    return ' '.join(predictions)

def main():
    video_url = input("Enter the YouTube video URL: ")

    # Download video
    video_path = download_video(video_url)
    if not video_path:
        return

    print("Extracting frames from the video...")
    frames = extract_frames(video_path)

    if not frames:
        print("No frames extracted.")
        return


    print("Predicting sign language from frames (mock prediction)...")
    predictions = mock_predict_sign_language(frames)

    recognized_text = convert_predictions_to_text(predictions)
    print(f"Recognized Text: {recognized_text}")

if __name__ == "__main__":
    main()


# In[ ]:


pip install gTTS


# In[ ]:


import os
from gtts import gTTS
import cv2


def download_video(video_url):
    os.system(f"yt-dlp -o 'downloaded_video.mp4' {video_url}")

def extract_frames(video_path):

    cap = cv2.VideoCapture(video_path)
    frames = []

    if not cap.isOpened():
        print("Error: Could not open video.")
        return frames

    # Extract frames
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)

    cap.release()
    print(f"Extracted {len(frames)} frames.")
    return frames
def mock_predict_sign_language(frames):
    predictions = ['Hello', 'Age', 'Name', 'Address', 'Native place']
    return predictions


def convert_predictions_to_text(predictions):
    return ' '.join(predictions)

def text_to_audio(recognized_text):
    tts = gTTS(text=recognized_text, lang='en')

    audio_file = "recognized_text_audio.mp3"
    tts.save(audio_file)
    print(f"Audio file saved as {audio_file}")
    os.system(f"start {audio_file}")  # Windows


# Main Process
def main():
    video_url = input("Enter the YouTube video URL: ")
    download_video(video_url)
    print("Extracting frames from the video...")
    video_path = 'downloaded_video.mp4'
    frames = extract_frames(video_path)
    print("Predicting sign language from frames (mock prediction)...")
    predictions = mock_predict_sign_language(frames)
    recognized_text = convert_predictions_to_text(predictions)
    print(f"Recognized Text: {recognized_text}")
    text_to_audio(recognized_text)

if __name__ == "__main__":
    main()


# In[ ]:


import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Set the path to your dataset
DATASET_PATH = r'C:\Users\Harsh\Downloads\asl_alphabet_test\asl_alphabet_test'  # Update with the actual path

def load_and_preprocess_data(img_size=(64, 64)):
    images, labels = [], []
    label_map = {}  # Maps label indices to text

    for idx, label in enumerate(os.listdir(DATASET_PATH)):
        label_path = os.path.join(DATASET_PATH, label)
        if os.path.isdir(label_path):
            label_map[idx] = label  # Map each folder to a label
            for image_file in os.listdir(label_path):
                img_path = os.path.join(label_path, image_file)
                img = tf.keras.preprocessing.image.load_img(img_path, target_size=img_size)
                img_array = tf.keras.preprocessing.image.img_to_array(img)
                images.append(img_array)
                labels.append(idx)

    images = np.array(images) / 255.0  # Normalize images
    labels = np.array(labels)
    X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test, label_map


# In[ ]:


from tensorflow.keras import layers, models

def create_model(input_shape=(64, 64, 3), num_classes=26):  # Adjust num_classes based on the dataset
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def train_and_save_model():
    X_train, X_test, y_train, y_test, label_map = load_and_preprocess_data()
    model = create_model(num_classes=len(label_map))
    model.summary()
    datagen = ImageDataGenerator(rotation_range=20, width_shift_range=0.2, height_shift_range=0.2,
                                 shear_range=0.2, zoom_range=0.2, horizontal_flip=True, fill_mode='nearest')

    model.fit(datagen.flow(X_train, y_train, batch_size=32), epochs=10, validation_data=(X_test, y_test))

    # Save the trained model
    model.save('model.h5')
    print("Model saved as model.h5")


# In[ ]:


import os
print(os.path.exists(DATASET_PATH))  # Should return True if the path is correct


# In[ ]:


import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models

# Set the path to your dataset
DATASET_PATH = r'C:\Users\Harsh\Downloads\archive (10)\asl_dataset'

def load_and_preprocess_data(img_size=(64, 64)):
    images, labels = [], []
    label_map = {}
    for idx, label in enumerate(os.listdir(DATASET_PATH)):
        label_path = os.path.join(DATASET_PATH, label)
        if os.path.isdir(label_path):
            label_map[idx] = label
            for image_file in os.listdir(label_path):
                img_path = os.path.join(label_path, image_file)
                img = tf.keras.preprocessing.image.load_img(img_path, target_size=img_size)
                img_array = tf.keras.preprocessing.image.img_to_array(img)
                images.append(img_array)
                labels.append(idx)
    images = np.array(images) / 255.0
    labels = np.array(labels)
    X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test, label_map

def create_model(input_shape=(64, 64, 3), num_classes=26):
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def train_and_save_model():
    X_train, X_test, y_train, y_test, label_map = load_and_preprocess_data()
    model = create_model(num_classes=len(label_map))
    model.summary()

    datagen = ImageDataGenerator(rotation_range=20, width_shift_range=0.2, height_shift_range=0.2,
                                 shear_range=0.2, zoom_range=0.2, horizontal_flip=True, fill_mode='nearest')

    model.fit(datagen.flow(X_train, y_train, batch_size=32), epochs=10, validation_data=(X_test, y_test))

    # Attempt to save the model with error handling
    try:
        model.save('model.h5')
        print("Model successfully saved as model.h5")
    except Exception as e:
        print(f"Error saving model: {e}")

train_and_save_model()


# In[ ]:


get_ipython().system('pip install kaggle')



# In[ ]:


kaggle datasets download -d jorgefernandopr/american-sign-language-digits


# In[ ]:


pip install yt-dlp


# In[ ]:


import yt_dlp

def download_video(video_url, output_path='video.mp4'):
    ydl_opts = {
        'format': 'bestvideo+bestaudio/best',
        'outtmpl': output_path,  # Set the output filename
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([video_url])

# Call the function with your YouTube video URL
download_video('https://youtu.be/sZLlnStyzT4?si=0U_II1fXGJbYKA3p', 'sign_language_video.mp4')


# In[ ]:


import cv2
import os

def extract_frames(video_path, output_folder='frames', frame_rate=30):
    """
    Extract frames from the video and save them to the specified output folder.

    :param video_path: Path to the input video file
    :param output_folder: Folder to save the extracted frames
    :param frame_rate: Frame extraction rate (frames per second)
    """
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Open the video file
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    # Get the frame rate of the video
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Video frame rate: {fps} frames per second.")

    # Initialize frame count
    frame_count = 0
    frame_num = 0

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        # Save a frame every "frame_rate" frames (or adjust as needed)
        if frame_count % int(fps // frame_rate) == 0:
            frame_filename = os.path.join(output_folder, f'frame_{frame_num:04d}.jpg')
            cv2.imwrite(frame_filename, frame)
            print(f"Saved: {frame_filename}")
            frame_num += 1

        frame_count += 1

    # Release the video capture object
    cap.release()
    print("Frame extraction completed.")

# Example usage
extract_frames('sign_language_video.mp4', output_folder='frames', frame_rate=1)


# In[ ]:


import os

# Check the number of images in each class folder
for root, dirs, files in os.walk('/content/frames'):
    if files:
        print(f"Files in {root}: {len(files)}")
    else:
        print(f"No files in {root}")


# In[ ]:


import pandas as pd
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array

# Path to the frames
frames_dir = '/content/frames'

# Get all the image file paths
image_files = [f for f in os.listdir(frames_dir) if f.endswith('.jpg')]

# Create a DataFrame with image file paths
df = pd.DataFrame(image_files, columns=['filename'])

# Create an ImageDataGenerator instance
datagen = ImageDataGenerator(rescale=1./255)

# Define a generator for loading images from the DataFrame
def image_generator(df, batch_size=32):
    while True:
        for i in range(0, len(df), batch_size):
            batch_filenames = df['filename'][i:i+batch_size]
            images = []
            for filename in batch_filenames:
                img_path = os.path.join(frames_dir, filename)
                img = load_img(img_path, target_size=(224, 224))  # Resize if needed
                img_array = img_to_array(img)
                images.append(img_array)
            yield np.array(images)

# Example usage of the generator
train_gen = image_generator(df)

# Get a batch of images
images = next(train_gen)
print(images.shape)


# In[ ]:


from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing import image
import numpy as np

# Load a pre-trained MobileNetV2 model (without the top layer)
model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Example: Pass the batch of images through the model
features = model.predict(images)

print(features.shape)  # Check the shape of the output (features)


# In[ ]:


import matplotlib.pyplot as plt

# Display the first image in the batch
plt.imshow(images[0].astype('uint8'))  # Convert to uint8 for correct display
plt.show()


# In[ ]:


import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Set the absolute path to the frames directory
frames_dir = '/content/frames'

datagen = ImageDataGenerator(rescale=1./255)

# Load the images using the directory path
train_data = datagen.flow_from_directory(
    frames_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

# Print the class labels
print(train_data.class_indices)


# In[ ]:


from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing import image
import numpy as np

# Load a pre-trained MobileNetV2 model (without the top layer)
model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Example: Pass the batch of images through the model
features = model.predict(images)

print(features.shape)  # Check the shape of the output (features)


# In[ ]:


from tensorflow.keras.models import load_model

# Load your custom-trained model
model = load_model('path_to_your_model.h5')

# Predict on the batch of images
predictions = model.predict(images)

# Check the predictions (for classification, the output might be probabilities)
print(predictions)


# In[ ]:


import os
import numpy as np
import cv2
from tensorflow.keras.preprocessing.image import ImageDataGenerator


dataset_dir = '/content/frames'


datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

# Train generator
train_generator = datagen.flow_from_directory(
    dataset_dir,
    target_size=(64, 64),
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

# Validation generator
validation_generator = datagen.flow_from_directory(
    dataset_dir,
    target_size=(64, 64),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)


# In[ ]:


VIDEO_PATH = 'sign_language_video.mp4'  # Path to the downloaded video file


# In[ ]:


from tensorflow.keras.preprocessing.image import ImageDataGenerator

def load_and_preprocess_data(img_size=(64, 64), batch_size=32, train_dir='path/to/training/data'):
    # Create an ImageDataGenerator to load and preprocess images
    datagen = ImageDataGenerator(rescale=1./255)  # Normalize the images to [0, 1]

    # Load training data from a directory with subdirectories for each class (letter A-Z)
    train_data = datagen.flow_from_directory(
        train_dir,  # Path to your training data directory
        target_size=img_size,  # Resize images to 64x64
        batch_size=batch_size,
        class_mode='sparse'  # Use sparse labels (integers) for multi-class classification
    )

    return train_data


# In[ ]:


import os

# Verify the directory and its contents
train_dir = '/content/frames'  # Your path
if os.path.isdir(train_dir):
    print("Directory found!")
    for subdir, dirs, files in os.walk(train_dir):
        print(subdir, len(files))
else:
    print("Directory not found.")


# In[ ]:


import os
import shutil

source_dir = '/content/frames'  # Current directory containing 127 files
output_dir = '/content/frames_sorted'  # New directory to organize files into subfolders

# Create subdirectories for each letter (A to Z)
letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
for letter in letters:
    os.makedirs(os.path.join(output_dir, letter), exist_ok=True)

# Example: Randomly assign files to subdirectories (you should organize your dataset according to your needs)
import random

files = os.listdir(source_dir)
for file in files:
    if file.endswith('.jpg') or file.endswith('.png'):
        letter = random.choice(letters)  # Assign a random letter (A-Z) to each file
        shutil.move(os.path.join(source_dir, file), os.path.join(output_dir, letter, file))

print("Files have been reorganized into subdirectories.")


# In[ ]:


from tensorflow.keras.preprocessing.image import ImageDataGenerator

def load_and_preprocess_data(img_size=(64, 64), batch_size=32, train_dir='/content/frames_sorted'):
    # Create an ImageDataGenerator to load and preprocess images
    datagen = ImageDataGenerator(rescale=1./255)  # Normalize the images to [0, 1]
    train_data = datagen.flow_from_directory(
        train_dir,  # Path to your training data directory
        target_size=img_size,  # Resize images to 64x64
        batch_size=batch_size,
        class_mode='sparse'  # Use sparse labels (integers) for multi-class classification
    )

    return train_data

# Define the path to the sorted training dataset
train_dir = '/content/frames_sorted'  # Update this to the new directory

# Load and preprocess the data
train_data = load_and_preprocess_data(train_dir=train_dir)

# Create the model
model = create_model()

# Train the model
model.fit(train_data, epochs=10)

# Save the trained model
model.save('model.h5')


# In[ ]:


def predict_from_video(model, video_path, output_dir, frame_rate=30):
    # Check if output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Output directory '{output_dir}' created.")

    # Extract frames from the video
    extract_frames(video_path, output_dir, frame_rate)

    predictions = []
    # Check if extracted frames directory has images
    if not os.listdir(output_dir):
        print(f"No frames found in directory {output_dir}. Make sure frames were extracted.")
        return

    # Define class labels manually (update this based on your classes)
    class_labels = ['class_0', 'class_1', 'class_2', 'class_3', 'class_4', 'class_5',
                    'class_6', 'class_7', 'class_8', 'class_9', 'class_10', 'class_11',
                    'class_12', 'class_13', 'class_14', 'class_15', 'class_16', 'class_17',
                    'class_18', 'class_19', 'class_20', 'class_21', 'class_22', 'class_23',
                    'class_24', 'class_25']  # Example, adjust according to your classes

    for frame_filename in os.listdir(output_dir):
        frame_path = os.path.join(output_dir, frame_filename)

        # Preprocess the frame
        img_array = preprocess_frame(frame_path)

        # Make prediction
        prediction = model.predict(img_array)
        predicted_class = np.argmax(prediction, axis=-1)  # Get the class with highest probability

        # Store prediction
        predictions.append(predicted_class[0])

        # Convert class index to label
        predicted_label = class_labels[predicted_class[0]]
        print(f"Frame {frame_filename}: Predicted gesture - {predicted_label}")

    # Optional: Aggregate predictions over frames
    aggregated_predictions = aggregate_predictions(predictions)
    print(f"Aggregated Predictions: {aggregated_predictions}")


# In[ ]:


import cv2
import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Load the trained model
model = load_model('model.h5')

# Define class labels manually (adjust based on your dataset)
class_labels = ['class_0', 'class_1', 'class_2', 'class_3', 'class_4', 'class_5',
                'class_6', 'class_7', 'class_8', 'class_9', 'class_10', 'class_11',
                'class_12', 'class_13', 'class_14', 'class_15', 'class_16', 'class_17',
                'class_18', 'class_19', 'class_20', 'class_21', 'class_22', 'class_23',
                'class_24', 'class_25']  # Example, adjust according to your classes

def extract_frames(video_path, output_dir, frame_rate=30):
    # Check if video file exists
    if not os.path.exists(video_path):
        print(f"Error: Video file '{video_path}' not found.")
        return

    # Open the video file
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error opening video stream or file")
        return

    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Output directory '{output_dir}' created.")

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Save frames at a specified frame rate
        if frame_count % frame_rate == 0:
            frame_filename = os.path.join(output_dir, f"frame_{frame_count}.jpg")
            cv2.imwrite(frame_filename, frame)  # Save the frame as an image
        frame_count += 1

    cap.release()
    print(f"Frames saved to {output_dir}")

def preprocess_frame(frame_path, target_size=(64, 64)):
    # Load and preprocess the image frame
    img = image.load_img(frame_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array /= 255.0  # Rescale the image
    return img_array

def aggregate_predictions(predictions):
    # Aggregate predictions (for example: majority voting)
    return np.bincount(predictions).argmax()  # Return the most frequent class

def predict_from_video(model, video_path, output_dir, frame_rate=30):
    # Check if output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Output directory '{output_dir}' created.")

    # Extract frames from the video
    extract_frames(video_path, output_dir, frame_rate)

    predictions = []
    # Check if extracted frames directory has images
    if not os.listdir(output_dir):
        print(f"No frames found in directory {output_dir}. Make sure frames were extracted.")
        return

    for frame_filename in os.listdir(output_dir):
        frame_path = os.path.join(output_dir, frame_filename)

        # Preprocess the frame
        img_array = preprocess_frame(frame_path)

        # Make prediction
        prediction = model.predict(img_array)
        predicted_class = np.argmax(prediction, axis=-1)  # Get the class with highest probability

        # Store prediction
        predictions.append(predicted_class[0])

        # Convert class index to label
        predicted_label = class_labels[predicted_class[0]]
        print(f"Frame {frame_filename}: Predicted gesture - {predicted_label}")

    # Aggregate predictions over frames (e.g., majority voting)
    aggregated_predictions = aggregate_predictions(predictions)
    aggregated_label = class_labels[aggregated_predictions]
    print(f"Aggregated Prediction: {aggregated_label}")

# Example usage
video_path = r'/content/downloaded_video.mp4'  # Replace with your actual video path
output_dir = 'extracted_frames'

predict_from_video(model, video_path, output_dir, frame_rate=30)


# In[ ]:


import cv2
import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Load the trained model
model = load_model('model.h5')


class_labels = {0: 'hello', 1: 'thanks', 2: 'please', 3: 'goodbye', 4: 'yes', 5: 'no',
                6: 'help', 7: 'more', 8: 'stop', 9: 'come', 10: 'go', 11: 'wait', 12: 'eat',
                13: 'drink', 14: 'sleep', 15: 'love', 16: 'happy', 17: 'sad', 18: 'angry',
                19: 'excuse me', 20: 'finish', 21: 'rest', 22: 'fine', 23: 'sorry'}

def extract_frames(video_path, output_dir, frame_rate=30):

    if not os.path.exists(video_path):
        print(f"Error: Video file '{video_path}' not found.")
        return

    # Open the video file
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error opening video stream or file")
        return

    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Output directory '{output_dir}' created.")

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Save frames at a specified frame rate
        if frame_count % frame_rate == 0:
            frame_filename = os.path.join(output_dir, f"frame_{frame_count}.jpg")
            cv2.imwrite(frame_filename, frame)  # Save the frame as an image
        frame_count += 1

    cap.release()
    print(f"Frames saved to {output_dir}")

def preprocess_frame(frame_path, target_size=(64, 64)):
    img = image.load_img(frame_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array /= 255.0  # Rescale the image
    return img_array


def predict_from_video(model, video_path, output_dir, frame_rate=30):
    # Extract frames from the video
    extract_frames(video_path, output_dir, frame_rate)

    predictions = []
    for frame_filename in os.listdir(output_dir):
        frame_path = os.path.join(output_dir, frame_filename)

        # Preprocess the frame
        img_array = preprocess_frame(frame_path)

        # Make prediction
        prediction = model.predict(img_array)

        print(f"Raw Prediction: {prediction}")

        predicted_class = np.argmax(prediction, axis=-1)

        print(f"Predicted Class Index: {predicted_class}")


        predictions.append(predicted_class[0])

        predicted_word = class_labels.get(predicted_class[0], "Unknown")
        print(f"Frame {frame_filename}: Predicted gesture - {predicted_word}")

    aggregated_prediction = aggregate_predictions(predictions)
    final_predicted_word = class_labels.get(aggregated_prediction, "Unknown")
    print(f"Final Predicted Gesture: {final_predicted_word}")


def aggregate_predictions(predictions):
    return max(set(predictions), key=predictions.count)

video_path = r'/content/downloaded_video.mp4'
output_dir = 'extracted_frames'

predict_from_video(model, video_path, output_dir, frame_rate=30)


# In[ ]:


import cv2
import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Load the trained model
model = load_model('model.h5')

# Mapping of class indices to words
class_labels = {0: 'hello', 1: 'thanks', 2: 'please', 3: 'goodbye', 4: 'yes', 5: 'no',
                6: 'help', 7: 'more', 8: 'stop', 9: 'come', 10: 'go', 11: 'wait', 12: 'eat',
                13: 'drink', 14: 'sleep', 15: 'love', 16: 'happy', 17: 'sad', 18: 'angry',
                19: 'excuse me', 20: 'finish', 21: 'bathroom', 22: 'fine', 23: 'sorry'}

def extract_frames(video_path, output_dir, frame_rate=30):
    if not os.path.exists(video_path):
        print(f"Error: Video file '{video_path}' not found.")
        return

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error opening video stream or file")
        return

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Output directory '{output_dir}' created.")

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_rate == 0:
            frame_filename = os.path.join(output_dir, f"frame_{frame_count}.jpg")
            cv2.imwrite(frame_filename, frame)
        frame_count += 1

    cap.release()
    print(f"Frames saved to {output_dir}")

def preprocess_frame(frame_path, target_size=(64, 64)):
    img = image.load_img(frame_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array

def predict_from_video(model, video_path, output_dir, frame_rate=30):
    extract_frames(video_path, output_dir, frame_rate)

    predictions = []
    for frame_filename in os.listdir(output_dir):
        frame_path = os.path.join(output_dir, frame_filename)

        img_array = preprocess_frame(frame_path)

        prediction = model.predict(img_array)

        print(f"Raw Prediction for {frame_filename}: {prediction}")

        predicted_class = np.argmax(prediction, axis=-1)

        print(f"Predicted Class Index: {predicted_class}")

        predictions.append(predicted_class[0])

        predicted_word = class_labels.get(predicted_class[0], "Unknown")
        print(f"Frame {frame_filename}: Predicted gesture - {predicted_word}")

    aggregated_prediction = aggregate_predictions(predictions)
    final_predicted_word = class_labels.get(aggregated_prediction, "Unknown")
    print(f"Final Predicted Gesture: {final_predicted_word}")

def aggregate_predictions(predictions):
    weighted_predictions = []
    for i, prediction in enumerate(predictions):
        weight = len(predictions) - i  # More recent frames get higher weight
        weighted_predictions.extend([prediction] * weight)

    return max(set(weighted_predictions), key=weighted_predictions.count)

# Example usage
video_path = r'/content/downloaded_video.mp4'
output_dir = 'extracted_frames'

predict_from_video(model, video_path, output_dir, frame_rate=30)


# In[ ]:


import cv2
import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Load the trained model
model = load_model('model.h5')

# Mapping of class indices to words
class_labels = {0: 'hello', 1: 'thanks', 2: 'please', 3: 'goodbye', 4: 'yes', 5: 'no',
                6: 'help', 7: 'more', 8: 'stop', 9: 'come', 10: 'go', 11: 'wait', 12: 'eat',
                13: 'drink', 14: 'sleep', 15: 'love', 16: 'happy', 17: 'sad', 18: 'angry',
                19: 'excuse me', 20: 'finish', 21: 'bathroom', 22: 'fine', 23: 'sorry'}

def extract_frames(video_path, output_dir, frame_rate=30):
    if not os.path.exists(video_path):
        print(f"Error: Video file '{video_path}' not found.")
        return

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error opening video stream or file")
        return

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Output directory '{output_dir}' created.")

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_rate == 0:
            frame_filename = os.path.join(output_dir, f"frame_{frame_count}.jpg")
            cv2.imwrite(frame_filename, frame)
        frame_count += 1

    cap.release()
    print(f"Frames saved to {output_dir}")

def preprocess_frame(frame_path, target_size=(64, 64)):
    img = image.load_img(frame_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array

def update_prediction_label(predicted_word):
    print(f"Predicted Gesture: {predicted_word}")

def display_frame_in_colab(frame_path):
    # Open image using OpenCV
    img = cv2.imread(frame_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    plt.imshow(img_rgb)
    plt.axis('off')  # Hide axes
    plt.show()

def predict_from_video(model, video_path, output_dir, frame_rate=30):
    extract_frames(video_path, output_dir, frame_rate)

    predictions = []
    for frame_filename in os.listdir(output_dir):
        frame_path = os.path.join(output_dir, frame_filename)

        img_array = preprocess_frame(frame_path)

        prediction = model.predict(img_array)

        print(f"Raw Prediction for {frame_filename}: {prediction}")

        predicted_class = np.argmax(prediction, axis=-1)

        predictions.append(predicted_class[0])

        predicted_word = class_labels.get(predicted_class[0], "Unknown")
        print(f"Frame {frame_filename}: Predicted gesture - {predicted_word}")

        # Display frame and prediction in the notebook
        display_frame_in_colab(frame_path)
        update_prediction_label(predicted_word)

    aggregated_prediction = aggregate_predictions(predictions)
    final_predicted_word = class_labels.get(aggregated_prediction, "Unknown")
    update_prediction_label(f"Final Predicted Gesture: {final_predicted_word}")
    print(f"Final Predicted Gesture: {final_predicted_word}")

def aggregate_predictions(predictions):
    weighted_predictions = []
    for i, prediction in enumerate(predictions):
        weight = len(predictions) - i  # More recent frames get higher weight
        weighted_predictions.extend([prediction] * weight)

    return max(set(weighted_predictions), key=weighted_predictions.count)

# Example usage
video_path = r'/content/sign_language_video.mp4'
output_dir = 'extracted_frames'

predict_from_video(model, video_path, output_dir, frame_rate=30)


# In[ ]:


import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import img_to_array
import cv2
from tensorflow.keras.models import load_model

# Define the function to load and preprocess the data (if needed)
def load_and_preprocess_data(img_size=(64, 64)):
    # Add your code here to load and preprocess the dataset
    pass
def create_model(input_shape=(64, 64, 3), num_classes=26):
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Define the function to predict from frame
def predict_from_frame(frame, model, label_map):
    img_resized = cv2.resize(frame, (64, 64))
    img_array = img_to_array(img_resized) / 255.0  # Normalize the image
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # Predict the class (sign language letter)
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction, axis=1)[0]
    predicted_label = label_map[predicted_class]

    return predicted_label

# Load the pre-trained model (assuming you've trained and saved it already)
model = load_model('model.h5')

# Sign language label map
label_map = {
    0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J',
    10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S',
    19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z'
}

# Path to the video
VIDEO_PATH = 'sign_language_video.mp4'  # Path to your video

# Open the video file
cap = cv2.VideoCapture(VIDEO_PATH)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Predict the letter from the current frame
    predicted_label = predict_from_frame(frame, model, label_map)

    # Display the prediction on the frame
    cv2.putText(frame, predicted_label, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2, cv2.LINE_AA)

    # Show the frame with prediction
    cv2.imshow('Sign Language Prediction', frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


# In[ ]:


import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import img_to_array, load_img

# Path to your dataset (Make sure to update the path)
DATASET_PATH = r'C:\Users\Harsh\Downloads\asl_alphabet_test\asl_alphabet_test'

def load_and_preprocess_data(img_size=(64, 64)):
    images, labels = [], []
    label_map = {}

    for idx, label in enumerate(os.listdir(DATASET_PATH)):
        label_path = os.path.join(DATASET_PATH, label)
        if os.path.isdir(label_path):
            label_map[idx] = label
            for image_file in os.listdir(label_path):
                img_path = os.path.join(label_path, image_file)
                img = load_img(img_path, target_size=img_size)
                img_array = img_to_array(img)
                images.append(img_array)
                labels.append(idx)

    # Normalize image data
    images = np.array(images) / 255.0
    labels = np.array(labels)

    # Split data into train and test
    X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test, label_map

def create_model(input_shape=(64, 64, 3), num_classes=26):
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def train_and_save_model():
    X_train, X_test, y_train, y_test, label_map = load_and_preprocess_data()
    model = create_model(num_classes=len(label_map))
    model.summary()

    # Train the model
    model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

    # Save the model
    model.save('model.h5')
    print("Model saved as model.h5")

train_and_save_model()


# In[ ]:


import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model

# Load the pre-trained model
model = load_model('model.h5')

# Path to video
VIDEO_PATH = 'path_to_your_video.mp4'
label_map = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J',
             10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T',
             20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z'}

def predict_from_frame(frame, model, label_map):

    img_resized = cv2.resize(frame, (64, 64))
    img_array = img_to_array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction, axis=1)[0]
    predicted_label = label_map[predicted_class]

    return predicted_label

# Open the video file
cap = cv2.VideoCapture(VIDEO_PATH)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    predicted_label = predict_from_frame(frame, model, label_map)
    cv2.putText(frame, predicted_label, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.imshow('Sign Language Prediction', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


# In[ ]:


import os

DATASET_PATH = r'C:\Users\Harsh\Downloads\asl_alphabet_test\asl_alphabet_test'

if os.path.exists(DATASET_PATH):
    print(f"Directory found: {DATASET_PATH}")
else:
    print(f"Directory not found: {DATASET_PATH}")



# In[ ]:


pip install kaggle


# In[ ]:


DATASET_PARENT_PATH = r'D:\asl_alphabet_test'

if os.path.exists(DATASET_PARENT_PATH):
    print("Contents of the parent directory:")
    print(os.listdir(DATASET_PARENT_PATH))  # List the files and folders inside 'archive (10)'
else:
    print(f"Parent path does not exist: {DATASET_PARENT_PATH}")



# In[ ]:


import os
drives = [drive for drive in os.popen('wmic logicaldisk get caption').read().splitlines() if len(drive) > 0]
print("Available drives:")
for drive in drives:
    print(drive)


# In[ ]:


import os

# This will attempt to list available drives
drives = [drive for drive in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ' if os.path.exists(f'{drive}:')]
print(f"Available drives: {drives}")


# In[ ]:


import os
import string

def check_drives():
    available_drives = []
    for letter in string.ascii_uppercase:
        drive = f'{letter}:\\'
        if os.path.exists(drive):
            available_drives.append(drive)
    return available_drives

drives = check_drives()
print("Available drives:", drives)

