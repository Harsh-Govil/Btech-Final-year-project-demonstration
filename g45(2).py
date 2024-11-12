#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from google.colab import files
uploaded = files.upload()


# In[ ]:


import os
print("Current Working Directory:", os.getcwd())


# In[ ]:


from google.colab import drive
drive.mount('/content/drive')


# In[ ]:


import os
print("Files in current directory:", os.listdir('/content'))


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
    if not os.path.exists(video_path):
        print(f"Error: Video file does not exist at {video_path}")
        return

    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Video frame rate: {fps} frames per second.")

    frame_count = 0
    frame_num = 0

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        if frame_count % int(fps // frame_rate) == 0:
            frame_filename = os.path.join(output_folder, f'frame_{frame_num:04d}.jpg')
            cv2.imwrite(frame_filename, frame)
            print(f"Saved: {frame_filename}")
            frame_num += 1

        frame_count += 1

    cap.release()
    print("Frame extraction completed.")

video_path = "/content/videoBTP.mp4"


extract_frames(video_path, output_folder='frames', frame_rate=1)


# In[ ]:


import pandas as pd
import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array


frames_dir = '/content/frames'

image_files = [f for f in os.listdir(frames_dir) if f.endswith('.jpg')]

df = pd.DataFrame(image_files, columns=['filename'])


datagen = ImageDataGenerator(rescale=1./255)

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

train_gen = image_generator(df)
images = next(train_gen)
print(images.shape)


# In[ ]:


from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing import image
import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input


model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
images_preprocessed = preprocess_input(images)
features = model.predict(images_preprocessed)
print(features.shape)


# In[ ]:


from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import MobileNetV2

base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.5)(x)
x = Dense(256, activation='relu')(x)
predictions = Dense(26, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)


for layer in base_model.layers:
    layer.trainable = False
model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()


# In[ ]:


from google.colab import files
import zipfile
import os

# Upload the dataset zip file
uploaded = files.upload()


# In[ ]:


import zipfile
import os

zip_file_path = '/content/asl_alphabet_test.zip'
with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    zip_ref.extractall('/content/asl_alphabet_test')

extracted_files = os.listdir('/content/asl_alphabet_test')
print(extracted_files)


# In[ ]:


import os

extracted_files = os.listdir('/content/asl_alphabet_test/asl_alphabet_test')
print(extracted_files)


# In[ ]:


import os

# Check the directory structure of the 'asl_alphabet_test' folder
extracted_files = os.listdir('/content/asl_alphabet_test')
print(extracted_files)


# In[ ]:


import os
train_val_dir = '/content/asl_alphabet_test/asl_alphabet_test'
print(os.listdir(train_val_dir))


# In[ ]:


import os
import shutil
import random

# Define the source and destination directories
source_dir = '/content/asl_alphabet_test/asl_alphabet_test'
train_dir = '/content/asl_alphabet_test/train'
val_dir = '/content/asl_alphabet_test/val'

# Create train and validation directories
if not os.path.exists(train_dir):
    os.makedirs(train_dir)
if not os.path.exists(val_dir):
    os.makedirs(val_dir)

# List of class labels (A-Z and 'nothing')
labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'space', 'nothing']

# Create subdirectories for each class in train and validation directories
for label in labels:
    os.makedirs(os.path.join(train_dir, label), exist_ok=True)
    os.makedirs(os.path.join(val_dir, label), exist_ok=True)

# Function to organize images
def organize_images():
    images = os.listdir(source_dir)
    for image in images:
        if not image.endswith('.jpg'):
            continue
        label = image.split('_')[0]  # Extract the label (e.g., 'A', 'B', etc.)
        if label in labels:
            if random.random() < 0.8:
                dest_dir = os.path.join(train_dir, label)
            else:
                dest_dir = os.path.join(val_dir, label)

            # Move the image to the appropriate folder
            shutil.move(os.path.join(source_dir, image), os.path.join(dest_dir, image))

# Organize the images
organize_images()

# Verify that the images are correctly organized
print("Training data directories:", os.listdir(train_dir))
print("Validation data directories:", os.listdir(val_dir))


# In[ ]:


import os
train_dir = '/content/frames'
if os.path.isdir(train_dir):
    print("Directory found!")
    for subdir, dirs, files in os.walk(train_dir):
        print(subdir, len(files))
else:
    print("Directory not found.")


# In[ ]:


import os
import shutil


# In[ ]:


import os
import shutil

source_dir = '/content/frames'
output_dir = '/content/frames_sorted'

# List of letters (A to Z)
letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

# Create subdirectories for each letter (A to Z) in the output directory
for letter in letters:
    os.makedirs(os.path.join(output_dir, letter), exist_ok=True)

# List all files in the source directory
files = os.listdir(source_dir)

for file in files:
    # Only process .jpg or .png files
    if file.endswith('.jpg') or file.endswith('.png'):
        # Extract the letter from the filename (assumes the first character of the filename is the letter)
        letter = file[0].upper()  # Convert the first character to uppercase to match the folder names
        if letter in letters:
            # Move the file to the appropriate subdirectory
            shutil.move(os.path.join(source_dir, file), os.path.join(output_dir, letter, file))

print("Files have been reorganized into subdirectories based on their labels.")


# In[ ]:


import os
import shutil
import random

# Source directory where all the images are sorted into A-Z subfolders
source_dir = '/content/frames_sorted'

# Create train and validation directories inside frames_sorted
train_dir = os.path.join(source_dir, 'train')
val_dir = os.path.join(source_dir, 'val')

# Create the directories if they do not exist
os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)

# Create subdirectories for each class inside train and val directories
letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

for letter in letters:
    os.makedirs(os.path.join(train_dir, letter), exist_ok=True)
    os.makedirs(os.path.join(val_dir, letter), exist_ok=True)

# Now, let's split the files from the source directory into train and val directories
for letter in letters:
    # Get all the image files for the current letter class
    files = os.listdir(os.path.join(source_dir, letter))
    files = [f for f in files if f.endswith('.jpg') or f.endswith('.png')]  # Filter for image files

    # Shuffle the files randomly for splitting
    random.shuffle(files)

    # Split into 80% train and 20% validation (you can adjust the ratio as needed)
    split_index = int(0.8 * len(files))  # 80% for training

    train_files = files[:split_index]
    val_files = files[split_index:]

    # Move the train files
    for file in train_files:
        shutil.move(os.path.join(source_dir, letter, file), os.path.join(train_dir, letter, file))

    # Move the validation files
    for file in val_files:
        shutil.move(os.path.join(source_dir, letter, file), os.path.join(val_dir, letter, file))

print("Files have been split into training and validation directories.")


# In[ ]:


import os
import shutil
import random


source_dir = '/content/frames'
train_dir = '/content/frames_sorted/train'
val_dir = '/content/frames_sorted/val'

letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
for letter in letters:
    os.makedirs(os.path.join(train_dir, letter), exist_ok=True)
    os.makedirs(os.path.join(val_dir, letter), exist_ok=True)

# Get all the files in the source directory and organize them by class
files = os.listdir(source_dir)
for file in files:
    if file.endswith('.jpg') or file.endswith('.png'):
        class_name = file[0].upper()
        if class_name in letters:
            if random.random() < 0.8:
                shutil.move(os.path.join(source_dir, file), os.path.join(train_dir, class_name, file))
            else:
                shutil.move(os.path.join(source_dir, file), os.path.join(val_dir, class_name, file))

print("Images have been assigned to train and val directories.")


# In[ ]:


from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import os

# Set the image size and number of classes
img_size = (224, 224)
train_dir = '/content/frames_sorted/train'
val_dir = '/content/frames_sorted/val'


train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=30, width_shift_range=0.2,
                                   height_shift_range=0.2, shear_range=0.2, zoom_range=0.2,
                                   horizontal_flip=True, fill_mode='nearest')
val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=img_size,
    batch_size=32,
    class_mode='categorical'
)

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=img_size,
    batch_size=32,
    class_mode='categorical'
)

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(train_generator.num_classes, activation='softmax')


model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])


history = model.fit(
    train_generator,
    epochs=10,
    validation_data=val_generator
)


test_loss, test_acc = model.evaluate(val_generator)
print(f"Test Accuracy: {test_acc}")
print(f"Test Loss: {test_loss}")


# In[ ]:


import cv2
import os
import numpy as np
from tensorflow.keras.models import load_model

frame_dir = '/content/frames_sorted'

labels = sorted(os.listdir(frame_dir))

model = load_model('/content/sign_language_model.h5')
print(model.summary())
img_size = (224, 224)

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam")
    exit()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame")
        break

    # Resize frame to the expected input size for the model
    img_resized = cv2.resize(frame, img_size)

    # Preprocess the image if required by the model (e.g., normalization)
    img_normalized = img_resized / 255.0
    img_input = np.expand_dims(img_normalized, axis=0)

    # Predict the label
    prediction = model.predict(img_input)
    predicted_class_index = np.argmax(prediction, axis=1)
    predicted_class_label = labels[predicted_class_index[0]]

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, predicted_class_label, (10, 30), font, 1, (255, 0, 0), 2, cv2.LINE_AA)

    cv2.imshow('Frame with Label', frame)

    label_dir = os.path.join(frame_dir, predicted_class_label)
    if not os.path.exists(label_dir):
        os.makedirs(label_dir)

    frame_filename = f"{predicted_class_label}_{str(int(np.random.random() * 10000))}.jpg"
    cv2.imwrite(os.path.join(label_dir, frame_filename), frame)

    print(f"Captured Frame: {frame_filename} with label: {predicted_class_label}")
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


# In[ ]:


import cv2
import os
import numpy as np
from tensorflow.keras.models import load_model

# Set up the directories where frames are stored (you can change this path as needed)
frame_dir = '/content/frames_sorted'

labels = sorted(os.listdir(frame_dir))

# Load your pre-trained model
model = load_model('/content/sign_language_model.h5')
print(model.summary())

img_size = (224, 224)

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam")
    exit()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame")
        break

    img_resized = cv2.resize(frame, img_size)

    img_normalized = img_resized / 255.0
    img_input = np.expand_dims(img_normalized, axis=0)
    prediction = model.predict(img_input)
    print("Prediction:", prediction)
    predicted_class_index = np.argmax(prediction, axis=1)
    print("Predicted Class Index:", predicted_class_index)
    predicted_class_label = labels[predicted_class_index[0]]
    print("Predicted Class Label:", predicted_class_label)

    # Display the frame with the label (A-Z)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, predicted_class_label, (10, 30), font, 1, (255, 0, 0), 2, cv2.LINE_AA)

    # Show the frame with the label
    cv2.imshow('Frame with Label', frame)
    cv2.waitKey(1)

    label_dir = os.path.join(frame_dir, predicted_class_label)
    if not os.path.exists(label_dir):
        os.makedirs(label_dir)

    frame_filename = f"{predicted_class_label}_{str(int(np.random.random() * 10000))}.jpg"
    print(f"Saving frame to {os.path.join(label_dir, frame_filename)}")
    cv2.imwrite(os.path.join(label_dir, frame_filename), frame)

    print(f"Captured Frame: {frame_filename} with label: {predicted_class_label}")

    # Press 'q' to quit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


# In[ ]:


import cv2
import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image


model = load_model('/content/sign_language_model.h5')

class_labels = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F',
                6: 'G', 7: 'H', 8: 'I', 9: 'J', 10: 'K', 11: 'L',
                12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R',
                18: 'S', 19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X',
                24: 'Y', 25: 'Z'}

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


def preprocess_frame(frame_path, target_size=(224, 224)):
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

        print(f"Raw Prediction: {prediction}")


        predicted_class = np.argmax(prediction, axis=-1)


        print(f"Predicted Class Index: {predicted_class}")


        predictions.append(predicted_class[0])

        predicted_gesture = class_labels.get(predicted_class[0], "Unknown")
        print(f"Frame {frame_filename}: Predicted gesture - {predicted_gesture}")


    aggregated_prediction = aggregate_predictions(predictions)
    final_predicted_gesture = class_labels.get(aggregated_prediction, "Unknown")
    print(f"Final Predicted Gesture: {final_predicted_gesture}")


def aggregate_predictions(predictions):
    return max(set(predictions), key=predictions.count)


video_path = r'/content/videoBTP.mp4'
output_dir = 'extracted_frames'

# Run the prediction process
predict_from_video(model, video_path, output_dir, frame_rate=30)


# In[ ]:


import cv2
import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image


model = load_model('/content/sign_language_model.h5')

class_labels = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F',
                6: 'G', 7: 'H', 8: 'I', 9: 'J', 10: 'K', 11: 'L',
                12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R',
                18: 'S', 19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X',
                24: 'Y', 25: 'Z'}

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

def preprocess_frame(frame_path, target_size=(224, 224)):
    img = image.load_img(frame_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Rescale the image
    return img_array

def predict_from_video(model, video_path, output_dir, frame_rate=30):
    extract_frames(video_path, output_dir, frame_rate)

    predictions = []
    for frame_filename in os.listdir(output_dir):
        frame_path = os.path.join(output_dir, frame_filename)

        img_array = preprocess_frame(frame_path)
        prediction = model.predict(img_array)
        predicted_class = np.argmax(prediction, axis=-1)
        predictions.append(predicted_class[0])

        predicted_gesture = class_labels.get(predicted_class[0], "Unknown")
        print(f"Frame {frame_filename}: Predicted gesture - {predicted_gesture}")

        frame = cv2.imread(frame_path)
        font = cv2.FONT_HERSHEY_SIMPLEX
        text = f"Predicted: {predicted_gesture}"
        position = (10, 30)
        color = (0, 255, 0)
        font_scale = 1
        thickness = 2
        cv2.putText(frame, text, position, font, font_scale, color, thickness)

        output_frame_path = os.path.join(output_dir, f"predicted_{frame_filename}")
        cv2.imwrite(output_frame_path, frame)

    aggregated_prediction = aggregate_predictions(predictions)
    final_predicted_gesture = class_labels.get(aggregated_prediction, "Unknown")
    print(f"Final Predicted Gesture: {final_predicted_gesture}")

def aggregate_predictions(predictions):
    return max(set(predictions), key=predictions.count)
video_path = r'/content/videoBTP.mp4'
output_dir = 'extracted_frames_with_predictions'

predict_from_video(model, video_path, output_dir, frame_rate=30)


# In[ ]:


import cv2
import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Load the trained model
model = load_model('/content/sign_language_model.h5')

class_labels = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F',
                6: 'G', 7: 'H', 8: 'I', 9: 'J', 10: 'K', 11: 'L',
                12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R',
                18: 'S', 19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X',
                24: 'Y', 25: 'Z'}


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

        # Save frames at a specified frame rate
        if frame_count % frame_rate == 0:
            frame_filename = os.path.join(output_dir, f"frame_{frame_count}.jpg")
            cv2.imwrite(frame_filename, frame)
        frame_count += 1

    cap.release()
    print(f"Frames saved to {output_dir}")

def preprocess_frame(frame_path, target_size=(224, 224)):
    img = image.load_img(frame_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array

def predict_from_video(model, video_path, output_dir, frame_rate=30):
    # Extract frames from the video
    extract_frames(video_path, output_dir, frame_rate)

    predictions = []
    for frame_filename in os.listdir(output_dir):
        frame_path = os.path.join(output_dir, frame_filename)
        img_array = preprocess_frame(frame_path)
        prediction = model.predict(img_array)
        predicted_class = np.argmax(prediction, axis=-1)
        predictions.append(predicted_class[0])

        predicted_gesture = class_labels.get(predicted_class[0], "Unknown")
        print(f"Frame {frame_filename}: Predicted gesture - {predicted_gesture}")

        frame = cv2.imread(frame_path)
        font = cv2.FONT_HERSHEY_SIMPLEX
        text = f"Predicted: {predicted_gesture}"
        position = (10, 30)
        color = (0, 255, 0)
        font_scale = 1
        thickness = 2

        cv2.putText(frame, text, position, font, font_scale, color, thickness)

        output_frame_path = os.path.join(output_dir, f"predicted_{frame_filename}")
        cv2.imwrite(output_frame_path, frame)

    aggregated_prediction = aggregate_predictions(predictions)
    final_predicted_gesture = class_labels.get(aggregated_prediction, "Unknown")
    print(f"Final Predicted Gesture: {final_predicted_gesture}")

def aggregate_predictions(predictions):
    return max(set(predictions), key=predictions.count)

video_path = r'/content/videoBTP.mp4'
output_dir = 'extracted_frames_with_predictions'

predict_from_video(model, video_path, output_dir, frame_rate=30)


# In[ ]:


from google.colab.patches import cv2_imshow

def predict_from_video(model, video_path, output_dir, frame_rate=30):
    extract_frames(video_path, output_dir, frame_rate)

    predictions = []
    for frame_filename in os.listdir(output_dir):
        frame_path = os.path.join(output_dir, frame_filename)

        img_array = preprocess_frame(frame_path)

        prediction = model.predict(img_array)

        predicted_class = np.argmax(prediction, axis=-1)

        predictions.append(predicted_class[0])

        predicted_gesture = class_labels.get(predicted_class[0], "Unknown")
        print(f"Frame {frame_filename}: Predicted gesture - {predicted_gesture}")

        frame = cv2.imread(frame_path)

        font = cv2.FONT_HERSHEY_SIMPLEX
        text = f"Predicted: {predicted_gesture}"
        position = (10, 30)
        color = (0, 255, 0)
        font_scale = 1
        thickness = 2
        cv2.putText(frame, text, position, font, font_scale, color, thickness)
        cv2_imshow(frame)

        cv2.waitKey(0)

    aggregated_prediction = aggregate_predictions(predictions)
    final_predicted_gesture = class_labels.get(aggregated_prediction, "Unknown")
    print(f"Final Predicted Gesture: {final_predicted_gesture}")


video_path = r'/content/videoBTP.mp4'
output_dir = 'extracted_frames_with_predictions'

predict_from_video(model, video_path, output_dir, frame_rate=30)


# In[ ]:


import cv2
import os
import numpy as np
from tensorflow.keras.models import load_model

frame_dir = '/content/frames_sorted'

labels = sorted(os.listdir(frame_dir))

model = load_model('/content/sign_language_model.h5')
print(model.summary())

img_size = (224, 224)

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam")
    exit()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame")
        break

    img_resized = cv2.resize(frame, img_size)

    img_normalized = img_resized / 255.0
    img_input = np.expand_dims(img_normalized, axis=0)


    prediction = model.predict(img_input)
    print("Prediction:", prediction)  # Check model output
    predicted_class_index = np.argmax(prediction, axis=1)
    print("Predicted Class Index:", predicted_class_index)
    predicted_class_label = labels[predicted_class_index[0]]
    print("Predicted Class Label:", predicted_class_label)

    # Display the frame with the label (A-Z)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, predicted_class_label, (10, 30), font, 1, (255, 0, 0), 2, cv2.LINE_AA)

    # Show the frame with the label
    cv2.imshow('Frame with Label', frame)
    cv2.waitKey(1)

    label_dir = os.path.join(frame_dir, predicted_class_label)
    if not os.path.exists(label_dir):
        os.makedirs(label_dir)

    frame_filename = f"{predicted_class_label}_{str(int(np.random.random() * 10000))}.jpg"
    print(f"Saving frame to {os.path.join(label_dir, frame_filename)}")
    cv2.imwrite(os.path.join(label_dir, frame_filename), frame)

    # Print the output in the console for every frame
    print(f"Captured Frame: {frame_filename} with label: {predicted_class_label}")

    # Press 'q' to quit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


# In[ ]:





# In[ ]:


test_loss, test_acc = model.evaluate(val_generator)
print(f'Test Loss: {test_loss}, Test Accuracy: {test_acc}')


# In[ ]:


# Check the contents of the top-level directory
top_level_dir = '/content/asl_alphabet_test'
print("Top-level directory contents:", os.listdir(top_level_dir))



# In[ ]:


# Check contents of the 'train' and 'val' directories
train_dir = '/content/asl_alphabet_test/train'
val_dir = '/content/asl_alphabet_test/val'

print("Training directory contents:", os.listdir(train_dir))
print("Validation directory contents:", os.listdir(val_dir))


# In[ ]:


import zipfile
import os

# Path to the uploaded ZIP file
zip_file = '/content/asl_alphabet_test.zip'

# Directory where you want to extract the files
extract_dir = '/content/asl_alphabet_test/'

# Extract the ZIP file
with zipfile.ZipFile(zip_file, 'r') as zip_ref:
    zip_ref.extractall(extract_dir)

# Check the contents of the extracted directory
extracted_dir = os.path.join(extract_dir, 'asl_alphabet_test')
print("Extracted directory contents:", os.listdir(extracted_dir))


# In[ ]:


import shutil

# Directory where the images were extracted
extracted_dir = '/content/asl_alphabet_test/asl_alphabet_test/'

# Define the list of class names (based on the filenames)
class_names = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'nothing', 'space']

# Create subdirectories for each class under train and validation directories
train_dir = '/content/asl_alphabet_test/asl_alphabet_test/train/'
val_dir = '/content/asl_alphabet_test/asl_alphabet_test/val/'

# Create subdirectories for training and validation data
for class_name in class_names:
    os.makedirs(os.path.join(train_dir, class_name), exist_ok=True)
    os.makedirs(os.path.join(val_dir, class_name), exist_ok=True)

# Move images to the appropriate subdirectories (train or validation)
for file_name in os.listdir(extracted_dir):
    if file_name.endswith('.jpg'):
        # Extract the class name from the file name (assuming the class is the first letter)
        class_name = file_name.split('_')[0]
        target_dir = train_dir if 'test' in file_name else val_dir
        shutil.move(os.path.join(extracted_dir, file_name), os.path.join(target_dir, class_name, file_name))

# Check the directory structure after organizing the images
print("Training data organized:", os.listdir(train_dir))
print("Validation data organized:", os.listdir(val_dir))


# In[ ]:


import os
import shutil

# Paths for validation images and organized class folders
val_images_dir = '/content/asl_alphabet_test/asl_alphabet_test/val'
classes = ['U', 'X', 'T', 'S', 'K', 'J', 'R', 'E', 'B', 'nothing', 'Z', 'F', 'H', 'Y', 'L', 'M', 'D', 'G', 'I', 'P', 'C', 'Q', 'N', 'W', 'space', 'V', 'A', 'O']

# Manually move validation images into class directories
for image_name in os.listdir(val_images_dir):
    if image_name.endswith('_test.jpg'):  # Only select the validation images
        # Get the class label from the image name (e.g., 'A_test.jpg' -> 'A')
        class_label = image_name.split('_')[0]

        # Ensure the class subdirectory exists
        class_dir = os.path.join(val_images_dir, class_label)
        if not os.path.exists(class_dir):
            os.makedirs(class_dir)

        # Move the image into the correct class folder
        source_path = os.path.join(val_images_dir, image_name)
        destination_path = os.path.join(class_dir, image_name)
        shutil.move(source_path, destination_path)

print("Validation data has been organized.")


# In[ ]:


import os

# Check the contents of the root directory where the extracted images might be
root_dir = '/content/asl_alphabet_test/asl_alphabet_test'
print("Root directory contents:", os.listdir(root_dir))


# In[ ]:


import os

# List contents of 'train' and 'val' directories
train_dir = '/content/asl_alphabet_test/asl_alphabet_test/train'
val_dir = '/content/asl_alphabet_test/asl_alphabet_test/val'

print("Training directory contents:", os.listdir(train_dir))
print("Validation directory contents:", os.listdir(val_dir))


# In[ ]:


# Check contents of a few class subdirectories (e.g., 'U', 'X', 'T', etc.)
class_dirs = ['U', 'X', 'T', 'S', 'K', 'J', 'R', 'E', 'B', 'nothing', 'Z', 'F', 'H', 'Y', 'L', 'M', 'D', 'G', 'I', 'P', 'C', 'Q', 'N', 'W', 'space', 'V', 'A', 'O']

for class_name in class_dirs:
    print(f"Contents of {class_name}:", os.listdir(os.path.join(train_dir, class_name)))
    print(f"Contents of {class_name} in validation:", os.listdir(os.path.join(val_dir, class_name)))


# In[ ]:


import os
import random
import shutil

# Set directories
train_dir = '/content/asl_alphabet_test/asl_alphabet_test/train'
val_dir = '/content/asl_alphabet_test/asl_alphabet_test/val'


def move_images_to_val():
    class_dirs = ['U', 'X', 'T', 'S', 'K', 'J', 'R', 'E', 'B', 'nothing', 'Z', 'F', 'H', 'Y', 'L', 'M', 'D', 'G', 'I', 'P', 'C', 'Q', 'N', 'W', 'space', 'V', 'A', 'O']

    for class_name in class_dirs:
        class_train_dir = os.path.join(train_dir, class_name)
        images = [f for f in os.listdir(class_train_dir) if f.endswith('.jpg')]
        if images:

            num_images_to_move = int(len(images) * 0.2)
            images_to_move = random.sample(images, num_images_to_move)

            for image in images_to_move:

                source_path = os.path.join(class_train_dir, image)
                dest_path = os.path.join(val_dir, class_name, image)


                os.makedirs(os.path.join(val_dir, class_name), exist_ok=True)


                shutil.move(source_path, dest_path)
                print(f"Moved {image} from {class_name} to validation.")

move_images_to_val()


# In[ ]:


import os
import shutil

train_dir = '/content/asl_alphabet_test/asl_alphabet_test/train'
val_dir = '/content/asl_alphabet_test/asl_alphabet_test/val'

for class_name in os.listdir(train_dir):
    class_train_dir = os.path.join(train_dir, class_name)
    class_val_dir = os.path.join(val_dir, class_name)

    if os.path.isdir(class_train_dir):
        images = [f for f in os.listdir(class_train_dir) if f.endswith('.jpg')]  # Assuming .jpg images

        if not os.path.exists(class_val_dir):
            os.makedirs(class_val_dir)

        for image in images:
            src = os.path.join(class_train_dir, image)
            dst = os.path.join(class_val_dir, image)
            shutil.move(src, dst)

print("Images moved to validation set successfully.")


# In[ ]:


from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)


train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical')

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical')


# In[ ]:


import os
import shutil

# Set the source and destination directories
train_dir = '/content/asl_alphabet_test/asl_alphabet_test/train'
val_dir = '/content/asl_alphabet_test/asl_alphabet_test/val'

# Loop through each class in the training set
for class_name in os.listdir(train_dir):
    class_train_dir = os.path.join(train_dir, class_name)
    class_val_dir = os.path.join(val_dir, class_name)

    # Check if it's a directory (for each class)
    if os.path.isdir(class_train_dir):
        images = [f for f in os.listdir(class_train_dir) if f.endswith('.jpg')]

        if not os.path.exists(class_val_dir):
            os.makedirs(class_val_dir)

        for image in images:
            src = os.path.join(class_train_dir, image)
            dst = os.path.join(class_val_dir, image)
            shutil.move(src, dst)

print("Images moved to validation set successfully.")


# In[ ]:


def verify_data(directory):
    for class_name in os.listdir(directory):
        class_dir = os.path.join(directory, class_name)
        if os.path.isdir(class_dir):
            print(f"{class_name}: {len(os.listdir(class_dir))} images")

print("Training Data:")
verify_data(train_dir)

print("\nValidation Data:")
verify_data(val_dir)


# In[ ]:


# Replace 'your_train_directory_path' with the actual path to your train directory
train_dir = '/content/asl_alphabet_train/train'

for class_name in os.listdir(train_dir):
    class_dir = os.path.join(train_dir, class_name)
    print(f'Contents of {class_name}: {os.listdir(class_dir)}')


# In[ ]:


import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
import os

# Path to dataset
train_dir = 'D:/asl_alphabet_test/asl_alphabet_test/train'
val_dir = 'D:/asl_alphabet_test/asl_alphabet_test/validation'

# Define image size and batch size
img_size = (224, 224)
batch_size = 32

# Use ImageDataGenerator to load and augment the dataset
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(rescale=1./255)  # Only rescaling for validation

# Load training and validation data
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical'  # Multi-class classification
)

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical'
)

# Define the model architecture
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(1024, activation='relu'),
    layers.Dense(len(train_generator.class_indices), activation='softmax')  # Output layer with the number of classes
])

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    epochs=10,  # You can increase this for better accuracy
    validation_data=val_generator,
    validation_steps=val_generator.samples // batch_size
)

# Save the trained model
model.save('sign_language_model.h5')


# In[ ]:


import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
import numpy as np
import os
from sklearn.model_selection import train_test_split
import cv2  # OpenCV for image processing

# Set the path where your images are stored
image_dir = '/path/to/your/images'  # Update this path to where your images are stored

# Initialize lists to hold images and labels
images = []
labels = []

# Assuming you have folders for each sign language gesture in the image directory
# Each folder name corresponds to the label for that gesture

for label in os.listdir(image_dir):
    gesture_folder = os.path.join(image_dir, label)
    if os.path.isdir(gesture_folder):
        for img_name in os.listdir(gesture_folder):
            img_path = os.path.join(gesture_folder, img_name)
            img = cv2.imread(img_path)  # Read image
            img = cv2.resize(img, (224, 224))  # Resize to 224x224 (standard size for many CNNs)
            img = img / 255.0  # Normalize image (pixel values between 0 and 1)

            images.append(img)
            labels.append(label)

# Convert images to numpy array
images = np.array(images)

# Convert labels to numerical format
# Assuming labels are strings, let's map them to integers
label_map = {label: idx for idx, label in enumerate(sorted(set(labels)))}
labels = np.array([label_map[label] for label in labels])

# Convert labels to one-hot encoding
labels = to_categorical(labels)

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(images, labels, test_size=0.2, random_state=42)


# In[ ]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, GlobalAveragePooling2D

# Define the model
model = Sequential()


model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))


model.add(GlobalAveragePooling2D())

model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(label_map), activation='softmax'))


model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()


# In[ ]:


# Train the model
history = model.fit(X_train, y_train, epochs=20, validation_data=(X_val, y_val), batch_size=32)


# In[ ]:


img_path = '/path/to/new/image.jpg'
img = cv2.imread(img_path)
img = cv2.resize(img, (224, 224))
img = img / 255.0  # Normalize


img = np.expand_dims(img, axis=0)

predictions = model.predict(img)

predicted_class_idx = np.argmax(predictions, axis=1)
predicted_class_label = list(label_map.keys())[list(label_map.values()).index(predicted_class_idx[0])]

print(f"Predicted sign language gesture: {predicted_class_label}")


# In[ ]:


model.save('sign_language_model.h5')
loaded_model = tf.keras.models.load_model('sign_language_model.h5')

