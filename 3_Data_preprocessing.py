# Data Handling and Processing
import pandas as pd
import numpy as np
import os

# Image Processing
from PIL import Image
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array, array_to_img, save_img

# Data Transformation and Scaling
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical

# Model Training and Evaluation
from sklearn.model_selection import train_test_split


# Function to preprocess the image
def preprocess_image(image_path, target_size):
    try:
        img = load_img(image_path, target_size=target_size)
        img_array = img_to_array(img)
        return img_array
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None

# Function to load images from dataframe
def load_images_from_dataframe(df, target_size):
    images = []
    labels = []
    
    for index, row in df.iterrows():
        image_path = row['Image']
        image = preprocess_image(image_path, target_size)
        if image is not None:
            images.append(image)
            labels.append(row['Label'])
    
    images = np.array(images)
    labels = np.array(labels)
    return images, labels

# Parameters
IMG_SIZE = (128, 128)

# Load images and labels for brain train dataset
brain_train_images, brain_train_labels = load_images_from_dataframe(Agumented_Train_Data_Brain, IMG_SIZE)

# Load images and labels for brain test dataset
brain_test_images, brain_test_labels = load_images_from_dataframe(Test_Data_Brain, IMG_SIZE)

# Load images and labels for bone train dataset
bone_train_images, bone_train_labels = load_images_from_dataframe(Agumented_Train_Data_Bone, IMG_SIZE)

# Load images and labels for bone test dataset
bone_test_images, bone_test_labels = load_images_from_dataframe(Test_Data_Bone, IMG_SIZE)

# Normalize images
brain_train_images = brain_train_images / 255.0
brain_test_images = brain_test_images / 255.0
bone_train_images = bone_train_images / 255.0
bone_test_images = bone_test_images / 255.0

# Label encoding and one-hot encoding
def encode_labels(labels):
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(labels)
    one_hot_encoded = to_categorical(integer_encoded)
    return one_hot_encoded

brain_train_labels_encoded = encode_labels(brain_train_labels)
brain_test_labels_encoded = encode_labels(brain_test_labels)
bone_train_labels_encoded = encode_labels(bone_train_labels)
bone_test_labels_encoded = encode_labels(bone_test_labels)