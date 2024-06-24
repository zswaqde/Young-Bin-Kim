# Data Handling and Processing
import pandas as pd
import numpy as np
import os

# Image Processing
from PIL import Image
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Model Building
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, BatchNormalization, MaxPooling2D, \
                                    Input, concatenate, multiply
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Visualization
import matplotlib.pyplot as plt
import cv2

# TensorFlow and Gradient Tape
import tensorflow as tf



# Load the pre-trained models
model_brain_only = load_model('best_model_brain_only.keras')
model_concat = load_model('best_model_concat.keras')
model_product = load_model('best_model_product.keras')

# Function to preprocess the image
def preprocess_image(image_path, target_size):
    try:
        img = load_img(image_path, target_size=target_size)
        img_array = img_to_array(img)
        return img_array
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None

# Function to get Grad-CAM heatmap
def get_grad_cam(model, img_array_brain, img_array_bone, category_index, layer_name, model_type='combined'):
    if model_type == 'brain_only':
        grad_model = Model(inputs=model.inputs, outputs=[model.get_layer(layer_name).output, model.output])
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_array_brain)
            loss = predictions[:, category_index]
        grads = tape.gradient(loss, conv_outputs)
    else:
        grad_model = Model(inputs=model.inputs, outputs=[model.get_layer(layer_name).output, model.output])
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model([img_array_brain, img_array_bone])
            loss = predictions[:, category_index]
        grads = tape.gradient(loss, conv_outputs)
        
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    pooled_grads = pooled_grads.numpy()
    conv_outputs = conv_outputs.numpy()
    
    for i in range(pooled_grads.shape[-1]):
        conv_outputs[:, :, i] *= pooled_grads[i]
        
    heatmap = np.mean(conv_outputs, axis=-1)
    heatmap = np.maximum(heatmap, 0)
    if np.max(heatmap) != 0:
        heatmap /= np.max(heatmap)
    heatmap = cv2.resize(heatmap, (img_array_brain.shape[2], img_array_brain.shape[1]))
    heatmap = np.uint8(255 * heatmap)
    return heatmap

# Example visualization for each category
categories = ['no hemorrhage', 'epidural', 'intraparenchymal', 'intraventricular', 'subarachnoid', 'subdural']
brain_layer_name = 'conv2d_1'  # Layer name for brain images in brain_only model
combined_brain_layer_name = 'conv2d_1'  # Layer name for brain images in combined models
combined_bone_layer_name = 'conv2d_3'   # Layer name for bone images in combined models

# Use one of the paths from your test set
img_path = Test_Data_Brain.iloc[0]['Image']  # Path to a test brain image
bone_img_path = Test_Data_Bone.iloc[0]['Image']  # Path to a test bone image

# Preprocess the images
brain_img = preprocess_image(img_path, IMG_SIZE)
bone_img = preprocess_image(bone_img_path, IMG_SIZE)

if brain_img is not None and bone_img is not None:
    brain_img_array = np.expand_dims(brain_img, axis=0) / 255.0
    bone_img_array = np.expand_dims(bone_img, axis=0) / 255.0

    plt.figure(figsize=(20, 40))
    for i, category in enumerate(categories):
        # Get Grad-CAM heatmap for brain image for each category (Brain Only Model)
        brain_heatmap_brain_only = get_grad_cam(model_brain_only, brain_img_array, None, category_index=i, layer_name=brain_layer_name, model_type='brain_only')
        
        # Get Grad-CAM heatmap for brain image for each category (Concat Model)
        brain_heatmap_concat = get_grad_cam(model_concat, brain_img_array, bone_img_array, category_index=i, layer_name=combined_brain_layer_name)

        # Get Grad-CAM heatmap for bone image for each category (Concat Model)
        bone_heatmap_concat = get_grad_cam(model_concat, brain_img_array, bone_img_array, category_index=i, layer_name=combined_bone_layer_name)
        
        # Get Grad-CAM heatmap for brain image for each category (Product Model)
        brain_heatmap_product = get_grad_cam(model_product, brain_img_array, bone_img_array, category_index=i, layer_name=combined_brain_layer_name)

        # Get Grad-CAM heatmap for bone image for each category (Product Model)
        bone_heatmap_product = get_grad_cam(model_product, brain_img_array, bone_img_array, category_index=i, layer_name=combined_bone_layer_name)
        
        # Display the Grad-CAM heatmap overlayed on the original image (Brain Only Model)
        plt.subplot(len(categories), 4, 4 * i + 1)
        plt.imshow(brain_img / 255.0)
        plt.imshow(brain_heatmap_brain_only, cmap='jet', alpha=0.5)
        plt.title(f'Grad-CAM Brain Only: Brain Image - {category}')

        # Display the Grad-CAM heatmap overlayed on the original image (Concat Model)
        plt.subplot(len(categories), 4, 4 * i + 2)
        plt.imshow(brain_img / 255.0)
        plt.imshow(brain_heatmap_concat, cmap='jet', alpha=0.5)
        plt.title(f'Grad-CAM Concat: Brain Image - {category}')

        plt.subplot(len(categories), 4, 4 * i + 3)
        plt.imshow(bone_img / 255.0)
        plt.imshow(bone_heatmap_concat, cmap='jet', alpha=0.5)
        plt.title(f'Grad-CAM Concat: Bone Image - {category}')
        
        # Display the Grad-CAM heatmap overlayed on the original image (Product Model)
        plt.subplot(len(categories), 4, 4 * i + 4)
        plt.imshow(brain_img / 255.0)
        plt.imshow(brain_heatmap_product, cmap='jet', alpha=0.5)
        plt.title(f'Grad-CAM Product: Brain Image - {category}')

        plt.subplot(len(categories), 4, 4 * i + 5)
        plt.imshow(bone_img / 255.0)
        plt.imshow(bone_heatmap_product, cmap='jet', alpha=0.5)
        plt.title(f'Grad-CAM Product: Bone Image - {category}')

    plt.tight_layout()
    plt.show()
else:
    print("Error: One or both images could not be loaded. Please check the image paths.")