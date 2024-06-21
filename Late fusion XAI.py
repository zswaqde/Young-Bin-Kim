import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
import matplotlib.pyplot as plt
import cv2
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Load the pre-trained model
model = load_model('intracranial_hemorrhage_model.h5')

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
def get_grad_cam(model, img_array_brain, img_array_bone, category_index, layer_name):
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
categories = ['Intraventricular', 'Intraparenchymal', 'Subarachnoid', 'Epidural', 'Subdural', 'No_Hemorrhage', 'Fracture_Yes_No']
brain_layer_name = 'conv2d_1'  # Layer name for brain images
bone_layer_name = 'conv2d_3'   # Layer name for bone images

# Use one of the paths from your test set
img_path = image_paths[0][0]  # Path to a test brain image
bone_img_path = image_paths[0][1]  # Path to a test bone image

# Preprocess the images
brain_img = preprocess_image(img_path, IMG_SIZE)
bone_img = preprocess_image(bone_img_path, IMG_SIZE)

if brain_img is not None and bone_img is not None:
    brain_img_array = np.expand_dims(brain_img, axis=0) / 255.0
    bone_img_array = np.expand_dims(bone_img, axis=0) / 255.0

    plt.figure(figsize=(20, 40))
    for i, category in enumerate(categories):
        # Get Grad-CAM heatmap for brain image for each category
        brain_heatmap = get_grad_cam(model, brain_img_array, bone_img_array, category_index=i, layer_name=brain_layer_name)

        # Get Grad-CAM heatmap for bone image for each category
        bone_heatmap = get_grad_cam(model, brain_img_array, bone_img_array, category_index=i, layer_name=bone_layer_name)
        
        # Display the Grad-CAM heatmap overlayed on the original image
        plt.subplot(len(categories), 2, 2 * i + 1)
        plt.imshow(brain_img / 255.0)
        plt.imshow(brain_heatmap, cmap='jet', alpha=0.5)
        plt.title(f'Grad-CAM: Brain Image - {category}')

        plt.subplot(len(categories), 2, 2 * i + 2)
        plt.imshow(bone_img / 255.0)
        plt.imshow(bone_heatmap, cmap='jet', alpha=0.5)
        plt.title(f'Grad-CAM: Bone Image - {category}')

    plt.tight_layout()
    plt.show()
else:
    print("Error: One or both images could not be loaded. Please check the image paths.")
