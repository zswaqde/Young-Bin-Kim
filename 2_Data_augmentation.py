## DATA AUGMENTATION

# Data Handling and Processing
import pandas as pd
import numpy as np
import os
import shutil

# Image Processing
from PIL import Image
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array, array_to_img, save_img

# Model Training and Evaluation
from sklearn.model_selection import train_test_split


# Data augmentation settings
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

def augment_data(brain_df, bone_df, output_folder, target_count=2500):
    os.makedirs(output_folder, exist_ok=True)

    augmented_brain_data = []
    augmented_bone_data = []

    for label in brain_df['Label'].unique():
        brain_label_df = brain_df[brain_df['Label'] == label]
        bone_label_df = bone_df[bone_df['Label'] == label]

        current_count = len(brain_label_df)
        label_brain_folder = os.path.join(output_folder, f"{label}_brain")
        label_bone_folder = os.path.join(output_folder, f"{label}_bone")
        os.makedirs(label_brain_folder, exist_ok=True)
        os.makedirs(label_bone_folder, exist_ok=True)

        # Sequentially copy existing images to label folders
        for idx, (brain_row, bone_row) in enumerate(zip(brain_label_df.itertuples(), bone_label_df.itertuples()), start=1):
            brain_img_path = brain_row.Image
            bone_img_path = bone_row.Image

            dest_brain_path = os.path.join(label_brain_folder, f"{label}_brain{idx}.jpg")
            dest_bone_path = os.path.join(label_bone_folder, f"{label}_bone{idx}.jpg")

            shutil.copy(brain_img_path, dest_brain_path)
            shutil.copy(bone_img_path, dest_bone_path)

            augmented_brain_data.append((dest_brain_path, label))
            augmented_bone_data.append((dest_bone_path, label))

        # Augment to make up for missing count
        augment_count = target_count - current_count
        if augment_count > 0:
            brain_imgs_to_augment = brain_label_df['Image'].tolist()
            bone_imgs_to_augment = bone_label_df['Image'].tolist()
            augment_per_img = augment_count // len(brain_imgs_to_augment) + 1

            for brain_img_path, bone_img_path in zip(brain_imgs_to_augment, bone_imgs_to_augment):
                brain_img = load_img(brain_img_path)
                bone_img = load_img(bone_img_path)

                x_brain = img_to_array(brain_img)
                x_bone = img_to_array(bone_img)

                x_brain = x_brain.reshape((1,) + x_brain.shape)
                x_bone = x_bone.reshape((1,) + x_bone.shape)

                i = 0
                for brain_batch, bone_batch in zip(datagen.flow(x_brain, batch_size=1), datagen.flow(x_bone, batch_size=1)):
                    if i >= augment_per_img or len(os.listdir(label_brain_folder)) >= target_count:
                        break

                    new_brain_img = array_to_img(brain_batch[0])
                    new_bone_img = array_to_img(bone_batch[0])

                    current_index = len(os.listdir(label_brain_folder)) + 1

                    new_brain_img_name = f"{label}_brain{current_index}.jpg"
                    new_bone_img_name = f"{label}_bone{current_index}.jpg"

                    new_brain_img_path = os.path.join(label_brain_folder, new_brain_img_name)
                    new_bone_img_path = os.path.join(label_bone_folder, new_bone_img_name)

                    save_img(new_brain_img_path, brain_batch[0])
                    save_img(new_bone_img_path, bone_batch[0])

                    augmented_brain_data.append((new_brain_img_path, label))
                    augmented_bone_data.append((new_bone_img_path, label))

                    i += 1

    # Create augmented DataFrames
    augmented_brain_df = pd.DataFrame(augmented_brain_data, columns=['Image', 'Label'])
    augmented_bone_df = pd.DataFrame(augmented_bone_data, columns=['Image', 'Label'])

    return augmented_brain_df, augmented_bone_df

def find_missing_numbers(nums, target_count=2500):
    """
    Find and return missing numbers in the given list.
    """
    missing_nums = [i for i in range(1, target_count + 1) if i not in nums]
    return missing_nums

def fill_missing_images(df, output_folder, target_count=2500):
    """
    Find missing numbers in the given DataFrame and augment data to fill those numbers.
    """
    global datagen

    augmented_brain_data = []
    augmented_bone_data = []

    for label in df['Label'].unique():
        label_folder = os.path.join(output_folder, f"{label}_brain")
        existing_files = [int(f.split(label + "_brain")[1].split(".jpg")[0]) for f in os.listdir(label_folder)]
        missing_nums = find_missing_numbers(existing_files, target_count)

        if not missing_nums:
            # Already have 2500 images
            continue

        for num in missing_nums:
            brain_img_path = df[df['Label'] == label]['Image'].sample().values[0]
            bone_img_path = brain_img_path.replace('brain', 'bone')

            brain_img = load_img(brain_img_path)
            bone_img = load_img(bone_img_path)

            x_brain = img_to_array(brain_img)
            x_bone = img_to_array(bone_img)

            x_brain = x_brain.reshape((1,) + x_brain.shape)
            x_bone = x_bone.reshape((1,) + x_bone.shape)

            brain_batch = next(datagen.flow(x_brain, batch_size=1))
            bone_batch = next(datagen.flow(x_bone, batch_size=1))

            new_brain_img = array_to_img(brain_batch[0])
            new_bone_img = array_to_img(bone_batch[0])

            new_brain_img_name = f"{label}_brain{num}.jpg"
            new_bone_img_name = f"{label}_bone{num}.jpg"

            new_brain_img_path = os.path.join(label_folder, new_brain_img_name)
            new_bone_img_path = os.path.join(label_folder.replace('brain', 'bone'), new_bone_img_name)

            save_img(new_brain_img_path, brain_batch[0])
            save_img(new_bone_img_path, bone_batch[0])

            augmented_brain_data.append((new_brain_img_path, label))
            augmented_bone_data.append((new_bone_img_path, label))

    # Add newly created images to the existing DataFrame
    augmented_brain_df = pd.concat([df, pd.DataFrame(augmented_brain_data, columns=['Image', 'Label'])])
    augmented_bone_df = pd.concat([df, pd.DataFrame(augmented_bone_data, columns=['Image', 'Label'])])

    return augmented_brain_df, augmented_bone_df

# Example DataFrames creation (replace with actual data)
# Train_Data_Brain = pd.DataFrame([...])  # Actual Train Brain DataFrame
# Train_Data_Bone = pd.DataFrame([...])  # Actual Train Bone DataFrame

# Set folder to save augmented data
output_folder = 'Augmented_Data'

# Perform data augmentation
Augmented_Train_Data_Brain, Augmented_Train_Data_Bone = augment_data(Train_Data_Brain, Train_Data_Bone, output_folder)

# Fill missing images if any and augment
filled_Augmented_Train_Data_Brain, filled_Augmented_Train_Data_Bone = fill_missing_images(Augmented_Train_Data_Brain, output_folder)
filled_Augmented_Train_Data_Brain, filled_Augmented_Train_Data_Bone = fill_missing_images(Augmented_Train_Data_Bone, output_folder)

# Print final results
print("Augmented Train Data Brain:")
print(filled_Augmented_Train_Data_Brain.head())

print("\nAugmented Train Data Bone:")
print(filled_Augmented_Train_Data_Bone.head())


## Code to reload the generated image files

def load_augmented_images(output_folder):
    brain_data = []
    bone_data = []

    for label_folder in os.listdir(output_folder):
        if label_folder.endswith('_brain'):
            label = label_folder.split('_brain')[0]
            brain_folder_path = os.path.join(output_folder, label_folder)
            bone_folder_path = brain_folder_path.replace('_brain', '_bone')
            
            for brain_img_file in os.listdir(brain_folder_path):
                brain_img_path = os.path.join(brain_folder_path, brain_img_file)
                bone_img_path = os.path.join(bone_folder_path, brain_img_file.replace('_brain', '_bone'))
                
                brain_data.append((brain_img_path, label))
                bone_data.append((bone_img_path, label))

    brain_df = pd.DataFrame(brain_data, columns=['Image', 'Label'])
    bone_df = pd.DataFrame(bone_data, columns=['Image', 'Label'])

    return brain_df, bone_df

# Folder where augmented data is saved
output_folder = 'Augmented_Data'

# Load images from the folder and create DataFrame
Augmented_Train_Data_Brain, Augmented_Train_Data_Bone = load_augmented_images(output_folder)

# Print final results
print("Augmented Train Data Brain:")
print(Augmented_Train_Data_Brain.head())

print("\nAugmented Train Data Bone:")
print(Augmented_Train_Data_Bone.head())

# Check the number of images
print("\nTotal number of brain images:", len(Augmented_Train_Data_Brain))
print("Total number of bone images:", len(Augmented_Train_Data_Bone))