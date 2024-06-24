import pandas as pd
import numpy as np
import os

# Model Training and Evaluation
from sklearn.model_selection import train_test_split

# Path to the Excel file
excel_path = "C:\\Users\\dktjw\\OneDrive\\바탕 화면\\Brain hemorrhage detection model\\hemorrhage_diagnosis_final.xlsx"

# Path to the image folder
image_base_path = "C:\\Users\\dktjw\\OneDrive\\바탕 화면\\Brain hemorrhage detection model\\CT_bone_data"

# Read the Excel file
df = pd.read_excel(excel_path)

# Set label column names
labels = {
    'Intraventricular(뇌실내출혈)': 'Intraventricular',
    'Intraparenchymal(뇌실질내출혈)': 'Intraparenchymal',
    'Subarachnoid(지주막하출혈)': 'Subarachnoid',
    'Epidural(경막외출혈)': 'Epidural',
    'Subdural(경막하출혈)': 'Subdural',
    'No_Hemorrhage': 'No hemorrhage'
}

# Initialize dictionaries to store results
brain_data = []
bone_data = []

# Function to convert to three-digit number
def make_folder_name(folder_num):
    return str(folder_num).zfill(3)

# Labeling image files
for index, row in df.iterrows():
    folder_name = make_folder_name(row['PatientNumber'])  # Convert folder name to three digits
    image_name = str(row['SliceNumber']) + '.jpg'  # Add extension to image name

    # Determine label
    label = None
    for col, label_name in labels.items():
        if row[col] == 1:
            label = label_name
            break
    
    # Label images in brain and bone folders
    for subfolder in ['brain', 'bone']:
        image_folder_path = os.path.join(image_base_path, folder_name, subfolder)
        image_path = os.path.join(image_folder_path, image_name)

        # Save results
        if subfolder == 'brain':
            brain_data.append((image_path, label))
        else:
            bone_data.append((image_path, label))

# Create DataFrames for Brain_Data and Bone_Data
CT_Data = pd.DataFrame(brain_data, columns=['Image', 'Label'])
Bone_Data = pd.DataFrame(bone_data, columns=['Image', 'Label'])

# Combine Brain and Bone datasets
combined_data = pd.concat([CT_Data, Bone_Data], axis=1)

# Split into Train and Test sets
Train_Data, Test_Data = train_test_split(combined_data, train_size=0.8, shuffle=True, random_state=42)

# Separate Brain data and Bone data in each Train and Test set
Train_Data_Brain = Train_Data.iloc[:, [0, 1]]  # Select first and second columns
Train_Data_Bone = Train_Data.iloc[:, [2, 3]]   # Select third and fourth columns
Test_Data_Brain = Test_Data.iloc[:, [0, 1]]    # Select first and second columns
Test_Data_Bone = Test_Data.iloc[:, [2, 3]]     # Select third and fourth columns

# Count the number of samples for each label in the Test set
brain_test_label_counts = Test_Data_Brain['Label'].value_counts()
bone_test_label_counts = Test_Data_Bone['Label'].value_counts()

# Print results
print("\n\nBRAIN TEST LABEL COUNTS:")
print(brain_test_label_counts)

print("\nBONE TEST LABEL COUNTS:")
print(bone_test_label_counts)