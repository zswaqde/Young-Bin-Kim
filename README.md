# Intracranial Hemorrhage and Fracture Diagnosis

This repository contains code for diagnosing intracranial hemorrhages from CT scans using a late fusion CNN model. The model processes two types of images: brain images and bone images, and it is capable of identifying multiple conditions such as intraventricular hemorrhage, intraparenchymal hemorrhage, subarachnoid hemorrhage, epidural hemorrhage, and subdural hemorrhage, as well as determining the absence of hemorrhage.

![image](https://github.com/zswaqde/Young-Bin-Kim/assets/173070113/78b83477-9c85-421f-a1ac-e424afe35e17)

## Project Summary

This project aims to develop a deep learning model that can accurately diagnose various types of intracranial hemorrhages and fractures from CT scans. The model leverages a late fusion approach, combining information from brain and bone images to improve classification performance. The results show promising accuracy and provide visual insights into the model's decision-making process through Grad-CAM heatmaps.

![image](https://github.com/zswaqde/Young-Bin-Kim/assets/173070113/174dd795-2e3a-4221-b51b-5c59acaea4bc)


## Dependencies

- Python 3.7+
- TensorFlow 2.x
- NumPy
- Pandas
- Matplotlib
- Seaborn
- OpenCV

Dependencies can be installed via pip:


sadfsad
pip install -r requirements.txt

or via conda:

conda install --file requirements.txt

Setup
Clone the repository:

git clone https://github.com/yourusername/intracranial-hemorrhage-diagnosis.git
cd intracranial-hemorrhage-diagnosis

Install the required packages:
pip install -r requirements.txt

Download the required data and place it in the appropriate directory structure:

Data should be placed in D:/late fusion_2/Patients_CT.
The Excel file hemorrhage_diagnosis1.xlsx should be located in D:/late fusion_2/.

Preprocessing
The images are preprocessed by resizing them to 128x128 pixels and normalizing the pixel values. The preprocessing function is defined as:

def preprocess_image(image_path, target_size):
    try:
        img = load_img(image_path, target_size=target_size)
        img_array = img_to_array(img)
        return img_array
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None

![image](https://github.com/zswaqde/Young-Bin-Kim/assets/173070113/b32b9711-f7f3-49e1-8e3d-0d5bba47152a)


Training
To train the model, run the script train_model.py. This script will load the dataset, preprocess the images, split the data into training, validation, and test sets, and train a late fusion CNN model.

python train_model.py

Model Architecture
The model consists of two parallel CNNs for processing brain and bone images, respectively. The outputs of these CNNs are concatenated and passed through a series of dense layers. The final output layer uses a sigmoid activation function for multi-label classification.

Evaluation
After training, the model is evaluated on the test set, and performance metrics such as accuracy and loss are printed. The results are also saved in a CSV and Excel file for further analysis.

test_loss, test_accuracy = model.evaluate([X_test_brain, X_test_bone], y_test)
print(f"Test Accuracy: {test_accuracy}, Test Loss: {test_loss}")


Results
Evaluation Tables
The evaluation results are saved in prediction_comparison.csv and prediction_comparison.xlsx. These files contain actual and predicted labels for the test set, allowing for detailed analysis.

Here is a sample of the evaluation table:

PatientNumber	SliceNumber	Actual	Predicted
62	3	[0, 0, 0, 0, 0, 1, 0]	[0, 0, 0, 0, 0, 1, 0]
89	3	[0, 0, 0, 0, 0, 1, 0]	[0, 0, 0, 0, 0, 1, 0]
51	7	[0, 0, 0, 0, 0, 1, 0]	[0, 0, 0, 0, 0, 1, 0]
56	7	[0, 0, 0, 0, 0, 1, 0]	[0, 0, 0, 0, 0, 1, 0]
97	14	[0, 0, 0, 0, 0, 1, 1]	[0, 0, 0, 0, 0, 1, 1]

![image](https://github.com/zswaqde/Young-Bin-Kim/assets/173070113/4994184f-920a-42ab-8e78-93c56f24e461)


Grad-CAM visualizations are generated to understand the regions of the images that the model focuses on for making predictions. The visualizations are saved as figures for each category of diagnosis.

Example Grad-CAM visualizations for each category:

Visualization
To generate Grad-CAM visualizations, run the script grad_cam.py.

python grad_cam.py

![image](https://github.com/zswaqde/Young-Bin-Kim/assets/173070113/4f027124-080e-40e7-9dfc-b709ea189394)


Example visualizations for each category will be displayed, helping to interpret the model's predictions.

Downloading the Trained Model
If a pre-trained model is not provided within the repository, you can download it from this link. After downloading, place the model file in the root directory of the repository.

Contribution
Feel free to fork this repository and contribute by submitting pull requests. For major changes, please open an issue first to discuss what you would like to change.

Make sure to:

- Adjust the paths and repository URL to match your actual setup.
- Add a link to download the pre-trained model if it's not included in the repository.
- Create a `requirements.txt` file listing all the dependencies for easy setup.
- Include example figures or evaluation tables if they are available.

This `README.md` should now cover all the requirements specified in your instructions.

License
This project is licensed under the MIT License. See the LICENSE file for more details.

Acknowledgments
Special thanks to all the contributors and the open-source community for their invaluable tools and libraries.


Make sure to:

- Adjust the paths and repository URL to match your actual setup.
- Add a link to download the pre-trained model if it's not included in the repository.
- Create a `requirements.txt` file listing all the dependencies for easy setup.
- Include example figures or evaluation tables if they are available.

This `README.md` should now cover all the requirements specified in your instructions.









