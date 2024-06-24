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
from sklearn.metrics import accuracy_score

# Model Building
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, BatchNormalization, MaxPooling2D, \
                                    Input, concatenate, multiply
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


# Build the enhanced CNN model for brain images
input_brain = Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
x1 = Conv2D(32, (3, 3), activation='relu', padding='same')(input_brain)
x1 = BatchNormalization()(x1)
x1 = MaxPooling2D((2, 2))(x1)
x1 = Dropout(0.3)(x1)

x1 = Conv2D(64, (3, 3), activation='relu', padding='same')(x1)
x1 = BatchNormalization()(x1)
x1 = MaxPooling2D((2, 2))(x1)
x1 = Dropout(0.3)(x1)

x1 = Conv2D(128, (3, 3), activation='relu', padding='same')(x1)
x1 = BatchNormalization()(x1)
x1 = MaxPooling2D((2, 2))(x1)
x1 = Dropout(0.4)(x1)

x1 = Flatten()(x1)

# Build the enhanced CNN model for bone images
input_bone = Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
x2 = Conv2D(32, (3, 3), activation='relu', padding='same')(input_bone)
x2 = BatchNormalization()(x2)
x2 = MaxPooling2D((2, 2))(x2)
x2 = Dropout(0.3)(x2)

x2 = Conv2D(64, (3, 3), activation='relu', padding='same')(x2)
x2 = BatchNormalization()(x2)
x2 = MaxPooling2D((2, 2))(x2)
x2 = Dropout(0.3)(x2)

x2 = Conv2D(128, (3, 3), activation='relu', padding='same')(x2)
x2 = BatchNormalization()(x2)
x2 = MaxPooling2D((2, 2))(x2)
x2 = Dropout(0.4)(x2)

x2 = Flatten()(x2)

# Classification layer for brain-only model
output_brain_only = Dense(6, activation='softmax')(x1)

# Combine the outputs using concatenate
combined_concat = concatenate([x1, x2])

# Product fusion
combined_product = multiply([x1, x2])

# Classification layer for concatenate fusion
combined_concat = Dense(64, activation='relu')(combined_concat)
combined_concat = Dropout(0.5)(combined_concat)
output_concat = Dense(6, activation='softmax')(combined_concat)

# Classification layer for product fusion
combined_product = Dense(64, activation='relu')(combined_product)
combined_product = Dropout(0.5)(combined_product)
output_product = Dense(6, activation='softmax')(combined_product)


# Define models
model_concat = Model(inputs=[input_brain, input_bone], outputs=output_concat)
model_product = Model(inputs=[input_brain, input_bone], outputs=output_product)
model_brain_only = Model(inputs=input_brain, outputs=output_brain_only)


# Compile models
model_concat.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model_product.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model_brain_only.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Summary
# model_concat.summary()
# model_product.summary()
# model_brain_only.summar()

# Callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
model_checkpoint_concat = ModelCheckpoint('best_model_concat.keras', monitor='val_accuracy', save_best_only=True, mode='max')
model_checkpoint_product = ModelCheckpoint('best_model_product.keras', monitor='val_accuracy', save_best_only=True, mode='max')
model_checkpoint_brain_only = ModelCheckpoint('best_model_brain_only.keras', monitor='val_accuracy', save_best_only=True, mode='max')


# Train the models
history_concat = model_concat.fit([brain_train_images, bone_train_images], brain_train_labels_encoded, 
                                  epochs=50, 
                                  batch_size=32, 
                                  validation_data=([brain_test_images, bone_test_images], brain_test_labels_encoded), 
                                  callbacks=[early_stopping, model_checkpoint_concat])

history_product = model_product.fit([brain_train_images, bone_train_images], brain_train_labels_encoded, 
                                    epochs=50, 
                                    batch_size=32, 
                                    validation_data=([brain_test_images, bone_test_images], brain_test_labels_encoded), 
                                    callbacks=[early_stopping, model_checkpoint_product])

history_brain_only = model_brain_only.fit(brain_train_images, brain_train_labels_encoded,
                                          epochs=50,
                                          batch_size=32,
                                          validation_data=(brain_test_images, brain_test_labels_encoded),
                                          callbacks=[early_stopping])


# Plot training & validation accuracy values for all three models
plt.plot(history_brain_only.history['accuracy'])
plt.plot(history_brain_only.history['val_accuracy'])
plt.plot(history_concat.history['accuracy'])
plt.plot(history_concat.history['val_accuracy'])
plt.plot(history_product.history['accuracy'])
plt.plot(history_product.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Brain Only Train', 'Brain Only Val', 'Concat Train', 'Concat Val', 'Product Train', 'Product Val'], loc='upper left')
plt.show()

# Plot training & validation loss values for all three models
plt.plot(history_brain_only.history['loss'])
plt.plot(history_brain_only.history['val_loss'])
plt.plot(history_concat.history['loss'])
plt.plot(history_concat.history['val_loss'])
plt.plot(history_product.history['loss'])
plt.plot(history_product.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Brain Only Train', 'Brain Only Val', 'Concat Train', 'Concat Val', 'Product Train', 'Product Val'], loc='upper left')
plt.show()

# Print the best accuracy of each model
best_val_accuracy_brain_only = max(history_brain_only.history['accuracy'])
best_val_accuracy_concat = max(history_concat.history['accuracy'])
best_val_accuracy_product = max(history_product.history['accuracy'])

print(f'Best Validation Accuracy for Brain Only Model: {best_val_accuracy_brain_only:.4f}')
print(f'Best Validation Accuracy for Concatenate Model: {best_val_accuracy_concat:.4f}')
print(f'Best Validation Accuracy for Product Model: {best_val_accuracy_product:.4f}')


def labels_to_string(labels):
    return np.argmax(labels, axis=1)

def evaluate_and_compare_model(model_path, brain_test_images, bone_test_images, brain_test_labels_encoded, model_type='combined'):
    best_model = load_model(model_path)
    
    if model_type == 'brain_only':
        test_loss, test_accuracy = best_model.evaluate(brain_test_images, brain_test_labels_encoded)
        predicted_labels = best_model.predict(brain_test_images)
    else:
        test_loss, test_accuracy = best_model.evaluate([brain_test_images, bone_test_images], brain_test_labels_encoded)
        predicted_labels = best_model.predict([brain_test_images, bone_test_images])
    
    print(f"Test Accuracy: {test_accuracy}, Test Loss: {test_loss}")
    
    predicted_labels_binary = (predicted_labels > 0.5).astype(int)
    
    brain_test_labels_decoded = labels_to_string(brain_test_labels_encoded)
    predicted_labels_decoded = labels_to_string(predicted_labels_binary)
    
    test_accuracy = accuracy_score(brain_test_labels_decoded, predicted_labels_decoded)
    
    comparison = pd.DataFrame({
        'Actual': brain_test_labels_decoded,
        'Predicted': predicted_labels_decoded
    })
    
    return comparison

comparison_brain_only = evaluate_and_compare_model('best_model_brain_only.keras', brain_test_images, None, brain_test_labels_encoded, model_type='brain_only')
comparison_concat = evaluate_and_compare_model('best_model_concat.keras', brain_test_images, bone_test_images, brain_test_labels_encoded, model_type='combined')
comparison_product = evaluate_and_compare_model('best_model_product.keras', brain_test_images, bone_test_images, brain_test_labels_encoded, model_type='combined')

# Save the comparisons to CSV and Excel files
comparison_brain_only.to_excel('prediction_comparison_brain_only.xlsx', index=False)
comparison_concat.to_excel('prediction_comparison_concat.xlsx', index=False)
comparison_product.to_excel('prediction_comparison_product.xlsx', index=False)

print("Comparison for Brain Only model:")
print(comparison_brain_only.head())

print("Comparison for Concat model:")
print(comparison_concat.head())

print("Comparison for Product model:")
print(comparison_product.head())
