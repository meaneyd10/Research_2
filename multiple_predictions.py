# -*- coding: utf-8 -*-
"""
Created on Wed Jul 12 20:27:51 2023 -multiple predictions

@author: gizem
"""

from keras.models import model_from_json
from pathlib import Path
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import xception
from tensorflow.keras.layers import GlobalAveragePooling2D, Reshape
import numpy as np

# Load the json file that contains the model's structure
f = Path("C:/Users/gizem/Masaüstü/Thesis Codes/model_structure.json")
model_structure = f.read_text()

# Recreate the Keras model object from the json data
model = model_from_json(model_structure)

# Re-load the model's trained weights
model.load_weights("C:/Users/gizem/Masaüstü/Thesis Codes/model_weights.h5")

# Print the model summary to identify the correct layer name or index
model.summary()

# Set the path to your dataset directory
dataset_dir = "C:/Users/gizem/OneDrive/Belgeler/Fungi Images/Test Dataset"

# Get a list of image file paths in the dataset directory
image_paths = list(Path(dataset_dir).glob("*.jpg"))

# Create an empty list to store the predictions
predictions = []

# Loop over each image path, load and preprocess the images, and make predictions
for img_path in image_paths:
    # Load an image file to test, resizing it to the input shape of the model
    img = image.load_img(img_path, target_size=(299, 299))

    # Convert the image to a numpy array
    image_array = image.img_to_array(img)

    # Add a fourth dimension to the image (since Keras expects a batch of images, not a single image)
    images = np.expand_dims(image_array, axis=0)

    # Preprocess the input data specific to Xception
    images = xception.preprocess_input(images)

    # Use the pre-trained Xception model to extract features from the test image
    feature_extraction_model = xception.Xception(weights='imagenet', include_top=False, input_shape=(299, 299, 3))
    features = feature_extraction_model.predict(images)

    # Resize the features to match the expected input shape of the model
    resized_features = GlobalAveragePooling2D()(features)
    resized_features = Reshape((2, 2, 512))(resized_features)  # Modify the shape to match the expected input shape

    # Given the resized features, make a final prediction using our own model
    result = model.predict(resized_features)

    # Since we are only testing one image with a possible class, we only need to check the first result's first element
    single_result = result[0][0]

    # Append the prediction to the list
    predictions.append(single_result)

# Convert the list of predictions to a numpy array
predictions = np.array(predictions)

# Print the results for each image
for img_path, prediction in zip(image_paths, predictions):
    print("Image: {}, Likelihood that this image contains Aspergillus Fumigatus: {}%".format(img_path, int(prediction * 100)))

# Save the results to a file
output_directory = "C:/Users/gizem/Masaüstü/Thesis Codes"
output_file = Path(output_directory) / "predictions.txt"
with open(output_file, "w") as file:
    for img_path, prediction in zip(image_paths, predictions):
        file.write("Image: {}, Likelihood that this image contains Aspergillus Fumigatus: {}%\n".format(img_path, int(prediction * 100)))
