from pathlib import Path
import numpy as np
import joblib
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import xception
from PIL import Image

# Path to folders with image data
train_path = Path("C:/Users/gizem/OneDrive/Belgeler/Fungi Images/Training Dataset") 
test_path = Path("C:/Users/gizem/OneDrive/Belgeler/Fungi Images/Test Dataset") 

images = []
labels = []

# Load all the not-AFU images
for img in test_path.glob("*.jpg"):
    # Load the image from disk
    img = Image.open(img)

    # Resize the image to match the expected input shape of Xception
    img = img.resize((299, 299))

    # Convert the image to a numpy array
    image_array = image.img_to_array(img)

    # Add the image to the list of images
    images.append(image_array)

    # For each 'not AFU' image, the expected value should be 0
    labels.append(0)

# Load all the AFU images
for img in train_path.glob("*.jpg"):
    # Load the image from disk
    img = Image.open(img)

    # Resize the image to match the expected input shape of Xception
    img = img.resize((299, 299))

    # Convert the image to a numpy array
    image_array = image.img_to_array(img)

    # Add the image to the list of images
    images.append(image_array)

    # For each 'AFU' image, the expected value should be 1
    labels.append(1)

# Create a single numpy array with all the images we loaded
x_train = np.array(images)

# Also convert the labels to a numpy array
y_train = np.array(labels)

# Preprocess input data specific to Xception
x_train = xception.preprocess_input(x_train)

# Load a pre-trained neural network (Xception) to use as a feature extractor
pretrained_nn = xception.Xception(weights='imagenet', include_top=False, input_shape=(299, 299, 3))

# Extract features for each image (all in one pass)
features_x = pretrained_nn.predict(x_train)

# Save the array of extracted features to a file
joblib.dump(features_x, "x_train.dat")

# Save the matching array of expected values to a file
joblib.dump(y_train, "y_train.dat")
