# -*- coding: utf-8 -*-
"""
Created on Thu Jun 29 18:25:38 2023

@author: gizem
"""

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, BatchNormalization
from keras.optimizers import Adam
from pathlib import Path
import joblib
from sklearn.preprocessing import MinMaxScaler

# Load data set
x_train = joblib.load("x_train.dat")
y_train = joblib.load("y_train.dat")

# Reshape the input data to 2-dimensional format (assuming the input is image data)
num_samples, img_height, img_width, num_channels = x_train.shape
x_train = x_train.reshape(num_samples, img_height * img_width * num_channels)

# Data normalization
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)

# Create a model and add layers
model = Sequential()

model.add(Flatten(input_shape=x_train.shape[1:]))
model.add(Dense(256, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

# Compile the model with a custom learning rate
learning_rate = 0.001
optimizer = Adam(lr=learning_rate)
model.compile(
    loss="binary_crossentropy",
    optimizer=optimizer,
    metrics=['accuracy']
)

# Train the model
epochs = 30
batch_size = 32
model.fit(
    x_train,
    y_train,
    epochs=epochs,
    batch_size=batch_size,
    shuffle=True
)

# Specify the directory path to save the model files
save_dir = Path("/usr/src/myapp")

# Save neural network structure
model_structure = model.to_json()
with open(save_dir / "model_structure_2.json", "w") as json_file:
    json_file.write(model_structure)

# Save neural network's trained weights
model.save_weights(save_dir / "model_weights_2.h5")
