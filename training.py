import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

# Load X data from pickle file
with open('x.pickle', 'rb') as f:
    X_data = pickle.load(f)

# Load y data from pickle file
with open('y.pickle', 'rb') as f:
    y_data = pickle.load(f)

# Normalize the input data
X_data = X_data.astype('float32') / 255.0

# Convert labels to one-hot encoding
y_data = tf.keras.utils.to_categorical(y_data)

# Split data into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.2, random_state=42)

# Define hyperparameters
batch_size = 32
epochs = 30
input_shape = X_train.shape[1:]
num_classes = y_train.shape[1]

# Define the model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(num_classes, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(X_test, y_test))

# Save the trained model
model.save('shape_recognition_model.keras')

# Convert the model
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save tflite model
with open('model.tflite', 'wb') as f:
  f.write(tflite_model)