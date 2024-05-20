import tensorflow as tf
from tensorflow.keras import layers, models
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load data
train_df = pd.read_csv("C:/Users/dawou/OneDrive/Bureau/ML/ChuteDetc/Train.csv")
test_df = pd.read_csv('C:/Users/dawou/OneDrive/Bureau/ML/ChuteDetc/Test.csv')

# Drop unnecessary columns
train_df.drop(['Unnamed: 0'], axis=1, inplace=True)
test_df.drop(['Unnamed: 0'], axis=1, inplace=True)

# Split data into features and target
X_train = train_df.drop(['fall', 'label'], axis=1)
y_train = train_df['fall']
X_test = test_df.drop(['fall', 'label'], axis=1)
y_test = test_df['fall']

# Normalize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Build a neural network model using TensorFlow's Keras API
model = models.Sequential([
    layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    layers.Dense(32, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

from tensorflow.keras.models import load_model
model = load_model('saved_model.h5')
from tensorflow.contrib import lite
converter = lite.TFLiteConverter.from_keras_model_file( 'final_model.h5')
tfmodel = converter.convert()
open ("model.tflite" , "wb") .write(tfmodel)
# Convert the model to TensorFlow Lite format
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the TensorFlow Lite model to a file
with open('model.tflite', 'wb') as f:
    f.write(tflite_model)

print("TensorFlow Lite model has been saved.")
