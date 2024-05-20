import tensorflow as tf
from tensorflow.keras import layers, models
from scikeras.wrappers import KerasClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import random
import warnings

random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

plt.rcParams.update({'font.size': 25})
sns.set_theme(color_codes=True)
warnings.filterwarnings('ignore')

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

train_df = pd.read_csv("C:/Users/dawou/OneDrive/Bureau/ML/ChuteDetc/Train.csv")
test_df = pd.read_csv('C:/Users/dawou/OneDrive/Bureau/ML/ChuteDetc/Test.csv')

train_df.drop(['Unnamed: 0'], axis=1, inplace=True)
test_df.drop(['Unnamed: 0'], axis=1, inplace=True)

X_train = train_df.drop(['fall', 'label'], axis=1)
y_train = train_df['fall']
X_test = test_df.drop(['fall', 'label'], axis=1)
y_test = test_df['fall']

scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

def create_model(learning_rate=0.01, dropout_rate=0.2):
    model = models.Sequential()
    model.add(layers.Dense(64, input_dim=X_train.shape[1], activation='relu'))
    model.add(layers.Dropout(dropout_rate))
    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dropout(dropout_rate))
    model.add(layers.Dense(1, activation='sigmoid'))
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

model = KerasClassifier(model=create_model, verbose=0)

param_dist = {
    'model__learning_rate': [0.001, 0.01, 0.1],
    'model__dropout_rate': [0.2, 0.3, 0.4],
    'batch_size': [32, 64, 128],
    'epochs': [50, 100, 200]
}

random_search = RandomizedSearchCV(estimator=model, param_distributions=param_dist,
                                   n_iter=10, cv=3, verbose=2, random_state=42, n_jobs=-1)
random_search.fit(X_train, y_train)
best_params = random_search.best_params_

final_model = create_model(learning_rate=best_params['model__learning_rate'], dropout_rate=best_params['model__dropout_rate'])
history = final_model.fit(X_train, y_train, 
                          epochs=best_params['epochs'], 
                          batch_size=best_params['batch_size'], 
                          validation_data=(X_test, y_test), 
                          verbose=2)

def evaluate(model, test_features, test_labels):
    predictions = (model.predict(test_features) > 0.5).astype("int32")
    predictions = np.ravel(predictions)  # Ensure predictions is a 1D array
    accuracy = ((predictions == test_labels).sum() / test_labels.shape[0]) * 100
    print('Performance du modèle')
    print('Précision = {:0.2f}%.'.format(accuracy))
    return accuracy

accuracy = evaluate(final_model, X_test, y_test)
print(f"La meilleure précision obtenue est : {accuracy:.2f}%.")

# Define the directory and file path to save the model
save_path = 'C:/Users/dawou/OneDrive/Bureau/ML/ChuteDetc/saved_model.h5'

# Save the Sequential model in HDF5 format
final_model.save(save_path)

# Convert the saved HDF5 model to TensorFlow Lite format
converter = tf.lite.TFLiteConverter.from_keras_model(final_model)
tflite_model = converter.convert()

# Define the file path for the TensorFlow Lite model
tflite_model_path = 'model.tflite'

# Save the TensorFlow Lite model to the specified file path
with open(tflite_model_path, 'wb') as f:
    f.write(tflite_model)

print(f"Model has been converted to TensorFlow Lite format and saved as {tflite_model_path}")