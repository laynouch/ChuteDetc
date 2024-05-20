import tensorflow as tf
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import GradientBoostingClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.feature_selection import mutual_info_regression
from sklearn.metrics import accuracy_score
import warnings
import os
import random
from sklearn.preprocessing import StandardScaler

# Set random seed for reproducibility
random.seed(42)
plt.rcParams.update({'font.size': 25})
sns.set_theme(color_codes=True)
warnings.filterwarnings('ignore')

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

# Calculate mutual information scores
def make_mi_scores(X, y):
    mi_scores = mutual_info_regression(X, y, discrete_features=False)
    mi_scores = pd.Series(mi_scores, name="MI Scores", index=X.columns)
    mi_scores = mi_scores.sort_values(ascending=False)
    return mi_scores

mi_scores = make_mi_scores(X_train, y_train)

# Normalize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Define hyperparameter grid
n_estimators = [200, 400, 600, 800, 1000]
learning_rate = [0.001, 0.01, 0.1, 0.2, 0.3]
max_depth = [3, 5, 7, 9]
min_samples_split = [2, 5, 9, 12]
min_samples_leaf = [1, 3, 5, 7]
max_features = ['auto', 'sqrt']

random_grid = {
    'n_estimators': n_estimators,
    'learning_rate': learning_rate,
    'max_depth': max_depth,
    'min_samples_split': min_samples_split,
    'min_samples_leaf': min_samples_leaf,
    'max_features': max_features
}

# Hyperparameter search
gb = GradientBoostingClassifier()
gb_random = RandomizedSearchCV(estimator=gb,
                               param_distributions=random_grid,
                               n_iter=100, cv=5,
                               verbose=2,
                               random_state=42,
                               n_jobs=-1)
gb_random.fit(X_train, y_train)
best_params = gb_random.best_params_

# Train the best model
gb_best = GradientBoostingClassifier(**best_params)
gb_best.fit(X_train, y_train)

# Convert the scikit-learn model to a TensorFlow model
input_shape = X_train.shape[1]
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(input_shape,)),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Transfer the weights from the scikit-learn model to the TensorFlow model
gb_best_params = gb_best.get_params()
model.set_weights([np.array(gb_best_params[f'estimators_'][i][1].coef_.flatten()) for i in range(best_params['n_estimators'])])

# Convert the TensorFlow model to a TensorFlow Lite model
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the TensorFlow Lite model to a file
with open('model.tflite', 'wb') as f:
    f.write(tflite_model)

print("TensorFlow Lite model saved to 'model.tflite'.")
