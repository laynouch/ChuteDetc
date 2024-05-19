
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.feature_selection import mutual_info_regression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
import warnings
import os
import random
from sklearn.preprocessing import StandardScaler

random.seed(42)
plt.rcParams.update({'font.size': 25})
sns.set_theme(color_codes=True)
warnings.filterwarnings('ignore')

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

train_df = pd.read_csv("C:/Users/dawou/OneDrive/Bureau/ML/ChuteDetc/Train.csv")
test_df = pd.read_csv('C:/Users/dawou/OneDrive/Bureau/ML/ChuteDetc/Test.csv')

"The Unnamed: 0 column in the dataframes is unnecessary, lets drop it."

train_df.drop(['Unnamed: 0'], axis = 1, inplace = True)
test_df.drop(['Unnamed: 0'], axis = 1, inplace = True)
print(train_df.head())
print(test_df.head())
print(f"Training data shape: {train_df.shape}\nTest data shape: {test_df.shape}")


X_train = train_df.drop(['fall','label'],axis=1)
y_train = train_df['fall']
X_test =  test_df.drop(['fall','label'],axis=1)
y_test =  test_df['fall']








'''-------------------------------------------------------------------------------------------------------------------------'''

from sklearn.neural_network import MLPClassifier

# Importing required libraries
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV

# Normalizing the data
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# Defining the grid of hyperparameters
hidden_layer_sizes = [(50,), (100,), (50, 50), (100, 100)]
activation = ['logistic', 'tanh', 'relu']
solver = ['adam', 'sgd']
alpha = [0.0001, 0.001, 0.01, 0.1]
learning_rate = ['constant', 'adaptive']

random_grid = {'hidden_layer_sizes': hidden_layer_sizes,
               'activation': activation,
               'solver': solver,
               'alpha': alpha,
               'learning_rate': learning_rate}

# Creating the ANN classifier
ann = MLPClassifier()

# Performing Randomized Search
ann_random = RandomizedSearchCV(estimator=ann,
                                param_distributions=random_grid,
                                n_iter=100,
                                cv=5,
                                verbose=2,
                                random_state=42,
                                n_jobs=-1)

ann_random.fit(X_train, y_train)
ann_random.best_params_

# Defining a function to evaluate the model
def evaluate_ann(model, test_features, test_labels):
    predictions = model.predict(test_features)
    accuracy = ((predictions == test_labels).sum() / test_labels.shape[0]) * 100
    print('Model Performance')
    print('Accuracy = {:0.2f}%.'.format(accuracy))
    return accuracy

params = ann_random.best_params_

# Defining ranges for hyperparameter tuning
hidden_layer_sizes = [(50,), (100,), (150,), (200,)]
alpha_values = [0.0001, 0.001, 0.01]
learning_rates = ['constant', 'adaptive']

highest_accuracy = 0

# Loop for hyperparameter tuning
for hidden_layer in hidden_layer_sizes:
    params['hidden_layer_sizes'] = hidden_layer
    for alpha_value in alpha_values:
        params['alpha'] = alpha_value
        for learning_rate_option in learning_rates:
            params['learning_rate'] = learning_rate_option
            model = MLPClassifier(**params)
            model.fit(X_train, y_train)
            accuracy = evaluate_ann(model, X_test, y_test)
            highest_accuracy = max(highest_accuracy, accuracy)

print(f"The highest accuracy obtained is: {highest_accuracy}%.")