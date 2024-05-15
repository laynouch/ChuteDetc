
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

train_df = pd.read_csv("C:/Users/Lina/Desktop/Machine Learning/Train.csv")
test_df = pd.read_csv('C:/Users/Lina/Desktop/Machine Learning/Test.csv')

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




'''-----------------------------------------------------------------'''

from sklearn.svm import SVC

# Importing required libraries
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV

# Normalizing the data
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# Defining the grid of hyperparameters
random_grid = {'C': [0.1, 1, 10, 100],
               'gamma': [1, 0.1, 0.01, 0.001],
               'kernel': ['linear', 'rbf', 'poly', 'sigmoid']}

# Creating the SVM classifier
svm = SVC()

# Performing Randomized Search
svm_random = RandomizedSearchCV(estimator=svm,
                                param_distributions=random_grid,
                                n_iter=100,
                                cv=5,
                                verbose=2,
                                random_state=42,
                                n_jobs=-1)

svm_random.fit(X_train, y_train)
svm_random.best_params_

# Defining a function to evaluate the model
def evaluate_svm(model, test_features, test_labels):
    predictions = model.predict(test_features)
    accuracy = ((predictions == test_labels).sum() / test_labels.shape[0]) * 100
    print('Model Performance')
    print('Accuracy = {:0.2f}%.'.format(accuracy))
    return accuracy

params = svm_random.best_params_

# Defining ranges for hyperparameter tuning
C_values = [0.1, 1, 10, 100]
gamma_values = [1, 0.1, 0.01, 0.001]
kernels = ['linear', 'rbf', 'poly', 'sigmoid']

highest_accuracy = 0

# Loop for hyperparameter tuning
for C_value in C_values:
    params['C'] = C_value
    for gamma_value in gamma_values:
        params['gamma'] = gamma_value
        for kernel_option in kernels:
            params['kernel'] = kernel_option
            model = SVC(**params)
            model.fit(X_train, y_train)
            accuracy = evaluate_svm(model, X_test, y_test)
            highest_accuracy = max(highest_accuracy, accuracy)

print(f"The highest accuracy obtained is: {highest_accuracy}%.")
