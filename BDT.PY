
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



def make_mi_scores(X, y):
    mi_scores = mutual_info_regression(X, y, discrete_features=False)
    mi_scores = pd.Series(mi_scores, name="MI Scores", index=X.columns)
    mi_scores = mi_scores.sort_values(ascending=False)
    return mi_scores

mi_scores = make_mi_scores(X_train, y_train)

def plot_utility_scores(scores):
    y = scores.sort_values(ascending=True)
    width = np.arange(len(y))
    ticks = list(y.index)
    plt.barh(width, y)
    plt.yticks(width, ticks)
    plt.title("Mutual Information Scores(overall feature)")

plt.figure(dpi=100, figsize=(8, 5))
plt.xlabel("Score")
plt.ylabel("Feature")
plot_utility_scores(mi_scores)









from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import RandomizedSearchCV
import numpy as np

# Normalisation des données
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# Définition de l'espace des hyperparamètres
n_estimators = [200, 400, 600, 800, 1000]
learning_rate = [0.001, 0.01, 0.1, 0.2, 0.3]
max_depth = [3, 5, 7, 9]
min_samples_split = [2, 5, 9, 12]
min_samples_leaf = [1, 3, 5, 7]
max_features = ['auto', 'sqrt']

random_grid = {'n_estimators': n_estimators,
               'learning_rate': learning_rate,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'max_features': max_features}

# Recherche des meilleurs hyperparamètres
gb = GradientBoostingClassifier()
gb_random = RandomizedSearchCV(estimator=gb,
                               param_distributions=random_grid,
                               n_iter=100, cv=5,
                               verbose=2,
                               random_state=42,
                               n_jobs=-1)
gb_random.fit(X_train, y_train)
best_params = gb_random.best_params_

# Création du modèle avec les meilleurs hyperparamètres trouvés
gb_best = GradientBoostingClassifier(**best_params)
gb_best.fit(X_train, y_train)

# Évaluation du modèle
def evaluate(model, test_features, test_labels):
    predictions = model.predict(test_features)
    accuracy = ((predictions == test_labels).sum() / test_labels.shape[0]) * 100
    print('Performance du modèle')
    print('Précision = {:0.2f}%.'.format(accuracy))
    
    return accuracy

# Évaluation du modèle Gradient Boosting
accuracy = evaluate(gb_best, X_test, y_test)
print(f"La meilleure précision obtenue est : {accuracy}%.")
                            
