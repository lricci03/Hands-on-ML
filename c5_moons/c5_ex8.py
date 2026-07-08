from sklearn.datasets import make_moons
from sklearn.model_selection import ShuffleSplit, train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from graphviz import Source
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter 
from sklearn.metrics import accuracy_score
from scipy.stats import mode


# the moons dataset consists of points that are arranges in two interleaving moons 
# The classification says to which moon each point belogs
# That is, each data has two features x1, x2 (coordinates) and there are two classes 0 or 1.

x_moons, y_moons = make_moons(n_samples=10000, noise=0.4, random_state=42)

# Generate one training set and one test set
x_moons_train, x_moons_test, y_moons_train, y_moons_test = train_test_split(x_moons, y_moons, random_state=42, train_size=0.8)

# Generate 1000 subset of the training set, each containing 100 instances selected randomly
# train_size: if int it represents the number of train samples
# test_size: if float it represents the proportion of the data set to include in the test set
random_split = ShuffleSplit(n_splits=1000, random_state=42, train_size = 100, test_size=None)

'''
y_moons_pred =[]

for instance in x_moons_test:
    predictions = []
    for train_index, test_index in random_split.split(x_moons_train):
        X_train = x_moons[train_index]
        y_train = y_moons[train_index]

        # The parameters are the best parameters found in the previous exercise c5_ex7.py
        tree_clf = DecisionTreeClassifier(random_state=42, max_depth=8, max_leaf_nodes=32, min_samples_leaf=10)
        tree_clf.fit(X_train,y_train)
        prediction = tree_clf.predict(instance.reshape(1, -1))
        predictions.append(prediction.item())
    final_pred = Counter(predictions).most_common(1)[0][0]
    y_moons_pred.append(final_pred)
   
accuracy = accuracy_score(y_moons_test,y_moons_pred)
print(f"Accuracy on the test set is: {accuracy}")

'''
'''
models = []

for train_index, _ in random_split.split(x_moons_train):
    X_train = x_moons_train[train_index]
    y_train = y_moons_train[train_index]

    tree_clf = DecisionTreeClassifier(random_state=42, max_depth=8, max_leaf_nodes=32, min_samples_leaf=10)
    tree_clf.fit(X_train,y_train)
    models.append(tree_clf)

y_moons_pred =[]

# This predicts one test sample at the time
for instance in x_moons_test:
    predictions = []
    for model in models:
        prediction = model.predict(instance.reshape(1,-1))
        predictions.append(prediction.item())
    #final_pred = Counter(predictions).most_common(1)[0][0]
    #y_moons_pred.append(final_pred)
    y_pred_majority = mode(predictions, keepdims=False).mode
    y_moons_pred.append(y_pred_majority)

# This predicts the test samples all at once, much faster
all_predictions = np.array([
    model.predict(x_moons_test)
    for model in models
])

y_moons_pred = mode(all_predictions, axis=0, keepdims=False).mode

accuracy = accuracy_score(y_moons_test,y_moons_pred)
print(f"Accuracy on the test set is: {accuracy}")'''
# Accuracy on the test set is: 0.833
# This is probably because on only 100 samples the restriction min_samples_leaf=10 is too strong
# Try to remove it

models = []

for train_index, _ in random_split.split(x_moons_train):
    X_train = x_moons_train[train_index]
    y_train = y_moons_train[train_index]

    tree_clf = DecisionTreeClassifier(random_state=42, max_depth=8, max_leaf_nodes=32)
    tree_clf.fit(X_train,y_train)
    models.append(tree_clf)

y_moons_pred =[]
all_predictions = np.array([
    model.predict(x_moons_test)
    for model in models
])

y_moons_pred = mode(all_predictions, axis=0, keepdims=False).mode

accuracy = accuracy_score(y_moons_test,y_moons_pred)
print(f"Accuracy on the test set is: {accuracy}")
# Accuracy on the test set is: 0.872
# Slightly better than training on the whole train set (0.871)