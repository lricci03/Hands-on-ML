from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.model_selection import GridSearchCV
from graphviz import Source
import numpy as np
import matplotlib.pyplot as plt

# the moons dataset consists of points that are arranges in two interleaving moons 
# The classification says to which moon each point belogs
# That is, each data has two features x1, x2 (coordinates) and there are two classes 0 or 1.

x_moons, y_moons = make_moons(n_samples=10000,noise=0.4, random_state=42)

x_moons_train, x_moons_test, y_moons_train, y_moons_test = train_test_split(x_moons, y_moons, test_size=0.2, random_state=42)

'''
# use grid search with cross-validation to find good hyperparameter values for a DecisionTreeClassifier
# can either put 'random_state':[42] as a parameter or do tree_clf = DecisionTreeClassifier(random_state=42)
parameters = {'max_depth':[2,3,4,5,6,8],'min_samples_leaf':[5,9,10,11,15,20]}
tree_clf = DecisionTreeClassifier(random_state=42)
grid_search = GridSearchCV(tree_clf,parameters, cv=3)
grid_search.fit(x_moons_train,y_moons_train)

print(grid_search.best_params_)
print(grid_search.best_score_)
# {'max_depth': 6, 'min_samples_leaf': 10}

# Add parameter max_leaf_nodes
# with depth is 6, there are max 2^5 = 32 leaves
parameters = {'max_depth':[2,3,4,5,6,8,10],'min_samples_leaf':[5,9,10,11,15,20], 'max_leaf_nodes': [10,16,32,64]}
tree_clf = DecisionTreeClassifier(random_state=42)
grid_search = GridSearchCV(tree_clf,parameters, cv=3)
grid_search.fit(x_moons_train,y_moons_train)

print(grid_search.best_params_)
print(grid_search.best_score_)
# {'max_depth': 8, 'max_leaf_nodes': 32, 'min_samples_leaf': 10}
'''

tree_clf1 = DecisionTreeClassifier(random_state=42, max_depth=8, max_leaf_nodes=32, min_samples_leaf=10)
tree_clf1.fit(x_moons_train,y_moons_train)
final_score = tree_clf1.score(x_moons_test,y_moons_test)
print(final_score)
# 0.871

'''
# print the decision tree
export_graphviz(
    tree_clf1,
    out_file='moons_tree.dot',
    feature_names=['x1','x2'],
    class_names=['0','1'],
    rounded=True,
    filled=True
)

graph = Source.from_file('moons_tree.dot')
graph.render('moons_tree', format='png', view=True)
'''

# plot the decision boundaries 

def plot_decision_boundary(clf, X, y):
    x1_min, x1_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    x2_min, x2_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5

    xx1, xx2 = np.meshgrid(
        np.linspace(x1_min, x1_max, 500),
        np.linspace(x2_min, x2_max, 500)
    )

    X_new = np.c_[xx1.ravel(), xx2.ravel()]
    y_pred = clf.predict(X_new).reshape(xx1.shape)


    plt.scatter(X[:, 0], X[:, 1], c=y, s=10)
    plt.contourf(xx1, xx2, y_pred, alpha=0.3)
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.savefig("decision_boundary.png", dpi=300, bbox_inches="tight")
    plt.show()

plot_decision_boundary(tree_clf1, x_moons_train, y_moons_train)



import matplotlib.pyplot as plt
from sklearn.inspection import DecisionBoundaryDisplay

# 1. Load sample data (using the first two features for 2D visualization)

X = x_moons_train 
y = y_moons_train


# 3. Plot decision boundaries
fig, ax = plt.subplots(figsize=(8, 6))

# 4. Overlay the actual data points
scatter = ax.scatter(
    X[:, 0], 
    X[:, 1], 
    c=y, 
    cmap=plt.cm.RdYlBu, 
    edgecolor="k"
)
# DecisionBoundaryDisplay automatically evaluates the meshgrid for the estimator
disp = DecisionBoundaryDisplay.from_estimator(
    tree_clf1,
    X,
    response_method="predict",
    cmap=plt.cm.RdYlBu,
    alpha=0.8,
    ax=ax,
    xlabel='x1',
    ylabel='x2',
)


plt.title("Decision Tree Classifier Decision Boundaries")
plt.savefig("decision_boundary2.png", dpi=300, bbox_inches="tight")
plt.show()
