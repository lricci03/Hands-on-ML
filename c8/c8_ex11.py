import os
import numpy as np
from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline, FeatureUnion
from sklearn.preprocessing import FunctionTransformer

'''
    Fetching the data
'''
if os.path.exists("olivetti_data.npz"):
    # Load from file
    olivetti = np.load("olivetti_data.npz", allow_pickle=True)
    X, y = olivetti["X"], olivetti["y"]
else:
    # Download
    olivetti = fetch_olivetti_faces()
    X, y = olivetti.data, olivetti.target
    # Save for next time
    np.savez("olivetti_data.npz", X=X, y=y)

# Stratified sampling
# into train, validation and test
# first do 60% train, 40% val+test, 
# then divide the 40% into 50% validation and 50% test

X_train, X_val_test_set, y_train, y_val_test_set = train_test_split(X,y, test_size=0.4, stratify=y, random_state=42)

X_val, X_test, y_val, y_test = train_test_split(X_val_test_set,y_val_test_set, test_size=0.5, stratify=y_val_test_set, random_state=42)

'''
# Train a random forest classifier on the train test

rnd_clf = RandomForestClassifier(n_jobs=-1, random_state=42)
rnd_clf.fit(X_train,y_train)
y_val_pred = rnd_clf.predict(X_val)
print(f'The accuracy score of random forest clf is: {accuracy_score(y_val, y_val_pred)}')
'''
# The accuracy score of random forest clf is: 0.925


# Use k-means as a dimensionality reduction tool
# cluster the images into k clusters, and substitute each with the centroid of its cluster
# First experimenting and visualizing what happens:
# with k small the faces of a cluster merge into one
'''
kmeans = KMeans(n_clusters=20, random_state=42).fit(X_train)
segmented_img = kmeans.cluster_centers_[kmeans.labels_]

# Printing the i-th image of the train set side-by-side with its segmented version
# Just for fun/cool how it changes as the number of cluster changes
# with k = 20 the person completely changes

i = 10

# 1. Create a figure with 1 row and 2 columns
# figsize=(6, 3) sets the width to 6 inches and height to 3 inches
fig, axes = plt.subplots(1, 2, figsize=(6, 3))

# 2. Plot the first image on the first axis (axes[0])
img_0 = X_train[i].reshape(64, 64)
axes[0].imshow(img_0, cmap="gray")
axes[0].set_title("Original image")
# axes[0].axis("off")  # Hide the pixel grid lines

# 3. Plot the second image on the second axis (axes[1])
img_1 = segmented_img[i].reshape(64, 64)
axes[1].imshow(img_1, cmap="gray")
axes[1].set_title("Segmented")
# axes[1].axis("off")  # Hide the pixel grid lines

# 4. Display the side-by-side plot
plt.tight_layout()  # Automatically adjusts spacing so titles don't overlap
plt.show()

'''


# Train the Random Forest Classifier on the reduced set,
# searching for the number k of clusters that allows it to get the best performance

'''
k=100

kmeans = KMeans(n_clusters=k, random_state=42)
kmeans.fit(X_train)
X_train_segmented = kmeans.cluster_centers_[kmeans.labels_]
X_val_clusters = kmeans.predict(X_val) # clusters of the instances in X_val
X_val_segmented = kmeans.cluster_centers_[X_val_clusters]


rnd_clf = RandomForestClassifier(random_state=42)
rnd_clf.fit(X_train_segmented,y_train)
y_val_segmented_pred = rnd_clf.predict(X_val_segmented)

print(f'The score of Random Forest clf with {k} clusters is: {accuracy_score(y_val, y_val_segmented_pred)}')
'''
# The score of Random Forest clf with 100 clusters is: 0.7875


# Use kmeans to reduce dimension: use the distances from the centroids as features 
# these distances are soft scores 
'''
k=200

kmeans = KMeans(n_clusters = k, random_state=42)
kmeans.fit(X_train)
X_train_distances = kmeans.transform(X_train) 
X_val_distances = kmeans.transform(X_val)

rnd_clf = RandomForestClassifier(random_state=42)
rnd_clf.fit(X_train_distances,y_train)
y_val_distances_pred = rnd_clf.predict(X_val_distances)

print(f'The score of Random Forest clf with {k} clusters over the centroid distances is: {accuracy_score(y_val, y_val_distances_pred)}')
'''
# The score of Random Forest clf with 200 clusters over the centroid distances is: 0.825


# Use cross validation to find the best number of clusters
# Remove the validation test

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, stratify=y, random_state=42)

'''
kmeans_rnd_clf = make_pipeline(KMeans(random_state=42),RandomForestClassifier(random_state=42))


# We have 320 instances in the train set and with 3 cross validations (cv=3) the train set of gridsearch contains 2/3*320=212 instances
# so n_clusters must be <= 212
param_grid = [
    {'kmeans__n_clusters':[100,150,200,212]}
]


grid_search = GridSearchCV(kmeans_rnd_clf, param_grid, cv=3)
grid_search.fit(X_train,y_train)

print(grid_search.best_params_)
print(grid_search.best_score_)

# {'kmeans__n_clusters': 150}
# 0.7968318344795157

y_test_pred = grid_search.predict(X_test)
print(f'The accuracy score on the test set is: {accuracy_score(y_test,y_test_pred)}')
# The accuracy score on the test set is: 0.8125
'''

# We want to add the distances from the clusters to the original features and try with that
# We create a custom transformer that stacks the data

stack_transformer = FeatureUnion([
    ('original', 'passthrough'),
    ('distances', KMeans(random_state=42))
])

stacked_pipeline = make_pipeline(stack_transformer,RandomForestClassifier(random_state=42))

param_grid = [
    {'featureunion__distances__n_clusters':[100,150,200,212]}
]

grid_search = GridSearchCV(stacked_pipeline, param_grid, cv=3)
grid_search.fit(X_train,y_train)

print(f'Best parameters are: {grid_search.best_params_}')
print(f'The corresponding score on the train set is: {grid_search.best_score_}')

y_test_pred = grid_search.predict(X_test)
print(f'The score on the test set is: {accuracy_score(y_test,y_test_pred)}')

# Best parameters are: {'featureunion__distances__n_clusters': 212}
# The corresponding score on the train set is: 0.9155351789807794
# The score on the test set is: 0.9375
