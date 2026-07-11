import os
import numpy as np
from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt

# from c8_ex10 import plot_cluster_images


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


# step 1. reduce data set dimensionality using PCA keeping 99% of variance
pca = PCA(n_components=0.99)

X_train_reduced = pca.fit_transform(X_train)
X_val_reduced = pca.transform(X_val)
X_test_reduced = pca.transform(X_test)

print(f'The numer of components after PCA is: {pca.n_components_}')

# The numer of components after PCA is: 177

# Use Gaussian Mixture to cluster the faces
# Determine the optimal number of clusters using BIC parameter


'''BICs = {}

for k in range(1,50):
    gm = GaussianMixture(n_components=k, n_init=1, random_state=42, covariance_type='diag')
    gm.fit_predict(X_train_reduced)
    BICs[k] = gm.bic(X_train_reduced)

plt.plot(range(1,50), BICs.values(), 'o-')
plt.xlabel('number k of clusters')
plt.ylabel('BIC(k)')
plt.title('BIC for different number of clusters')
plt.savefig('c8_ex12_bic.png')
plt.show()
'''

# The BIC is a straight line with positive slope

# Try the AIC
'''
AICs = {}

for k in range(1,50):
    gm = GaussianMixture(n_components=k, n_init=1, random_state=42, covariance_type='diag')
    gm.fit_predict(X_train_reduced)
    AICs[k] = gm.aic(X_train_reduced)

plt.plot(range(1,50), AICs.values(), 'o-')
plt.xlabel('number k of clusters')
plt.ylabel('AIC(k)')
plt.title('AIC for different number of clusters')
plt.savefig('c8_ex12_aic.png')
plt.show()
'''
# lowest is for k=2, then there is also a drop for k=42,43, 49.

# Let's visualize the clusters with k=49
# To plot the clusters we use the function plot_cluster_images from c8_ex10

gm = GaussianMixture(n_components=49, n_init=1, random_state=42, covariance_type='diag')
labels = gm.fit_predict(X_train_reduced)
# plot_cluster_images(X_train_reduced, labels, target_cluster=0) 


# we generate new faces

X_new_reduced, y_new = gm.sample(n_samples=5)
X_new = pca.inverse_transform(X_new_reduced)

# plot the new faces
'''
# 1. Create a figure with 1 row and 5 columns
# figsize=(6, 3) sets the width to 6 inches and height to 3 inches
fig, axes = plt.subplots(1, 5, figsize=(6, 3))

# 2. Plot the first image on the first axis (axes[0])
for i in range(5):
    img = X_new[i].reshape(64, 64)
    axes[i].imshow(img, cmap="gray")

# 4. Display the side-by-side plot
plt.tight_layout()  # Automatically adjusts spacing so titles don't overlap
plt.show()
'''

# Can we find anomalies?
# Compare the score_samples() for normal images and anomalies
gm.score_samples(X_new_reduced)
gm.score_samples(X_train_reduced)
