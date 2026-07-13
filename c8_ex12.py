import os
import numpy as np
from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.metrics import f1_score, precision_score, recall_score

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
# 1. generate anomalies on the validation set by rotation, flipping and darkening
# Note: Never evalate on the train set. Thus we generate anomalies from the val set and evaluate on those
# 2. Add anomalies to validation set 
# 3. Find percentile on the training set, that performs best in detecting anomalies in the modified validation set 

# 1. generate anomalies on the validation set by rotation, flipping and darkening

# how to rotate one single image and plot it
'''
X_val_img = Image.fromarray(X_val[0].reshape(64,64)) # convert 64x64 pixels into image
X_val_rot_90_img = X_val_img.rotate(90)
X_val_rot_90 = np.array(X_val_rot_90_img) # convert back to 64x64 array

plt.imshow(X_val_rot_90, cmap='gray')
plt.show()
'''
# rotation
X_val_rot = []
for i in range(len(X_val)):
    im = X_val[i]
    im_img = Image.fromarray(im.reshape(64,64))
    im_img_rot = im_img.rotate(90)
    im_rot = np.array(im_img_rot).reshape(4096,)
    X_val_rot.append(im_rot)

# flip along central vertical axis
X_val_flip = []
for i in range(len(X_val)):
    im = X_val[i]
    im_flip = np.flip(im.reshape(64,64), axis=1)
    X_val_flip.append(im_flip.reshape(4096,))

# darken (reduce brightness by 40%)
X_val_dark = []
for i in range(len(X_val)):
    im = X_val[i]
    im_dark = im * 0.6
    X_val_dark.append(im_dark)

# 2. Add anomalies to validation set
# dimension is reduced with PCA

X_val_anomalies = np.concatenate([X_val_rot,X_val_flip,X_val_dark])
X_val_anomalies_reduced = pca.transform(X_val_anomalies)
X_val_new_reduced = np.concatenate([X_val_reduced, X_val_anomalies_reduced])

# Label original instances with 0, and anomalies with 1

y_val_true = np.zeros(len(X_val))
y_val_anomaly_true = np.ones(len(X_val_anomalies))

y_val_new = np.concatenate([y_val_true,y_val_anomaly_true])


# 3. Find percentile on the training set, that performs best in detecting anomalies in the modified validation set 

train_densities = gm.score_samples(X_train_reduced)

perc = [1,2,5,10,15,20] # possible percentiles
density_thresholds = []
f1_scores = []
precision_scores = []
recall_scores = []
for p in perc:
    density_threshold = np.percentile(train_densities,p)
    density_thresholds.append(round(float(density_threshold),3))
    new_val_densities = gm.score_samples(X_val_new_reduced)
    # detected_anomalies = X_val_new_reduced[new_val_densities < density_threshold] # extract elements of new validation set whose density is less than threshold
    y_val_pred = (new_val_densities < density_threshold).astype(int) # density below threshold iff 'True' and label is 1
    # we have two lists of labels 0 and 1's: y_val_new and y_val_pred. Each element correspond to one instance.
    # The labels say if the instance is an anomaly (1) or not (0)
    # Let (x,y) be the labels of the same instance in the two lists, then we have:
    # (0,0) : true negative. (0,1): false positive. (1,0): false negative, (1,1): true positive
    # we evaluate using the f1 score
    f1_scores.append(round(f1_score(y_val_new,y_val_pred),3))
    precision_scores.append(round(precision_score(y_val_new,y_val_pred),3))
    recall_scores.append(round(recall_score(y_val_new,y_val_pred),3))
print(f'The each pecentile the f1 score is: {list(zip(perc,f1_scores))}')
# [(1, 0.766), (2, 0.776), (5, 0.826), (10, 0.839), (15, 0.85), (20, 0.86)]
print(f'The each pecentile the recall score is: {list(zip(perc,recall_scores))}')
# [(1, 0.688), (2, 0.713), (5, 0.8), (10, 0.838), (15, 0.871), (20, 0.908)]
print(f'The each pecentile the precision score is: {list(zip(perc,precision_scores))}')
# [(1, 0.864), (2, 0.851), (5, 0.853), (10, 0.841), (15, 0.829), (20, 0.816)]
print(f'The each pecentile the density threshold is: {list(zip(perc,density_thresholds))}')
# [(1, -54.016), (2, -51.154), (5, -44.115), (10, -38.468), (15, -32.672), (20, -28.05)]
print(gm.score_samples(X_val_anomalies_reduced[:10]))
print(gm.score_samples(X_val_reduced[:10]))
print(gm.score_samples(X_train_reduced[:10]))


# precision score: true positive over total positive (avoid false alarm)
# recall score: true positive over true pos + false neg. which percentage of true is detected (catch true anomalies)

