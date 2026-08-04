import os
import numpy as np
from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

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


# Elbow visualization to find optimal number k of clusters 
# I thought that since there are 40ppl we should restrict to max 40 clusters
# In the solutions they say that since the same person can look different in pictures (w/ or w/out glasses, profile,..) one should use more clusters
'''
inertias = {}
# plotting inertia for k = 1, ..., 40
for k in range(1,41):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit_predict(X_train)
    inertias[k] = kmeans.inertia_
plt.plot(range(1,41), inertias.values(), 'o-')
plt.xlabel('number of clusters k')
plt.ylabel('inertia')
plt.title('intertia of olivetti faces')
plt.xticks(list(range(1,41)), rotation=45) # integer ticks on x-axis, labels are rotated
plt.show()
plt.savefig('c8_ex10_intertia.png')

# The elbow might be at k=5 or k=20

# Silhouette score visualization to find optimal number k of clusters 
# k

silhouettes = {}
for k in range(2,41):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit_predict(X_train)
    silhouettes[k] = silhouette_score(X_train, kmeans.labels_)
plt.plot(range(2,41), silhouettes.values(), 'o-')
plt.xlabel('number of clusters k')
plt.ylabel('silhouette score')
plt.title('silhouette score of olivetti faces')
plt.xticks(list(range(2,41)), rotation=45) # integer ticks on x-axis, labels are rotated
plt.show()
plt.savefig('c8_ex10_silhouette_score.png')

# peak at 35 and 40
'''

# We choose k clusters and plot the images in each cluster to see if they are similar to each other
# The code is AI generated


def plot_cluster_images(X_data, cluster_labels, target_cluster, max_images=10):
    # Find the indices of all images assigned to the target_cluster
    indices = np.where(cluster_labels == target_cluster)[0]
    
    # Cap the number of images to plot based on what is available
    n_images = min(len(indices), max_images)
    
    if n_images == 0:
        print(f"Cluster {target_cluster} is empty!")
        return

    # Set up a grid row for the images
    fig, axes = plt.subplots(1, n_images, figsize=(n_images * 1.5, 2))
    fig.suptitle(f"Faces in Cluster {target_cluster}", fontsize=14, y=1.15)
    
    # If there's only 1 image, axes won't be an array, so we wrap it
    if n_images == 1:
        axes = [axes]

    for i, idx in enumerate(indices[:n_images]):
        # Reshape the flat 4096 array back to a 64x64 square image
        img = X_data[idx].reshape(64, 64)
        
        axes[i].imshow(img, cmap="gray")
        axes[i].axis("off")
        axes[i].set_title(f"Img #{idx}", fontsize=8)

    plt.show()

# View the contents of Cluster 0 and Cluster 1

k=20

kmeans = KMeans(n_clusters=k, random_state=42)
kmeans.fit_predict(X_train)
labels = kmeans.labels_
plot_cluster_images(X_train, labels, target_cluster=0)
plot_cluster_images(X_train, labels, target_cluster=1)
