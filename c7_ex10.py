import os
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.manifold import TSNE, LocallyLinearEmbedding, MDS
import matplotlib.pyplot as plt
from matplotlib import colormaps
from sklearn.decomposition import PCA


'''
    Fetching the data
'''
if os.path.exists("mnist_data.npz"):
    # Load from file
    loaded = np.load("mnist_data.npz", allow_pickle=True)
    X, y = loaded["X"], loaded["y"]
else:
    # Download
    mnist = fetch_openml('mnist_784', as_frame=False)
    X, y = mnist.data, mnist.target
    # Save for next time
    np.savez("mnist_data.npz", X=X, y=y)

'''
For this exercise we want to visualize only the first 5'000 images of the MNIST data set
'''
X_5, y_5 = X[:5000], y[:5000]

# Using TSNE

'''tsne = TSNE(n_components=2)

X_reduced = tsne.fit_transform(X_5)

scatter = plt.scatter(X_reduced[:,0], X_reduced[:,1], c=y_5.astype(int))
plt.xlabel('$z_1$')
plt.ylabel('$z_2$')
plt.legend(*scatter.legend_elements())
plt.title('MNIST visualization with TSNE')
plt.savefig("c7_ex10_tsne.png", dpi=300, bbox_inches="tight")
plt.show()
'''

# USING PCA
'''
pca = PCA(n_components=2)

X_reduced = pca.fit_transform(X_5)

scatter = plt.scatter(X_reduced[:,0], X_reduced[:,1], c=y_5.astype(int))
plt.xlabel('$z_1$')
plt.ylabel('$z_2$')
plt.legend(*scatter.legend_elements())
plt.title('MNIST visualization with PCA')
plt.savefig("c7_ex10_pca.png", dpi=300, bbox_inches="tight")
plt.show()
'''
# USING LLE
'''
lle = LocallyLinearEmbedding(n_components=2)

X_reduced = lle.fit_transform(X_5)

scatter = plt.scatter(X_reduced[:,0], X_reduced[:,1], c=y_5.astype(int))
plt.xlabel('$z_1$')
plt.ylabel('$z_2$')
plt.legend(*scatter.legend_elements())
plt.title('MNIST visualization with LLE')
plt.savefig("c7_ex10_lle.png", dpi=300, bbox_inches="tight")
plt.show()
'''
# Using MDS
'''
mds = MDS(n_components=2)

X_reduced = mds.fit_transform(X_5)

scatter = plt.scatter(X_reduced[:,0], X_reduced[:,1], c=y_5.astype(int))
plt.xlabel('$z_1$')
plt.ylabel('$z_2$')
plt.legend(*scatter.legend_elements())
plt.title('MNIST visualization with MDS')
plt.savefig("c7_ex10_mds.png", dpi=300, bbox_inches="tight")
plt.show()'''



# Plotting the images of the digits, plotting an image only if no other has already been plotted at a close distance
# We do it for TSNE
# The code is taken from the online solutions of the book

from sklearn.preprocessing import MinMaxScaler
from matplotlib.offsetbox import AnnotationBbox, OffsetImage

def plot_digits(X, y, min_distance=0.04, images=None, figsize=(13, 10)):
    # Let's scale the input features so that they range from 0 to 1
    X_normalized = MinMaxScaler().fit_transform(X)
    # Now we create the list of coordinates of the digits plotted so far.
    # We pretend that one is already plotted far away at the start, to
    # avoid `if` statements in the loop below
    neighbors = np.array([[10., 10.]])
    # The rest should be self-explanatory
    plt.figure(figsize=figsize)
    cmap = plt.cm.jet
    digits = np.unique(y)
    for digit in digits:
        plt.scatter(X_normalized[y == digit, 0], X_normalized[y == digit, 1],
                    c=[cmap(float(digit) / 9)], alpha=0.5)
    plt.axis("off")
    ax = plt.gca()  # get current axes
    for index, image_coord in enumerate(X_normalized):
        closest_distance = np.linalg.norm(neighbors - image_coord, axis=1).min()
        if closest_distance > min_distance:
            neighbors = np.r_[neighbors, [image_coord]]
            if images is None:
                plt.text(image_coord[0], image_coord[1], str(int(y[index])),
                         color=cmap(float(y[index]) / 9),
                         fontdict={"weight": "bold", "size": 16})
            else:
                image = images[index].reshape(28, 28)
                imagebox = AnnotationBbox(OffsetImage(image, cmap="binary"),
                                          image_coord)
                ax.add_artist(imagebox)
    plt.savefig('c7_ex10_tsne_digits.png')
    plt.show()


tsne = TSNE(n_components=2)

X_reduced = tsne.fit_transform(X_5)
plot_digits(X_reduced, y_5, images=X_5, figsize=(35, 25))