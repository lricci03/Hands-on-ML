from sklearn.datasets import fetch_openml
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDClassifier
import joblib

'''
need to 
pip install pandas
pip install scikit-learn
pip install matplotlib
'''

'''
Code to download the MNIST data, only if they're not already downloaded
- X is a 2D array with 70'000 rows and 28x28= 784 columns.
    Each row corresponds to an image, which is given by 28x28 pixels (the columns)
    Each feature represents the pixel intensity with a number from 0 (white) to 255 (black).
- y is a 1D array with 70'000 rows, each containing the image description, as a string.

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
plot_digit() takes a 1D array of 784 features e.g. a row of X and plots it according to 
the corresponding pixel's intensities.
'''

def plot_digit(image_data):
    # Reshape the 784 vector into a 28x28 array
    image=image_data.reshape(28,28)
    #cmap="binary" is a grayscale color map from 0 to 255
    plt.imshow(image,cmap="binary")
    plt.axis("off")

'''
Visualize some of the images in the dataset.
We need to save them in a png.
'''
some_digit=X[0]
plot_digit(some_digit)
plt.savefig(f"0_digit({y[0]}).png")

'''
Creating a training set and a test set
splitting in 60'000 (train) and 10'000 (test)
'''
X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]

'''
Binary classifier
Distinguishing between two classes: 5 and non-5
'''

''' Each of y_train_5 and y_test_5 is a vector whose entries are True or False,
 depending if the corresponding entry in y_train, y_test resp., is '5' or not.
 '''
y_train_5 = (y_train =='5')
t_test_5 = (y_test == '5')


'''
Train the model
and saving the trained model using joblib, if not already saved
'''

if os.path.exists("sgd_model.pkl"):
    # Load from file
    sgd_clf = joblib.load("sgd_model.pkl")
else:
    sgd_clf = SGDClassifier(random_state=42)
    sgd_clf.fit(X_train,y_train_5)
    joblib.dump(sgd_clf,"sgd_model.pkl")

''' 
Need to wrap X_train[10] in [] to make it a 2D-array, as it is expected by sgd_cld.predict()
'''
print(sgd_clf.predict([X_train[10]]))
