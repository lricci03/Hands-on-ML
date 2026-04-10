# Chapter 3, exercise 2.
# Write a function that can shift an MNIST image in any direction
# (left, right, up, or down) by one pixel.⁠
# Then, for each image in the training set, create four shifted copies (one per direction)
# and add them to the training set.
# Finally, train your best model on this expanded training set and measure its accuracy on the test set.
# You should observe that your model performs even better now!
# This technique of artificially growing the training set is called data augmentation or training set expansion.

import numpy as np #pip install numpy
import os
import matplotlib.pyplot as plt #pip install matplotlib
# pip install scikit-learn
from sklearn.neighbors import KNeighborsClassifier, NeighborhoodComponentsAnalysis
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline
import joblib
from sklearn.metrics import precision_score, recall_score, confusion_matrix, accuracy_score
import itertools
from itertools import chain
from scipy.ndimage import shift

def main():
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


    # split into train set and test set
    X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]
     
    # The following terminates with n=10 (the full training set is n=12)
    if not (os.path.exists("X_train_augm_250k.pkl") and os.path.exists("X_train_augm_250k.pkl")):
        X_train_augm_dic={}
        y_train_augm_dic={}
        X_train_sliced={}
        y_train_sliced={}
        for n in range(10):
            if not os.path.exists(f"X_train_augm_{n}.pkl"):
                create_shifts(n)
            X_train_augm_dic[n]=joblib.load(f"X_train_augm_{n}.pkl")
            y_train_augm_dic[n]=joblib.load(f"y_train_augm_{n}.pkl")
            X_train_sliced[n]=X_train[(5000*n):(5000*(n+1))]
            y_train_sliced[n]=y_train[(5000*n):(5000*(n+1))]
        for n in range(10):
            if n==0:
                X_train_augm = np.concatenate([X_train_sliced[n],X_train_augm_dic[n]])
                y_train_augm = np.concatenate([y_train_sliced[n],y_train_augm_dic[n]])
            else:
                X_train_augm = np.concatenate([X_train_augm,X_train_sliced[n],X_train_augm_dic[n]])
                y_train_augm = np.concatenate([y_train_augm,y_train_sliced[n],y_train_augm_dic[n]])
        # Save the augmented training data. 9 <-> first 10*5000 training data augmented.
        # Tot training data is 10*25'000 = 250'000   
        joblib.dump(X_train_augm,"X_train_augm_250k.pkl")
        joblib.dump(y_train_augm,"y_train_augm_250k.pkl")
    
    else:
        X_train_augm=joblib.load("X_train_augm_250k.pkl")
        y_train_augm=joblib.load("y_train_augm_250k.pkl")
    knn=KNeighborsClassifier(n_neighbors=4,weights='distance')
    knn.fit(X_train_augm,y_train_augm)
    print(knn.score(X_test,y_test))
    # Output: Training on the augmented first 5000 data (so 25.000 data) scores 0.9495
    # Output: Training on the augmented first 10000 data (so 50.000 data) scores 0.9607
    # Output: Training on the augmented first 50000 data (so 250.000 data) scores 0.9752
    

def plot_digit(image_data,name):
    # Reshape the 784 vector into a 28x28 array
    image=image_data.reshape(28,28)
    #cmap="binary" is a grayscale color map from 0 to 255
    plt.imshow(image,cmap="binary")
    plt.axis("off")
    plt.savefig(name)

# A MNIST file is a array. We first reshape it in a 28x28 image and then shift it by v vertically, by h horizontally
# v >0 shift down, v<0 shift up
# h >0 shift right, h<0 shift left
def shift_array(array,v,h):
    image=array.reshape(28,28)
    return shift(image,(v,h))

def image_to_array(image):
    return image.reshape(784,)

#Create a list shifted which contains the four shifts of each data x in X_train
# We create the shifted version for x in X[5000*k:5000(k+1)]. Max k=12 (60,000)
def create_shifts(k):
    shifted=[]
    start=5000*k
    end=5000*(k+1)
    for x in X_train[start:end]:
        x_u = image_to_array(shift_array(x,-1,0))
        x_d = image_to_array(shift_array(x,1,0))
        x_r = image_to_array(shift_array(x,0,1))
        x_l = image_to_array(shift_array(x,0,-1))
        shifted.append(x_u)
        shifted.append(x_d)
        shifted.append(x_r)
        shifted.append(x_l)
    #Turn this list into an array
    X_train_shifted = np.array(shifted)
    #Create a list shifted_labels which contains the labels of the shifted elements
    shifted_labels=[]
    for y in y_train[start:end]:
        shifted_labels.append(y)
        shifted_labels.append(y)
        shifted_labels.append(y)
        shifted_labels.append(y)
    #Turn this list into an array
    y_train_shifted = np.array(shifted_labels)
    joblib.dump(X_train_shifted,f"X_train_augm_{k}.pkl")
    joblib.dump(y_train_shifted,f"y_train_augm_{k}.pkl")

if __name__ == '__main__':
    main()