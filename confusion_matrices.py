import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDClassifier
from sklearn.datasets import fetch_openml
import joblib
from sklearn.metrics import precision_score, recall_score, confusion_matrix, accuracy_score, ConfusionMatrixDisplay

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

X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]

y_train_pred = joblib.load("y_train_pred.pkl")

# Confusion Matrix
ConfusionMatrixDisplay.from_predictions(y_train,y_train_pred)
plt.savefig(f"CM.png")
# Confusion Matrix normalized on the rows: (i,j)-entry shows which percentage of i is predicted as j
ConfusionMatrixDisplay.from_predictions(y_train,y_train_pred,normalize="true",values_format=".0%")
plt.savefig(f"CM_norm_true.png")
# Confusion Matrix normalized on the rows removing correct predictions:
# (i,j)-entry shows the percentage of misclassified i are predicted as j
sample_weight = (y_train != y_train_pred) #give weight 0 to correct predictions
ConfusionMatrixDisplay.from_predictions(y_train,y_train_pred,sample_weight = sample_weight,normalize="true",values_format=".0%")
plt.savefig(f"CM_err_norm_true.png")
# Confusion Matrix normalized on the columns removing correct preditions:
# (i,j)-entry shows what percentage of instances predicted as j are actually i
ConfusionMatrixDisplay.from_predictions(y_train,y_train_pred,sample_weight = sample_weight,normalize="pred",values_format=".0%")
plt.savefig(f"CM_err_norm_pred.png")
