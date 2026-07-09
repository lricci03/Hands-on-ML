import os
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from time import process_time
from sklearn.decomposition import PCA
from sklearn.linear_model import SGDClassifier



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
Creating a training set, validation set and a test set
splitting in 60'000 (train) and 10'000 (test)
X_train and X_test are numpy.ndarray of dimensions (60000,784) and (10000,784), respectively 
y_train and y_test are numpy.ndarray of dimensions (60000,) and (10000,) respectively
'''
X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]


# Train a Random Forest Classifier on the total dimensional MNIST set

rdf_clf = RandomForestClassifier(random_state=42, n_jobs=-1,n_estimators = 200)
n=10000

t = process_time()
rdf_clf.fit(X_train[:n],y_train[:n])
print(f'Time to train Random Forest Clf with {n} instances is: {process_time()-t}')
y_pred_rdf = rdf_clf.predict(X_test)
print(f'Accuracy of Random Forest trained on {n} instances is: {accuracy_score(y_test,y_pred_rdf)}')

# Time to train Random Forest Clf with 10000 instances is: 11.773963
# Accuracy of Random Forest trained on 10000 instances is: 0.9525


# Reduce the dimension of MNIST with PCA preserving 95% variance

pca = PCA(n_components=0.95)

t1 = process_time()
X_reduced = pca.fit_transform(X_train[:n])
print(f'Time to train and transform {n} instances with PCA: {process_time()-t1}')
print(f'The number of components is: {pca.n_components_}')

# Time to train and transform 10000 instances with PCA: 0.6768400000000003
# The number of components is: 150

# Train a Random Forest Classifier on the reduced MNIST set

t2 = process_time()
rdf_clf.fit(X_reduced[:n],y_train[:n])
print(f'Time to train Random Forest Clf with {n} instances on the reduced set is: {process_time()-t2}')
X_test_reduced = pca.transform(X_test)
y_pred_reduced_rdf = rdf_clf.predict(X_test_reduced)
print(f'Accuracy of Random Forest trained on {n} instances on the reduced set is: {accuracy_score(y_test,y_pred_reduced_rdf)}')

# Time to train Random Forest Clf with 10000 instances on the reduced set is: 26.830107
# Accuracy of Random Forest trained on 10000 instances on the reduced set is: 0.9182

# Train a SGD Classifier on the total dimensional MNIST set

sgd_clf = SGDClassifier(n_jobs=-1)
n=10000

t3 = process_time()
sgd_clf.fit(X_train[:n],y_train[:n])
print(f'Time to train SGD Clf with {n} instances is: {process_time()-t3}')
y_pred_sgd = sgd_clf.predict(X_test)
print(f'Accuracy of SGD trained on {n} instances is: {accuracy_score(y_test,y_pred_sgd)}')

# Time to train SGD Clf with 10000 instances is: 12.710327999999997
# Accuracy of SGD trained on 10000 instances is: 0.8818


# Train a SGD Classifier on the reduced MNIST set

t4 = process_time()
sgd_clf.fit(X_reduced[:n],y_train[:n])
print(f'Time to train SGD Clf with {n} instances on the reduced set is: {process_time()-t4}')
# X_test_reduced = pca.transform(X_test)
y_pred_reduced_sgd = sgd_clf.predict(X_test_reduced)
print(f'Accuracy of SGD trained on {n} instances on the reduced set is: {accuracy_score(y_test,y_pred_reduced_sgd)}')

# Time to train SGD Clf with 10000 instances on the reduced set is: 4.2578359999999975
# Accuracy of SGD trained on 10000 instances on the reduced set is: 0.8084

