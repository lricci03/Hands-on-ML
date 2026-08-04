import os
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, StackingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.base import clone


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
splitting in 50'000 (train), 10'000 (validation) and 10'000 (test)
X_train and X_val, X_test are numpy.ndarray of dimensions (50000,784) and (10000,784), respectively
y_train and y_val, y_test are numpy.ndarray of dimensions (50000,) and (10000,) respectively
'''
X_train, X_val, X_test, y_train, y_val, y_test = X[:50000], X[50000:60000], X[60000:], y[:50000], y[50000:60000], y[60000:]

rnd_clf = RandomForestClassifier(n_estimators = 500, n_jobs = -1, random_state= 42)
xtra_clf = ExtraTreesClassifier(n_estimators = 500, n_jobs= -1, random_state= 42)
svc_clf = SVC()
'''
# Train the blender classifier 
rnd_clf.fit(X_train[:10000],y_train[:10000])
xtra_clf.fit(X_train[:10000],y_train[:10000])
svc_clf.fit(X_train[:10000],y_train[:10000])

y_rnd_val = rnd_clf.predict(X_val)
y_xtra_val = xtra_clf.predict(X_val)
y_svc_val = svc_clf.predict(X_val)

X_train_blend = np.column_stack((y_rnd_val, y_xtra_val, y_svc_val))

svc_clf_blend = SVC()
svc_clf_blend.fit(X_train_blend,y_val)

# Predict the test set using the blender classifier

y_rnd_test = rnd_clf.predict(X_test)
y_xtra_test = xtra_clf.predict(X_test)
y_svc_test = svc_clf.predict(X_test)
X_test_blend = np.column_stack((y_rnd_test, y_xtra_test, y_svc_test))
y_test_blend = svc_clf_blend.predict(X_test_blend)
print(f'The blender score is: {accuracy_score(y_test,y_test_blend):.4f}')

# The blender score is:  0.9500
'''
# Using a stacking classifier

stacking_clf = StackingClassifier(
    estimators=[
        ('rnd', rnd_clf),
        ('xtra', xtra_clf),
        ('svc', svc_clf)
    ],
    final_estimator=SVC(), 
    cv = 5
)

stacking_clf.fit(X_val,y_val)
y_stacking_pred = stacking_clf.predict(X_test)
print(f'The stacking clf score is: {accuracy_score(y_test,y_stacking_pred):.4f}')

# The stacking clf score is: 0.9592
