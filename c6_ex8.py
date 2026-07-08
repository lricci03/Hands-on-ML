import os
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, VotingClassifier
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


''' Random forest classifier, Extra-trees classifier, SVM classifier'''
'''
rnd_clf = RandomForestClassifier(n_jobs = -1, random_state= 42)
param_grid= {'n_estimators':[100,200,500],'max_leaf_nodes': [8,16,32]}
grid_search=GridSearchCV(estimator=rnd_clf, param_grid=param_grid,cv=3,scoring="accuracy")
grid_search.fit(X_train,y_train)

print(f"The best parameters are: {grid_search.best_params_}")
print(f"The best accuracy score is: {grid_search.best_score_}")
'''
# The best parameters are: {'max_leaf_nodes': 32, 'n_estimators': 500}
# The best accuracy score is: 0.866440121095574

rnd_clf = RandomForestClassifier(n_estimators = 500, max_leaf_nodes= 32, n_jobs = -1, random_state= 42)

'''
xtra_clf = ExtraTreesClassifier(n_jobs= -1, random_state= 42)
param_grid= {'n_estimators':[100,200,500],'max_leaf_nodes': [8,16,32]}
grid_search=GridSearchCV(estimator=xtra_clf, param_grid=param_grid,cv=3,scoring="accuracy")
grid_search.fit(X_train,y_train)

print(f"The best parameters are: {grid_search.best_params_}")
print(f"The best accuracy score is: {grid_search.best_score_}")
'''
# The best parameters are: {'max_leaf_nodes': 32, 'n_estimators': 500}
# The best accuracy score is: 0.8537001130852216

xtra_clf = ExtraTreesClassifier(n_estimators = 500, max_leaf_nodes= 32, n_jobs= -1, random_state= 42)

svc_clf = SVC()
'''
rnd_clf.fit(X_train[:10000],y_train[:10000])
xtra_clf.fit(X_train[:10000],y_train[:10000])
svc_clf.fit(X_train[:10000],y_train[:10000])

y_rnd_val = rnd_clf.predict(X_val)
y_xtra_val = xtra_clf.predict(X_val)
y_svc_val = svc_clf.predict(X_val)

print(f"The accuracy score for Random Forest is {accuracy_score(y_val, y_rnd_val)}")
print(f"The accuracy score for Extra Trees is {accuracy_score(y_val, y_xtra_val)}")
print(f"The accuracy score for SVC is {accuracy_score(y_val, y_svc_val)}")
'''
# The accuracy score for Random Forest is 0.8777
# The accuracy score for Extra Trees is 0.859
# The accuracy score for SVC is 0.966

'''
hard_voting_clf = VotingClassifier(
    estimators=[
        ('rnd', rnd_clf),
        ('xtra', xtra_clf),
        ('svc', svc_clf)
    ]
)

soft_voting_clf = clone(hard_voting_clf)
soft_voting_clf.voting = 'soft'

soft_voting_clf.set_params(svc__probability=True)

# VotingClassifier outputs prediction as integers, so to compute scores we need to change y_val, y_test to int

y_val = y_val.astype(int)
y_train = y_train.astype(int)

hard_voting_clf.fit(X_train[:10000],y_train[:10000])
soft_voting_clf.fit(X_train[:10000],y_train[:10000])

for name, clf in hard_voting_clf.named_estimators_.items():
    print(name, 'score: ', clf.score(X_val,y_val))

print('hard voting clf score: ', hard_voting_clf.score(X_val,y_val))
print('soft voting clf score: ', soft_voting_clf.score(X_val,y_val))
'''
# rnd score:  0.8777
# xtra score:  0.859
# svc score:  0.966
# hard voting clf score:  0.8911
# soft voting clf score:  0.9599


# Try again removing the constrain on the max number of leaves

rnd_clf = RandomForestClassifier(n_estimators = 500, n_jobs = -1, random_state= 42)
xtra_clf = ExtraTreesClassifier(n_estimators = 500, n_jobs= -1, random_state= 42)

hard_voting_clf = VotingClassifier(
    estimators=[
        ('rnd', rnd_clf),
        ('xtra', xtra_clf),
        ('svc', svc_clf)
    ]
)

soft_voting_clf = clone(hard_voting_clf)
soft_voting_clf.voting = 'soft'

soft_voting_clf.set_params(svc__probability=True)

# VotingClassifier outputs prediction as integers, so to compute scores we need to change y_val, y_test to int

y_val = y_val.astype(int)
y_train = y_train.astype(int)

hard_voting_clf.fit(X_train[:10000],y_train[:10000])
soft_voting_clf.fit(X_train[:10000],y_train[:10000])

for name, clf in hard_voting_clf.named_estimators_.items():
    print(name, 'score: ', clf.score(X_val,y_val))

print('hard voting clf score: ', hard_voting_clf.score(X_val,y_val))
print('soft voting clf score: ', soft_voting_clf.score(X_val,y_val))

# rnd score:  0.9563
# xtra score:  0.9617
# svc score:  0.966
# hard voting clf score:  0.9624
# soft voting clf score:  0.9663