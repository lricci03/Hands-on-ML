# Chapter 3, exercise 1.
# Try to build a classifier for the MNIST dataset that achieves over 97% accuracy on the test set.
#   Hint: the KNeighborsClassifier works quite well for this task; 
#   you just need to find good hyperparameter values (try a grid search on the weights and _neighbors hyperparameters).

import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier, NeighborhoodComponentsAnalysis
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline
import joblib
from sklearn.metrics import precision_score, recall_score, confusion_matrix, accuracy_score
import itertools
from itertools import chain



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

    '''
    Creating a training set and a test set
    splitting in 60'000 (train) and 10'000 (test)
    X_train and X_test are numpy.ndarray of dimensions (60000,784) and (10000,784), respectively
    y_train and y_test are numpy.ndarray of dimensions (60000,) and (10000,) respectively
    '''
    X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]

    '''
    We scale the data X. It is important to fit only the train, and transform both train and test.
    '''
    
    #scaler = StandardScaler()
    #X_train_scaled = scaler.fit_transform(X_train)
    #X_test_scaled = scaler.transform(X_test)

    '''
    Testing out different k for n_neighbros = k)
    We store the accuracy scores in a dictionary acc with keys k and values accuracy for k-nbhs
    '''
    #acc ={}
    #for k in range(2,20):
    #    knc = KNeighborsClassifier(n_neighbors=k)
    #    knc.fit(X_train_scaled[:2000], y_train[:2000])
    #    y_train_pred = knc.predict(X_train_scaled[:2000])
    #    acc[k]=accuracy_score(y_train[:2000],y_train_pred)

    '''
    We plot the k->acc(k) function
   
    plt.plot(range(2,20), acc.values())
    plt.xlabel('k')
    plt.ylabel('Accuracy')
    plt.savefig(f"accuracy.png")

    print(max(acc, key=acc.get), max(acc.values()))
    '''


    '''
    Cross validation scores - self-implemented version of search grid
    '''
    '''
    knn = KNeighborsClassifier()
    for k, weights in itertools.product(range(2,20),("uniform","distance")):
        knn.set_params(**{'n_neighbors':k,'weights':weights})
        acc=cross_val_score(knn,X_train_scaled[:2000],y_train[:2000],cv=4,scoring="accuracy")
        print(k,weights,min(acc))
    '''

    '''
    Plotting the minimal accuracy scores (across 3-fold cross-validation) for uniform and distance weights
    '''
    '''
    knn = KNeighborsClassifier()
    acc={}
    for k in range(2,20):
        knn.set_params(**{'n_neighbors':k,'weights':'uniform'})
        acc[k]=min(cross_val_score(knn,X_train_scaled[:2000],y_train[:2000],cv=3,scoring="accuracy"))
    plt.plot(range(2,20), acc.values())
    plt.xlabel('k')
    plt.ylabel('Accuracy')
    acc={}
    for k in range(2,20):
        knn.set_params(**{'n_neighbors':k,'weights':'distance'})
        acc[k]=min(cross_val_score(knn,X_train_scaled[:2000],y_train[:2000],cv=3,scoring="accuracy"))
    plt.plot(range(2,20), acc.values())
    plt.legend(['Unifom', 'Distance'])
    plt.savefig(f"c3_ex1_accuracy_uniform_distance.png")
    '''

    '''
    Implementing GridSearchCV
    '''
    '''
    knn = KNeighborsClassifier()
    param_grid= {'n_neighbors':range(2,20),'weights':['distance','uniform']}
    grid_search=GridSearchCV(estimator=knn, param_grid=param_grid,cv=3,scoring="accuracy")
    grid_search.fit(X_train_scaled,y_train)

    print(f"The best parameters are: {grid_search.best_params_}")
    print(f"The best accuracy score is: {grid_search.best_score_}")
    # Output is:
    # The best parameters are: {'n_neighbors': 4, 'weights': 'distance'}
    # The best accuracy score is: 0.9437666666666668
    '''

    '''
    We finally evaluate on the test set using n_neighbors=4, weights="distance"
    '''

    '''
    knn_clf = KNeighborsClassifier(n_neighbors=4, weights="distance")
    knn_clf.fit(X_train_scaled,y_train)
    y_test_pred = knn_clf.predict(X_test_scaled)
    print("The accuracy for n_neighbors=4 and weights=distance is:", accuracy_score(y_test,y_test_pred))
   # output: The accuracy for n_neighbors=4 and weights=disntace is: 0.9489
    '''

    '''
    Try again but with unscaled data
    '''
    '''
    knn = KNeighborsClassifier()
    param_grid= {'n_neighbors':range(1,10),'weights':['distance','uniform']}
    grid_search=GridSearchCV(estimator=knn, param_grid=param_grid,cv=3,scoring="accuracy")
    grid_search.fit(X_train,y_train)

    print(f"The best parameters are: {grid_search.best_params_} (with full data)")
    print(f"The best accuracy score is: {grid_search.best_score_} (with full data)")
    # Output is:
    # The best parameters are: {'n_neighbors': 4, 'weights': 'distance'} (with 2000 data)
    # The best accuracy score is: 0.8930069499784642 (with 2000 data)
    # The best parameters are: {'n_neighbors': 4, 'weights': 'distance'} (with 5000 data)
    # The best accuracy score is: 0.9294007945109657 (with 5000 data)
    # The best parameters are: {'n_neighbors': 4, 'weights': 'distance'} (with 10000 data)
    # The best accuracy score is: 0.9397994088551026 (with 10000 data)
    # The best parameters are: {'n_neighbors': 4, 'weights': 'distance'} (with full data)
    # The best accuracy score is: 0.9703500000000002 (with full data)
    '''
    '''
    Evaluating on the test_set with n_neighbors = 4 and weights='distance'
    '''
    '''
    knn=KNeighborsClassifier(n_neighbors=4,weights='distance')
    knn.fit(X_train,y_train)
    score=knn.score(X_test,y_test)
    print('Score for the test set training on the full training set is', score)
    # Output: Score for the test set training on the full training set is 0.9714
    '''

    ''' Try using NCA
    hyperparameters are:
    - For NCA: n_components (2-784)
    - For KNN: n_neighbors (1-20), weights (uniform or distance)
    tuned with GridSearch
    '''

    '''
    knn_nca=Pipeline([
        ('scaler',StandardScaler()),
        ('nca',NeighborhoodComponentsAnalysis()),
        ('knn',KNeighborsClassifier())
    ])
    param_grid= {'knn__n_neighbors':[4],'knn__weights':['distance','uniform'],'nca__n_components':range(13,20)}
    grid_search=RandomizedSearchCV(estimator=knn_nca, param_distributions=param_grid,n_iter=15,cv=3,scoring="accuracy")
    grid_search.fit(X_train[:2000],y_train[:2000])
    print(f"The best parameters are: {grid_search.best_params_} (with 2000 data)")
    print(f"The best accuracy score is: {grid_search.best_score_} (with 2000 data)")
    '''

    '''
    results=[]
    param_grid= {'n_neighbors':range(2,5),'weights':['distance','uniform']}
    for n_comp in range(8,10):
        scaler_nca=Pipeline([
        ('scaler',StandardScaler()),
        ('nca',NeighborhoodComponentsAnalysis(n_components=n_comp))
        ])
        X_scaled_embedded=scaler_nca.fit_transform(X_train[:5000],y_train[:5000])
        knn=KNeighborsClassifier()
        grid_search=RandomizedSearchCV(estimator=knn, param_distributions=param_grid,n_iter=5,cv=3,scoring="accuracy")
        grid_search.fit(X_scaled_embedded,y_train[:5000])
        results.append({
            'n_components':n_comp,
            'best_param': grid_search.best_params_,
            'accuracy':grid_search.best_score_
        })
    print(*results, sep='\n')
    # Output:
    # {'n_components': 8, 'best_param': {'weights': 'distance', 'n_neighbors': 2}, 'accuracy': np.float64(0.9679990772553771)}
    # {'n_components': 9, 'best_param': {'weights': 'distance', 'n_neighbors': 2}, 'accuracy': np.float64(0.9746005180716558)}
    '''
    
    '''
    We take as hyperparameters:
    - nca: n_components = 9
    - knn: n_neighbors = 2, weights = 'distance'
    '''
    '''
    knn_nca=Pipeline([
        ('scaler',StandardScaler()),
        ('nca',NeighborhoodComponentsAnalysis(n_components=9)),
        ('knn',KNeighborsClassifier(n_neighbors=3,weights='distance'))
    ])
    knn_nca.fit(X_train[:10000],y_train[:10000])
    print('Score with 10000 train data, n_components=9 and n_neighbors=3:',knn_nca.score(X_test,y_test))
    # outputs: 
    #   Score with 10000 train data, n_components=9 and n_neighbors=2: 0.9195   
    #   Score with 10000 train data, n_components=9 and n_neighbors=3: 0.929
    #   Score with 10000 train data, n_components=8 and n_neighbors=3: 0.9135
    #   Score with 10000 train data, n_components=8 and n_neighbors=1: 0.9135
    '''


   
    '''
    scores=[]
    param_grid= {'n_neighbors':range(1,5),'weights':['distance']}
    for n_comp in range(5,15):
        scaler_nca=Pipeline([
        ('scaler',StandardScaler()),
        ('nca',NeighborhoodComponentsAnalysis(n_components=n_comp,random_state=42))
        ])
        X_scaled_embedded=scaler_nca.fit_transform(X_train[:2000],y_train[:2000])
        knn=KNeighborsClassifier()
        grid_search=GridSearchCV(estimator=knn, param_grid=param_grid,cv=3,scoring="accuracy")
        grid_search.fit(X_scaled_embedded,y_train[:2000])
        scores.append(grid_search.best_score_)
        print(f'For {n_comp} the best parameters are {grid_search.best_params_} with score {grid_search.best_score_}')
    plt.plot(range(5,15), scores, marker='o')
    plt.xlabel('Number of Components')
    plt.ylabel('CV Score')
    plt.savefig(f"c3_ex1_accuracy_2000_nca.png")
    '''

if __name__ == "__main__":
    main()