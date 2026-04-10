import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDClassifier
from sklearn.datasets import fetch_openml
import joblib
from sklearn.metrics import precision_score, recall_score, confusion_matrix, accuracy_score

def main():
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
    Training binary classifiers for each number i from 0 to 9.
        Each decides if the number is 'i' or 'not i'.
        The models are saved in the list called sgd_clfs saved as file "sgd_models.pkl"
        If the file already exists and is trained for all i,
        then we just load it back.
    '''
    if os.path.exists("sgd_models.pkl"):
        # Load from file
        sgd_clfs = joblib.load("sgd_models.pkl")
        for i in range(len(sgd_clfs),10):
            y_train_new = (y_train ==str(i))
            new_clf=SGDClassifier(random_state=42)
            new_clf.fit(X_train,y_train_new)
            sgd_clfs.append(new_clf)
        joblib.dump(sgd_clfs,"sgd_models.pkl")
    else:
        i=0
        sgd_clfs = []
        while i<=9:
            y_train_new = (y_train ==str(i))
            new_clf=SGDClassifier(random_state=42)
            new_clf.fit(X_train,y_train_new)
            sgd_clfs.append(new_clf)
            i= i+1
        joblib.dump(sgd_clfs,"sgd_models.pkl")
    
    '''
    Classify digit based on the higher score it gets from each classifier
    '''
    def predict(digit):
        scores=[]
        for clf in sgd_clfs:
            scores.append(clf.decision_function([digit]))
        return scores.index(max(scores))

    y_train_pred_list = []
    for digit in X_train:
        y_train_pred_list.append(str(predict(digit)))
    y_train_pred = list_to_nparray(y_train_pred_list)
    joblib.dump(y_train_pred,"y_train_pred.pkl")

    '''
    We Use SGDClassifier() directly
    '''
    if os.path.exists("SGD_model.pkl"):
        sgd=joblib.load("SGD_model.pkl")
    else:
        sgd=SGDClassifier(random_state=42)
        sgd.fit(X_train,y_train)
        joblib.dump(sgd,"SGD_model.pkl")
    
    y_train_pred_sgd=sgd.predict(X_train)

    '''
    The accuracy, precision and recall are better for the model I implemented myself
    '''
    # print(f"Accuracy for sgd is: {accuracy_score(y_train,y_train_pred_sgd)}")
    # result: 0.8723666666666666
    # print(f"Precision score for sgd is: {precision_score(y_train,y_train_pred_sgd,average='macro')}")
    # result: 0.8880589490756167
    # print(f"Recall score for sgd is: {recall_score(y_train,y_train_pred_sgd,average='macro')}")
    # result: 0.8712474527847544
    # print(f"Accuracy for ovR is: {accuracy_score(y_train,y_train_pred)}")
    # result: 0.8822333333333333
    # print(f"Precision score for sgd is: {precision_score(y_train,y_train_pred,average='macro')}")
    # result: 0.8915072399052143
    # print(f"Recall score for sgd is: {recall_score(y_train,y_train_pred,average='macro')}")
    # result: 0.8815470689120609


    '''
    The confusion matrix is a 10x10 matrix, whose
    -rows are the actual y_values
    -columns are the predicted y_values
    The cell (i,j) tells you how many actual i's have been predicted as j.
    '''
    #cm = confusion_matrix(y_train,y_train_pred)
    #print("The confusion matrix is:")
    #print(cm)

   # cm_norm_true = confusion_matrix(y_train,y_train_pred,normalize='true')
   # print(cm_norm_true)


''' 
Transform a list into an (len(list),)-shaped array. I.e. one column and len(list)-many rows
'''
def list_to_nparray(list):
    return np.array(list)

'''
List of scores for each category 0 to 9
Attention! This doesn't work outside of main() !
'''
def fct_scores(digit):
    scores=[]
    for clf in sgd_clfs:
        scores.append(clf.decision_function([digit]))
    return scores

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
'''
some_digit=X_train[18]
plot_digit(some_digit)
plt.savefig(f"18_digit({y_train[18]}).png")
'''

if __name__=='__main__':
    main()