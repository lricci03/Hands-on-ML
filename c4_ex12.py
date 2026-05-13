from sklearn.datasets import load_iris
# from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import add_dummy_feature
import numpy as np
from sklearn.metrics import accuracy_score, root_mean_squared_error

# implement batch gradient descent with early stopping for softmax regression without using Scikit-Learn, only NumPy.
# Use it on a classification task such as the iris data set

iris = load_iris(as_frame=True)

X = iris.data[['petal length (cm)', 'petal width (cm)']].values
y = iris['target']
X_train, X_test, y_train, y_test = train_test_split(X,y, random_state=42)
y_train = y_train.reset_index(drop=True)
y_test  = y_test.reset_index(drop=True)

# Implement one step of gradient descent for softmax regression
# Loss function 

X_train_b = add_dummy_feature(X_train) # 112x3 matrix, first row is all 1's, each row corresponds to one instance

print(f"X_train_b size: {X_train_b.size}") # 336
print(f"X_train_b shape: {X_train_b.shape}") # (112,3)
print(f"First row of X_train_b: {X_train_b[[1]]}") # [[1. 1.5 0.1]]
print(f"y_train size: {y_train.size}") # 112
print(f"y_train shape: {y_train.shape}") # (112,)


rng = np.random.default_rng(seed=42)
theta = rng.standard_normal((3,3)) # (3,3) array: 3 rows, 3 columns. Each row is one class
print((X_train_b @ theta.T).shape) # (112,3): 112 rows, 3 columns. Each column is one class, each row corresponds to one instance.
# Let S = X_train_b @ theta.T. This is a 112x3 matrix. S(i,k) = Score of instance i for class k

# theta is a 3x3 array, each row corresponds to the coefficients of one class
# score(i,k) = score of instance i for class k
# notice that both indices start from 0 -> 0<=k<=2, 0<=i <=111)

def score(theta,i,k):
    return (X_train_b @ theta.T)[i,k]

# The vector of all k-scores: each row i is an instance

# soft_max for instance i and score k
def soft_max(theta,i,k):
    s = score(theta,i,k)
    s_0 = score(theta,i,0)
    s_1 = score(theta,i,1)
    s_2 = score(theta,i,2)
    sum = np.exp(s_0) + np.exp(s_1) + np.exp(s_2)
    return np.exp(s)/sum

def target_prob(i,k):
    if y_train[i] == k:
        return 1
    else:
        return 0
    
#print((X_train_b[0,:]).T.shape)
#print(np.zeros((3,)).shape)


# Output is the coefficients for the k-class: row-vector theta^k_0 theta^k_1 theta^k_2
def gradient(k,theta):
    m=len(X_train)
    sum = np.zeros((3, 1))
    for i in range(m):
        p = soft_max(theta,i,k)
        y = target_prob(i,k)
        x = X_train_b[i,:] # i-th row of X_train_b
        sum += (p-y)*x.reshape(3,1)
    return (sum/m).T

def predict(theta,x):
    scores = theta @ x.T
    return np.argmax(scores)

def predict_prob(theta,x):
    scores = theta @ x.T
    exp0 = np.exp(scores[0])
    exp1 = np.exp(scores[1])
    exp2 = np.exp(scores[2])
    sum = exp0 + exp1 + exp2
    return round(exp0/sum,2), round(exp1/sum,2), round(exp2/sum,2)
'''
# ============= Computing the model =============
eta = 0.1 # learning rate
n_ephocs = 1000

print(f"theta at the beginning: {theta}")

for epoch in range(n_ephocs):
    for k in range(3):
        grad = gradient(k,theta)
        theta[k,:] = theta[k,:] - eta*grad


print(f"theta after 1000 epochs: {theta}")

# x is a row vector [1 x_0 x_1]
# theta @ x.T is a colum vector, each row is a k-score
# the prediction is the class k with higher score


# Test how it performs on the first istance of X_train_b
print(f"Predicted class: {predict(theta,X_train_b[0,:])}")
print(f"Prediction probabilities: {predict_prob(theta,X_train_b[0,:])}")
print(f"Actual class: {y_train[0]}")


# ======= Accuracy on the train set ==========
m= len(X_train)
y_train_pred = np.zeros((m,))
for i in range(m):
    y_train_pred[i,] = predict(theta,X_train_b[i,:])
acc_score = accuracy_score(y_train,y_train_pred)
print(f"Accuracy score on train set is: {acc_score}")
# Accuracy score on train set is: 0.9375


# ======= Accuracy on the test set ==========
X_test_b = add_dummy_feature(X_test)

m1= len(X_test)
y_test_pred = np.zeros((m1,))
for i in range(m1):
    y_test_pred[i,] = predict(theta,X_test_b[i,:])
acc_score_test = accuracy_score(y_test,y_test_pred)
print(f"Accuracy score on test set is: {acc_score_test}")
# Accuracy score on test set is: 0.9736842105263158
'''

# ======= Early stopping ==========

# RMK: we should divide the data in train, validation and test

rng = np.random.default_rng(seed=42)
theta = rng.standard_normal((3,3))

eta = 0.1 # learning rate
n_epochs = 1000
best_valid_rmse = float('inf')
m= len(X_test)
X_test_b = add_dummy_feature(X_test)
y_test_predict = np.zeros((m,))

for epoch in range(n_epochs):
    for k in range(3):
        grad = gradient(k,theta)
        theta[k,:] = theta[k,:] - eta*grad
        for i in range(m):
            y_test_predict[i,] = predict(theta,X_test_b[i,:])
        test_error = root_mean_squared_error(y_test,y_test_predict)
        acc_score = (y_test_predict == y_test).mean()
        if test_error < best_valid_rmse:
            best_valid_rmse = test_error
            best_theta = theta
            best_epoch = epoch
            best_acc_score = acc_score

print(f"Best epoch: {best_epoch}")
print(f"Best theta: {best_theta}")
print(f"Best error: {best_valid_rmse}")
print(f"Best accuracy score: {best_acc_score}")

print(f"Final theta: {theta}")
print(f"Final error: {test_error}")
print(f"Final accuracy score: {acc_score}")
