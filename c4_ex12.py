from sklearn.datasets import load_iris
# from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import add_dummy_feature
import numpy as np

# implement batch gradient descent with early stopping for softmax regression without using Scikit-Learn, only NumPy.
# Use it on a classification tast such as the iris data set

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

# The vector of all k-scores: each row is an instance


def soft_max(theta,i,k):
    s_k = score(theta,i,k)
    s_0 = score(theta,i,0)
    s_1 = score(theta,i,1)
    s_2 = score(theta,i,2)
    sum = np.exp(s_0) + np.exp(s_1) + np.exp(s_2)
    return np.exp(s_k)/sum

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

eta = 0.1 # learning rate
n_ephocs = 10

print(f"theta at the beginning: {theta}")

for epoch in range(n_ephocs):
    for k in range(3):
        grad = gradient(k,theta)
        theta[k,:] = theta[k,:] - eta*grad


print(f"theta after 10 epochs: {theta}")

# x is a row vector [1 x_0 x_1]
# theta @ x.T is a colum vector, each row is a k-score
# the prediction is the class k with higher score
def predict(theta,x):
    scores = theta @ x.T
    return np.argmax(scores)

def predict_prob(theta,x):
    scores = theta @ x.T
    return scores

print(f"Prediction: {predict(theta,X_train_b[0,:])}")
print(f"Prediction probabilities: {predict_prob(theta,X_train_b[0,:])}")

# SOME MISTAKE BC THE PROBABILITIES ARE NEGATIVE 