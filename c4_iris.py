from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import numpy as np

iris = load_iris(as_frame=True)

# ============= Understanding the data =================
"""
print(list(iris))
# ['data', 'target', 'frame', 'target_names', 'DESCR', 'feature_names', 'filename', 'data_module']

print(iris.data.head())
'''
   sepal length (cm)  sepal width (cm)  petal length (cm)  petal width (cm)
0                5.1               3.5                1.4               0.2
1                4.9               3.0                1.4               0.2
2                4.7               3.2                1.3               0.2
3                4.6               3.1                1.5               0.2
4                5.0               3.6                1.4               0.2
'''

print(iris.target.head())
'''
0    0
1    0
2    0
3    0
4    0'''

print(iris.target_names)
# ['setosa' 'versicolor' 'virginica']
print(iris.target[-1:])
# 149   2
"""


# =============== Classfiy Iris virginica based on width feature ===========

X=iris.data[['petal width (cm)']].values
y=iris.target_names[iris.target]=='virginica'

X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=42)


# Find coefficients theta_0 and theta_1 so that sigma(theta_0 + theta_1*X)
# is the probability of X being 'virginica'
# sigma is the logistic function 1/(1+exp(-t))
log_reg = LogisticRegression(random_state=42)
log_reg.fit(X_train,y_train)

X_new = np.linspace(0,3,1000).reshape(-1,1)
y_proba = log_reg.predict_proba(X_new)

# print(y_proba[:,1]>=0.5)
# Output is an array of True's and False's 
# The i-th position is True if the probability that X(i) (X[i,0]) is 'virginica' is >=0.5

print(X_new[y_proba[:,1]>=0.5])
# prints all the rows of X for which the corresponding y_proba is >=0.5

print(X_new[y_proba[:,1]>=0.5][0,0]) # [i,j] = i-th row, j-th column
# prints the (0,0)-element of the rows in X for which the corresponding y_proba is >=0.5
# Since the rows in X are ordered between 0 and 3, this is the smallest number between 0 and 3 
# for which the probability of being 'virginica' is >=0.5.
# Call it X_0.
# The probability is monotonically increasing (logistic function), and the slope theta_1 is >0 (one checks)
# hence p(theta_0 + theta_1 X) is monotonically increasing in X.
# Thus for all X>=X_0, the probability will be >= 0.5
# Hence X_0 is the 'decision boundary'

# This works because the petals of virginica are large, while the petals of the other kinds are small.

# This wouldn't work with the petals of versicolor, which have a middle size compared to virginica and setosa (smaller ones)
