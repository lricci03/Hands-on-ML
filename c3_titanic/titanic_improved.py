import csv
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OrdinalEncoder, FunctionTransformer, PowerTransformer,StandardScaler,OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt


# Compared to titanic.py we add the categories 'FamilySize' and 'IsAlone'


'''
# create a list of dictionaries from the csv
# Each dictionary is one row
# the keys are: PassengerId,Survived,Pclass,Name,Sex,Age,SibSp,Parch,Ticket,Fare,Cabin,Embarked 
X_train=[]
y_train=[]
with open('train.csv') as file:
    reader = csv.DictReader(file)
    for row in reader:
        X_train.append({
            'PassengerId':row['PassengerId'],
            #'Survived':row['Survived'],
            'Pclass':row['Pclass'],
            'Name':row['Name'],
            'Sex':row['Sex'],
            'Age':row['Age'],
            'SibSp':row['SibSp'],
            'Parch':row['Parch'],
            'Ticket':row['Ticket'],
            'Fare':row['Fare'],
            'Cabin':row['Cabin'],
            'Embarked':row['Embarked']
        })
        y_train.append({
            'PassengerId':row['PassengerId'],
            'Survived':row['Survived'] 
        })
'''

# load the train.csv file into a Pandas DataFrame object
titanic_full = pd.read_csv('train.csv')

# print the top five rows of the data
# print(titanic_full.head())

# Create a train and a test set
train_set, test_set = train_test_split(titanic_full,test_size=0.2,random_state=42)

# Separate the predictors and the labels
titanic = train_set.drop('Survived',axis=1)
titanic_labels = train_set['Survived'].copy()


# Info about the data
#print(titanic_full.info())
'''
#   Column       Non-Null Count  Dtype  
---  ------       --------------  -----  
 0   PassengerId  891 non-null    int64  
 1   Survived     891 non-null    int64  
 2   Pclass       891 non-null    int64  
 3   Name         891 non-null    str    
 4   Sex          891 non-null    str    
 5   Age          714 non-null    float64
 6   SibSp        891 non-null    int64  
 7   Parch        891 non-null    int64  
 8   Ticket       891 non-null    str    
 9   Fare         891 non-null    float64
 10  Cabin        204 non-null    str    
 11  Embarked     889 non-null    str    
'''

# ================ Adding columns ===================
# From the forum: we might use the family size + travelling alone as additional data

titanic_full['FamilySize'] = titanic_full['SibSp'] + titanic_full['Parch'] + 1
titanic['FamilySize'] = titanic['SibSp'] + titanic['Parch'] + 1

titanic_full['IsAlone'] = (titanic_full['FamilySize'] == 1).astype(int)
titanic['IsAlone'] = (titanic['FamilySize'] == 1).astype(int)

# ============== Standardize the data ===============

# Remove tail to Fare and Standardize
#fare_pipeline = make_pipeline(PowerTransformer(method='yeo-johnson'),StandardScaler())
# yeo-johnson vs log leads analogous results. Both better than without removing tail!
fare_pipeline = make_pipeline(SimpleImputer(strategy='median'),FunctionTransformer(np.log1p,feature_names_out="one-to-one"),StandardScaler())
# Fill missing Embarked data and encode 
embarked_pipeline = make_pipeline(SimpleImputer(strategy='most_frequent'),OneHotEncoder(handle_unknown='ignore'))
# Fill missing Age data and encode
age_pipeline = make_pipeline(SimpleImputer(strategy='mean'),StandardScaler())
# RMK: using median instead of mean leads to a better result with SGD but slightly worse with other classifiers
cat_pipeline = make_pipeline(SimpleImputer(strategy='most_frequent'),OneHotEncoder(handle_unknown='ignore'))


# Encode the categorical attributes Sex and Pclass
# Apply the above Pipelines


col_transformer = ColumnTransformer([
    ('log',fare_pipeline,['Fare']),
    ('embarked',embarked_pipeline,['Embarked']),
    ('age',age_pipeline,['Age']),
    ('cat',cat_pipeline,['Sex','Pclass']),
    ('filled',SimpleImputer(strategy='most_frequent'),['FamilySize','IsAlone'])
])
titanic_prepared= col_transformer.fit_transform(titanic)

print(titanic_prepared.shape)
print(col_transformer.get_feature_names_out())

#print(titanic.head())
#print(titanic_prepared[:5])




# ===================== Fit the data ====================

# ============== KNN classifier ===============
knn_clf = KNeighborsClassifier()
knn_clf.fit(titanic_prepared,titanic_labels)
knn_scores = cross_val_score(knn_clf,titanic_prepared,titanic_labels,cv=3,scoring='accuracy')
print(f"Cross validation with KNN: {knn_scores}, mean: {knn_scores.mean()}")

# Cross validation with KNN: [0.77731092 0.78481013 0.78902954], mean: 0.7837168622723351
# Worse than without new features FamilySize and IsAlone

# ============== SGD classifier ==============
sgd_clf = SGDClassifier(random_state=42)
sgd_clf.fit(titanic_prepared,titanic_labels)
sgd_scores = cross_val_score(sgd_clf,titanic_prepared,titanic_labels,cv=3,scoring='accuracy')
print(f"Cross validation with SGD: {sgd_scores}, mean: {sgd_scores.mean()}")


# Cross validation with SGD: [0.73109244 0.7721519  0.74261603], mean: 0.7486201231547471
# Better than without new features FamilySize and IsAlone

# ============== SVC classifier ===============
svc_clf = SVC()
svc_clf.fit(titanic_prepared,titanic_labels)
svc_scores = cross_val_score(svc_clf,titanic_prepared,titanic_labels,cv=3,scoring='accuracy')
print(f"Cross validation with SVC: {svc_scores}, mean: {svc_scores.mean()}")

# Cross validation with SVC: [0.82773109 0.82278481 0.8185654 ], mean: 0.8230271011358129
# Similar as without new features FamilySize and IsAlone
# Cross validation with SVC: [0.81092437 0.84388186 0.81434599], mean: 0.823050739283055

# ============= Random Forest classifier
rf_clf = RandomForestClassifier()
rf_clf.fit(titanic_prepared,titanic_labels)
rf_scores = cross_val_score(rf_clf,titanic_prepared,titanic_labels,cv=3,scoring='accuracy')
print(f"Cross validation with RF: {rf_scores}, mean: {rf_scores.mean()}")

# Cross validation with RF: [0.77310924 0.74683544 0.80168776], mean: 0.7738774834828446
# Worse than without new features FamilySize and IsAlone


# ======================== Testing on the test set ================

# We choose SVC

svc = make_pipeline(col_transformer,SVC())
svc.fit(titanic,titanic_labels)

titanic_test = test_set.drop('Survived',axis=1)
titanic_test_labels = test_set['Survived'].copy()

titanic_test['FamilySize'] = titanic_test['SibSp'] + titanic_test['Parch'] + 1
titanic_test['IsAlone'] = (titanic_test['FamilySize'] == 1).astype(int)

predictions = svc.predict(titanic_test)
accuracy = accuracy_score(titanic_test_labels,predictions)
print(f"Accuracy on the test set is: {accuracy}")

# Accuracy on the test set is: 0.8156424581005587
# Worse than without additional features FamilySize and IsAlone




# =============== train on the entire test set =========

X_titanic_full = titanic_full.drop('Survived',axis=1)
y_titanic_full = titanic_full['Survived'].copy()
X_titanic_full['FamilySize'] = X_titanic_full['SibSp'] + X_titanic_full['Parch'] + 1
X_titanic_full['IsAlone'] = (X_titanic_full['FamilySize'] == 1).astype(int)

svc.fit(X_titanic_full,y_titanic_full)


# =============== Kaggle Submission ================

# Official test data (from Kaggle)
test_data = pd.read_csv('test.csv')

# Preprocess test_data the SAME way as your training data
test_data['FamilySize'] = test_data['SibSp'] + test_data['Parch'] + 1
test_data['IsAlone'] = (test_data['FamilySize'] == 1).astype(int)


# Generate predictions (0 or 1) with our model knn
test_predictions = svc.predict(test_data)

# Create the formatted submission DataFrame
submission_improved = pd.DataFrame({
    "PassengerId": test_data["PassengerId"],
    "Survived": test_predictions
})

# Export to CSV (CRITICAL: index=False otherwise extra column with indices 1,2,3,... not supported by Kaggle upload)
submission_improved.to_csv('submission_improved.csv', index=False)
