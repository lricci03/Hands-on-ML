import csv
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import OrdinalEncoder, FunctionTransformer, PowerTransformer
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt


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
print(titanic_full.info())
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
# For most of the passengers the Cabin data misses.
# We remove the column Cabin
titanic.drop('Cabin',axis=1,inplace=True)
# We remove the column Embarked
titanic.drop('Embarked',axis=1,inplace=True)
# We set Age = 0 for the passengers who don't have it
titanic['Age'] = titanic['Age'].fillna(0)
# We remove Name
titanic.drop('Name',axis=1,inplace=True)
# We convert the attribute Sex to a numerical attribute
ordinal_encoder=OrdinalEncoder()
sex_cat=titanic[['Sex']]
sex_cat_encoded = ordinal_encoder.fit_transform(sex_cat)

# We look at ticket
ticket_cat = titanic[['Ticket']]
print(ticket_cat.head(10))
'''
Output:
               Ticket
331             113043
733              28425
382  STON/O 2. 3101293
704             350025
813             347082
118           PC 17558
536             113050
361      SC/PARIS 2167
29              349216
55               19947
'''
# we remove this category 
titanic.drop('Ticket',axis=1,inplace=True)
# SibSp: no. of siblings / spouses aboard the Titanic
# Parch: no. of parents / children aboard the Titanic	
# Embarked: Port of Embarkation	C = Cherbourg, Q = Queenstown, S = Southampton


# Plot attributes against each other
'''
attributes = ['Survived','Age','Sex','Pclass','Fare']
scatter_matrix(titanic_full[attributes],figsize=(15,10))
plt.savefig('correlation.png')
'''

# Get correlation indices with survive
corr_matrix = train_set.corr(numeric_only=True)
print(corr_matrix['Survived'].sort_values(ascending=False))

# Plot an histogram of the data
'''titanic_full.hist(bins=50,figsize=(15,10))
plt.savefig('histogram.png')
'''

# From the histogram:
#   The Fare attribute has a heavy tail to the right -> take the log
# Using log1p = log(1+x) b/c Fare contains 0
'''
log_transformer = FunctionTransformer(np.log1p)
log_fare = log_transformer.transform(titanic[['Fare']])
log_fare.hist(bins=50,figsize=(15,10))
plt.savefig('log_fare_hist.png')
'''

# It is not ideal. We use instead the Yeo-Johnson transformer
pt = PowerTransformer(method='yeo-johnson')
#titanic['Fare_Norm'] 
norm_fare = pt.fit_transform(titanic[['Fare']])
plt.hist(norm_fare,bins=50)
plt.savefig('norm_fare_hist.png')




print()