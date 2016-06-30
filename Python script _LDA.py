
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn import preprocessing
from sklearn.cross_validation import train_test_split
from sklearn import metrics
import numpy as np
from sklearn.metrics import log_loss
from sklearn.lda import LDA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from collections import Counter



#Load Data 
train = pd.read_csv("/Users/.../BigD/Kaggle/San_Francisco_Crime_Classification/train.csv",parse_dates = ['Dates'])
test = pd.read_csv("/Users/.../BigD/Kaggle/San_Francisco_Crime_Classification/test.csv", parse_dates = ['Dates'])


# In[2]:

#Get dummies weekdays, districts and hour
days = pd.get_dummies(train.DayOfWeek)
district = pd.get_dummies(train.PdDistrict)
hour = train.Dates.dt.hour
hour = pd.get_dummies(hour)
#Merge days & district to make a new array
train_features = pd.concat([district, hour], axis=1)


# In[3]:

#Repeat for test data
days = pd.get_dummies(test.DayOfWeek)
district = pd.get_dummies(test.PdDistrict)
hour = test.Dates.dt.hour
hour = pd.get_dummies(hour)
#Merge days & district of test_data to make a new array
test_data = pd.concat([district, hour], axis=1)


# In[4]:

#Convert crime labels to numbers
crime1 = preprocessing.LabelEncoder()
crime = crime1.fit_transform(train.Category)
type_of_crime = crime


# In[5]:
# splitting data for cross validation
X_train, X_test, y_train, y_test = train_test_split(train_features, type_of_crime, test_size = 0.28, random_state = 5) # 0.4 means 40% for testing. generally 20-40%. Random states garantees that every time you run it, it will spli the data in a same way.


# In[6]:

lda = LDA(solver='svd', shrinkage=None, priors=None, n_components=None, store_covariance=False, tol=0.001)
lda = lda.fit(X_train, y_train)
y_pred_lda = lda.predict(X_test)
y_predd_lda = lda.predict_proba(X_test)
print '\n lda metrics.accuracy_score:\n', metrics.accuracy_score(y_test, y_pred_lda)
print '\n lda metrics.accuracy_score:\n', log_loss(y_test, y_predd_lda)



clf = QuadraticDiscriminantAnalysis()
clf = clf.fit(X_train, y_train)
y_pred_clf = clf.predict(X_test)
y_predd_clf = clf.predict_proba(X_test)
print '\n clf metrics.accuracy_score:\n', metrics.accuracy_score(y_test, y_pred_clf)
print '\n clf metrics.accuracy_score:\n', log_loss(y_test, y_predd_clf)


# In[7]:

#Write results
final_pred = lda.predict_proba(test_data)
result=pd.DataFrame(final_pred, columns=crime1.classes_)
path='/Users/.../BigD/Kaggle/San_Francisco_Crime_Classification/'
result.to_csv(path + 'submission02.csv', index = True, index_label = 'Id' )


# In[ ]:



