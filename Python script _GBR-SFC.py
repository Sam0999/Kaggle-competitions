
# coding: utf-8

# In[ ]:

import pandas as pd
from sklearn import preprocessing
from sklearn.cross_validation import train_test_split
from sklearn import metrics
import numpy as np
from sklearn.metrics import log_loss
from sklearn.ensemble import GradientBoostingClassifier


#Load Data 
train = pd.read_csv("/Users/.../BigD/Kaggle/San_Francisco_Crime_Classification/train.csv",parse_dates = ['Dates'])
test = pd.read_csv("/Users/.../BigD/Kaggle/San_Francisco_Crime_Classification/test.csv", parse_dates = ['Dates'])


# In[ ]:

#Get dummies weekdays, districts and hour
days = pd.get_dummies(train.DayOfWeek)
district = pd.get_dummies(train.PdDistrict)
hour = train.Dates.dt.hour
hour = pd.get_dummies(hour)
#Merge days & district to make a new array
train_features = pd.concat([days, district, hour], axis=1)


# In[ ]:

#Repeat for test data
days = pd.get_dummies(test.DayOfWeek)
district = pd.get_dummies(test.PdDistrict)
hour = test.Dates.dt.hour
hour = pd.get_dummies(hour)
#Merge days & district of test_data to make a new array
test_data = pd.concat([days, district, hour], axis=1)


# In[ ]:

#Convert crime labels to numbers
crime1 = preprocessing.LabelEncoder()
crime = crime1.fit_transform(train.Category)
type_of_crime = crime


# In[ ]:

# Split the data in train and test sub-sets.
X_train, X_test, y_train, y_test = train_test_split(train_features, type_of_crime, test_size = 0.25, random_state = 4) # 0.4 means 40% for testing. generally 20-40%. Random states garantees that every time you run it, it will spli the data in a same way.


# In[ ]:
# initialising the classifier
gbrt = GradientBoostingClassifier(random_state=0)
gbrt.fit(X_train, y_train)



# In[12]:

y_predd = forest.predict_proba(X_test)
print log_loss(y_test, y_predd)


# In[19]:

#Write results
final_pred = gbrt.predict_proba(test_data)
result=pd.DataFrame(final_pred, columns=crime1.classes_)
path='/Users/.../BigD/Kaggle/San_Francisco_Crime_Classification/'
result.to_csv(path + 'submissionGBC.csv', index = True, index_label = 'Id' )





