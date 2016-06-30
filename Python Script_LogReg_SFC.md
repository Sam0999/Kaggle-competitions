

```python
import pandas as pd
from sklearn import preprocessing
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import numpy as np
from sklearn.metrics import log_loss


#Load Data 
train = pd.read_csv("/Users/Saman/Desktop/BigD/Kaggle/San_Francisco_Crime_Classification/train.csv",parse_dates = ['Dates'])
test = pd.read_csv("/Users/Saman/Desktop/BigD/Kaggle/San_Francisco_Crime_Classification/test.csv", parse_dates = ['Dates'])

```


```python
#Get dummies weekdays, districts and hour
days = pd.get_dummies(train.DayOfWeek)
district = pd.get_dummies(train.PdDistrict)
hour = train.Dates.dt.hour
hour = pd.get_dummies(hour)
#Merge days & district to make a new array
train_features = pd.concat([days, district, hour], axis=1)
```


```python
#Repeat for test data
days = pd.get_dummies(test.DayOfWeek)
district = pd.get_dummies(test.PdDistrict)
hour = test.Dates.dt.hour
hour = pd.get_dummies(hour)
#Merge days & district of test_data to make a new array
test_data = pd.concat([days, district, hour], axis=1)
```


```python
#Convert crime labels to numbers
crime1 = preprocessing.LabelEncoder()
crime = crime1.fit_transform(train.Category)
type_of_crime = crime
```


```python
#split the data for cross validation
X_train, X_test, y_train, y_test = train_test_split(train_features, type_of_crime, test_size = 0.35, random_state = 4) # 0.4 means 40% for testing. generally 20-40%. Random states garantees that every time you run it, it will spli the data in a same way.

```


```python
# optimzing C parameter( as an example C=0.08 is shown) and training the model
logreg = LogisticRegression(C=0.08).fit(X_train, y_train)
print("training set score: %f" % logreg.score(X_train, y_train))
print("test set score: %f" % logreg.score(X_test, y_test))

```




```python
# Cross check the performance of the model on the new data (X_test) from the training set.

y_predd = logreg.predict_proba(X_test)
print log_loss(y_test, y_predd)

```



```python
# applying the model on test data set and writing the results to submission1.csv file.
final_pred = logreg.predict_proba(test_data)
result=pd.DataFrame(final_pred, columns=crime1.classes_)
path='/Users/Saman/Desktop/BigD/San_Francisco_Crime_Classification/'
result.to_csv(path + 'submission1.csv', index = True, index_label = 'Id' )
```


