#imports
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
import pandas as pd
#%matplotlib inline
data = pd.read_csv("Majordataset.csv")
data
print(data.columns)
data.shape
print(list(data.isnull().any()))
data.isnull().sum()
data.describe()
data.Cover_Type.value_counts()
sb.countplot(x='Cover_Type', data=data)
plt.show()
col = ['Elevation', 'Aspect', 'Slope', 'Horizontal_Distance_To_Hydrology',
       'Vertical_Distance_To_Hydrology', 'Horizontal_Distance_To_Roadways',
       'Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm',
       'Horizontal_Distance_To_Fire_Points']
train = data[col]
train.hist(figsize=(13, 11))
plt.show()
plt.style.use('ggplot')
for i in col:
    plt.figure(figsize=(13, 7))
    plt.title(str(i) + " with " + str('Cover_Type'))
    sb.boxplot(x=data.Cover_Type, y=train[i])
    plt.show()
plt.figure(figsize=(12, 8))
corr = train.corr()
sb.heatmap(corr, annot=True)
plt.show()
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
#separate features and target
feature = data.iloc[:, :54] #Features of data
y = data.iloc[:, 54]  #Target of data
# Features Reduction
ETC = ExtraTreesClassifier()
ETC = ETC.fit(feature, y)
model = SelectFromModel(ETC, prefit=True)
X = model.transform(feature) #new features 
X.shape
#Split the data into test and train formate
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=10)
#Random Forest
from sklearn.ensemble import RandomForestClassifier
RFC = RandomForestClassifier(n_estimators=100)
#fit
RFC.fit(X_train, y_train)
#prediction
y_pred = RFC.predict(X_test)
#score
print("Accuracy -- ", RFC.score(X_test, y_test)*100)
print(y_pred)
print(y_test)
df=pd.DataFrame(y_pred,y_test)
df
df.head(20)
