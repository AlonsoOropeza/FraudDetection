import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import logisticModule as lm
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import r2_score, mean_squared_error

# get the data
df = pd.read_csv('chronic_kidney_disease.data')

# clean the data
for i in df.columns:
    df[i].fillna(df[i].mean(), inplace=True)
# df = df.dropna()

f = open ('kidney.txt','r')
print(''.join([line for line in f]))
f.close
# age,bp,sg,al,su,rbc,pc,pcc,ba,bgr,bu,sc,sod,pot,hemo,pcv,wbcc,rbcc,htn,dm,cad,appet,pe,ane
prediction = [56,90,1.015,2,0,0,0,0,0,129,107,6.7,131,4.8,9.1,29,6400,3.4,1,0,0,1,0,0] # should be 1
# Reshaping into 2Ds
prediction = np.array(prediction).reshape(1,-1)

# split into X and y
X = df.iloc[:,:24]
y = df[["class"]]
X = X.values
y = y.values.flatten() # to make 1d list
# print(np.shape(X),np.shape(y))

# split into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Scaling
scaler = preprocessing.StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test= scaler.transform(X_test)
prediction = scaler.transform(prediction)

def accuracy(y_true, y_pred):
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    return accuracy

# manual logistic regression
regressor = lm.LogisticRegression(learning_rate=0.0001, n_iters=1000)
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)
print('\nCoefficient of Determination', r2_score(y_test, y_pred)) 
print('MSE', mean_squared_error(y_test, y_pred)) 
if regressor.predict(prediction):
    print('You may have chronic kidney disease 💀')
else:
    print('You are safe from chronic kidney disease 🥳')

# framework logistic regression
clf = LogisticRegression(random_state=42).fit(X_train, y_train)
predictions = clf.predict(X_test)
print('\nFramework LR')
print('Coefficient of Determination', r2_score(y_test, predictions)) 
print('MSE', mean_squared_error(y_test, predictions)) 

# plot feature importance
importance = clf.coef_.flatten()
plt.bar([x for x in range(len(importance))], importance)
plt.show()
