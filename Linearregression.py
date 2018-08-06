import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from sklearn.linear_model import LinearRegression

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

#data imported in csv format
data=pd.read_csv("/Users/Santhosh/Downloads/50_Startups.csv")


# separating dependent and independent variables
X = data.iloc[:,:-1].values
Y = data.iloc[:,4].values

#using encoders and one hot since one of dependent variable is in categorical format
labelencoder=LabelEncoder()
X[:,3]=labelencoder.fit_transform(X[:,3])
onehotencoder=OneHotEncoder(categorical_features=[3])
X=onehotencoder.fit_transform(X).toarray()

# splitting variables for train and test
X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(X, Y, test_size = 0.33, random_state = 5)
print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)

# importing and initialising model
learn = LinearRegression()

#tarining model
learn.fit(X_train, Y_train)
Y_train_pred = learn.predict(X_train)
Y_test_pred = learn.predict(X_test)


#printing test and train data
df=pd.DataFrame(Y_test_pred,Y_test)
print(df)

#calculating error
mse = mean_squared_error(Y_test, Y_test_pred)
print(mse)
import numpy as np
rmse=np.sqrt(mse)
print(rmse)

#Plotting 
plt.scatter(Y_train_pred, Y_train_pred - Y_train,c='blue',marker='o',label='Training data')
plt.scatter(Y_test_pred, Y_test_pred - Y_test,c='lightgreen',marker='s',label='Test data')
plt.xlabel('Predicted values')
plt.ylabel('Residuals')
plt.legend(loc= 'upper left')
plt.hlines(y=0,xmin=40000,xmax=200000)
plt.plot()
plt.show()







