# Implementing-Artificial-Neural-Networks

You can download the dataset used in this project from [here](https://www.kaggle.com/datasets/shivan118/churn-modeling-dataset?resource=download)

## Problem Statement
The bank uses these independent variables in the data above and analyzes the behaviour of customers to see whether they leave the bank or stay. Now the bank has to create a predictive model based on this dataset in order to predict the behavior of new customers. This predictive model has to predict for any new customer whether he or she will stay in the bank or leave the bank so that the bank can offer something special to the customer whom the predictive model predicts

## Solution
```python

[ ]
#Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
[ ]
#Read our dataset
data = pd.read_csv('/content/Churn_Modelling.csv')
[ ]
#Previwieng our data
data

Next steps:
[ ]
#
X = data.iloc[:,3:13].values
Y = data.iloc[:,13].values
[ ]
print(X)
[[619 'France' 'Female' ... 1 1 101348.88]
 [608 'Spain' 'Female' ... 0 1 112542.58]
 [502 'France' 'Female' ... 1 0 113931.57]
 ...
 [709 'France' 'Female' ... 0 1 42085.58]
 [772 'Germany' 'Male' ... 1 0 92888.52]
 [792 'France' 'Female' ... 1 0 38190.78]]
[ ]
print(Y)
[1 0 1 ... 1 1 0]
[ ]
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

#Label-encoding gender column using sci-kit learn

labelencoder_X_2 = LabelEncoder()
X[:,2] = labelencoder_X_2.fit_transform(X[:,2])

print(X)

[[619 'France' 0 ... 1 1 101348.88]
 [608 'Spain' 0 ... 0 1 112542.58]
 [502 'France' 0 ... 1 0 113931.57]
 ...
 [709 'France' 0 ... 0 1 42085.58]
 [772 'Germany' 1 ... 1 0 92888.52]
 [792 'France' 0 ... 1 0 38190.78]]
[ ]
#Hot-encoding geography column
from sklearn.compose import ColumnTransformer
ct = ColumnTransformer([('ohe',OneHotEncoder(),[1])], remainder = 'passthrough')
#This hotencodes the second column[1], and making sure the remaining columns are passed through 
#without any changes

[ ]
#Transforming the data into an array and converting the data to string
X = np.array(ct.fit_transform(X), dtype = str)
[ ]
#Ignoring the first index because the same information can be achieved from the index1 and 2
X = X[:,1:]
[ ]
print(X)
[['0.0' '0.0' '619' ... '1' '1' '101348.88']
 ['0.0' '1.0' '608' ... '0' '1' '112542.58']
 ['0.0' '0.0' '502' ... '1' '0' '113931.57']
 ...
 ['0.0' '0.0' '709' ... '0' '1' '42085.58']
 ['1.0' '0.0' '772' ... '1' '0' '92888.52']
 ['0.0' '0.0' '792' ... '1' '0' '38190.78']]
[ ]
#Splitting the dataset into test and train sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)
[ ]
#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
[ ]
import pandas as pd
print (X_train)
[[-0.5698444   1.74309049  0.16958176 ...  0.64259497 -1.03227043
   1.10643166]
 [ 1.75486502 -0.57369368 -2.30455945 ...  0.64259497  0.9687384
  -0.74866447]
 [-0.5698444  -0.57369368 -1.19119591 ...  0.64259497 -1.03227043
   1.48533467]
 ...
 [-0.5698444  -0.57369368  0.9015152  ...  0.64259497 -1.03227043
   1.41231994]
 [-0.5698444   1.74309049 -0.62420521 ...  0.64259497  0.9687384
   0.84432121]
 [ 1.75486502 -0.57369368 -0.28401079 ...  0.64259497 -1.03227043
   0.32472465]]
[ ]
pd.DataFrame(X_train)

[ ]
#Building the Artificial Neural Network
[ ]
#Import the Keras libraries and packages
from tensorflow.keras.models import Sequential
from keras.layers import Dense
[ ]
#Initialise the ANN
classifier = Sequential()
[ ]
#Add Neurons
classifier.add(Dense(units =6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))
[ ]
#Add the next hidden layer
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))
[ ]
#Add the output layer
classifier.add(Dense(units =1, kernel_initializer = 'uniform', activation ='sigmoid'))
[ ]
#Compile the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics=['accuracy'])
[ ]
#Fitting the ANN model to the training set
classifier.fit(X_train,y_train, batch_size = 10, epochs = 100)
Epoch 1/100
800/800 [==============================] - 4s 3ms/step - loss: 0.4806 - accuracy: 0.7959
Epoch 2/100
800/800 [==============================] - 2s 3ms/step - loss: 0.4274 - accuracy: 0.7960
Epoch 3/100
800/800 [==============================] - 2s 2ms/step - loss: 0.4216 - accuracy: 0.8020
Epoch 4/100
800/800 [==============================] - 2s 2ms/step - loss: 0.4183 - accuracy: 0.8230
Epoch 5/100
800/800 [==============================] - 2s 2ms/step - loss: 0.4159 - accuracy: 0.8282
Epoch 6/100
800/800 [==============================] - 2s 2ms/step - loss: 0.4139 - accuracy: 0.8313
Epoch 7/100
800/800 [==============================] - 2s 2ms/step - loss: 0.4123 - accuracy: 0.8321
Epoch 8/100
800/800 [==============================] - 2s 3ms/step - loss: 0.4112 - accuracy: 0.8331
Epoch 9/100
800/800 [==============================] - 2s 3ms/step - loss: 0.4097 - accuracy: 0.8341
Epoch 10/100
800/800 [==============================] - 2s 2ms/step - loss: 0.4086 - accuracy: 0.8356
Epoch 11/100
800/800 [==============================] - 2s 2ms/step - loss: 0.4087 - accuracy: 0.8335
Epoch 12/100
800/800 [==============================] - 2s 2ms/step - loss: 0.4073 - accuracy: 0.8346
Epoch 13/100
800/800 [==============================] - 2s 2ms/step - loss: 0.4071 - accuracy: 0.8338
Epoch 14/100
800/800 [==============================] - 2s 2ms/step - loss: 0.4066 - accuracy: 0.8331
Epoch 15/100
800/800 [==============================] - 2s 3ms/step - loss: 0.4060 - accuracy: 0.8349
Epoch 16/100
800/800 [==============================] - 3s 3ms/step - loss: 0.4055 - accuracy: 0.8355
Epoch 17/100
800/800 [==============================] - 2s 2ms/step - loss: 0.4047 - accuracy: 0.8347
Epoch 18/100
800/800 [==============================] - 2s 2ms/step - loss: 0.4046 - accuracy: 0.8340
Epoch 19/100
800/800 [==============================] - 2s 2ms/step - loss: 0.4041 - accuracy: 0.8344
Epoch 20/100
800/800 [==============================] - 2s 2ms/step - loss: 0.4038 - accuracy: 0.8359
Epoch 21/100
800/800 [==============================] - 3s 3ms/step - loss: 0.4036 - accuracy: 0.8351
Epoch 22/100
800/800 [==============================] - 3s 4ms/step - loss: 0.4030 - accuracy: 0.8328
Epoch 23/100
800/800 [==============================] - 3s 3ms/step - loss: 0.4031 - accuracy: 0.8356
Epoch 24/100
800/800 [==============================] - 2s 2ms/step - loss: 0.4028 - accuracy: 0.8363
Epoch 25/100
800/800 [==============================] - 2s 2ms/step - loss: 0.4024 - accuracy: 0.8350
Epoch 26/100
800/800 [==============================] - 2s 2ms/step - loss: 0.4021 - accuracy: 0.8363
Epoch 27/100
800/800 [==============================] - 2s 2ms/step - loss: 0.4022 - accuracy: 0.8364
Epoch 28/100
800/800 [==============================] - 2s 2ms/step - loss: 0.4023 - accuracy: 0.8347
Epoch 29/100
800/800 [==============================] - 2s 3ms/step - loss: 0.4017 - accuracy: 0.8365
Epoch 30/100
800/800 [==============================] - 2s 3ms/step - loss: 0.4019 - accuracy: 0.8359
Epoch 31/100
800/800 [==============================] - 2s 2ms/step - loss: 0.4012 - accuracy: 0.8340
Epoch 32/100
800/800 [==============================] - 2s 2ms/step - loss: 0.4014 - accuracy: 0.8344
Epoch 33/100
800/800 [==============================] - 2s 2ms/step - loss: 0.4015 - accuracy: 0.8363
Epoch 34/100
800/800 [==============================] - 2s 3ms/step - loss: 0.4012 - accuracy: 0.8363
Epoch 35/100
800/800 [==============================] - 4s 5ms/step - loss: 0.4013 - accuracy: 0.8367
Epoch 36/100
800/800 [==============================] - 4s 4ms/step - loss: 0.4008 - accuracy: 0.8351
Epoch 37/100
800/800 [==============================] - 2s 3ms/step - loss: 0.4008 - accuracy: 0.8354
Epoch 38/100
800/800 [==============================] - 2s 3ms/step - loss: 0.4011 - accuracy: 0.8344
Epoch 39/100
800/800 [==============================] - 3s 4ms/step - loss: 0.4009 - accuracy: 0.8359
Epoch 40/100
800/800 [==============================] - 2s 3ms/step - loss: 0.4006 - accuracy: 0.8335
Epoch 41/100
800/800 [==============================] - 3s 3ms/step - loss: 0.4006 - accuracy: 0.8356
Epoch 42/100
800/800 [==============================] - 2s 3ms/step - loss: 0.4008 - accuracy: 0.8360
Epoch 43/100
800/800 [==============================] - 3s 4ms/step - loss: 0.4011 - accuracy: 0.8366
Epoch 44/100
800/800 [==============================] - 5s 6ms/step - loss: 0.4007 - accuracy: 0.8359
Epoch 45/100
800/800 [==============================] - 4s 5ms/step - loss: 0.4007 - accuracy: 0.8339
Epoch 46/100
800/800 [==============================] - 4s 5ms/step - loss: 0.4003 - accuracy: 0.8353
Epoch 47/100
800/800 [==============================] - 3s 3ms/step - loss: 0.4010 - accuracy: 0.8351
Epoch 48/100
800/800 [==============================] - 3s 4ms/step - loss: 0.4006 - accuracy: 0.8361
Epoch 49/100
800/800 [==============================] - 3s 4ms/step - loss: 0.4004 - accuracy: 0.8342
Epoch 50/100
800/800 [==============================] - 5s 7ms/step - loss: 0.4003 - accuracy: 0.8372
Epoch 51/100
800/800 [==============================] - 4s 5ms/step - loss: 0.4003 - accuracy: 0.8357
Epoch 52/100
800/800 [==============================] - 2s 3ms/step - loss: 0.4007 - accuracy: 0.8355
Epoch 53/100
800/800 [==============================] - 2s 2ms/step - loss: 0.4001 - accuracy: 0.8370
Epoch 54/100
800/800 [==============================] - 3s 3ms/step - loss: 0.3999 - accuracy: 0.8365
Epoch 55/100
800/800 [==============================] - 2s 3ms/step - loss: 0.4005 - accuracy: 0.8342
Epoch 56/100
800/800 [==============================] - 3s 3ms/step - loss: 0.4002 - accuracy: 0.8359
Epoch 57/100
800/800 [==============================] - 3s 3ms/step - loss: 0.4005 - accuracy: 0.8366
Epoch 58/100
800/800 [==============================] - 3s 4ms/step - loss: 0.4000 - accuracy: 0.8361
Epoch 59/100
800/800 [==============================] - 4s 5ms/step - loss: 0.4000 - accuracy: 0.8370
Epoch 60/100
800/800 [==============================] - 4s 4ms/step - loss: 0.3999 - accuracy: 0.8369
Epoch 61/100
800/800 [==============================] - 2s 3ms/step - loss: 0.3996 - accuracy: 0.8359
Epoch 62/100
800/800 [==============================] - 2s 2ms/step - loss: 0.4005 - accuracy: 0.8386
Epoch 63/100
800/800 [==============================] - 2s 2ms/step - loss: 0.4000 - accuracy: 0.8371
Epoch 64/100
800/800 [==============================] - 2s 2ms/step - loss: 0.4001 - accuracy: 0.8354
Epoch 65/100
800/800 [==============================] - 3s 3ms/step - loss: 0.4001 - accuracy: 0.8356
Epoch 66/100
800/800 [==============================] - 2s 3ms/step - loss: 0.3996 - accuracy: 0.8371
Epoch 67/100
800/800 [==============================] - 2s 2ms/step - loss: 0.4002 - accuracy: 0.8364
Epoch 68/100
800/800 [==============================] - 2s 2ms/step - loss: 0.4000 - accuracy: 0.8361
Epoch 69/100
800/800 [==============================] - 2s 2ms/step - loss: 0.3993 - accuracy: 0.8344
Epoch 70/100
800/800 [==============================] - 2s 2ms/step - loss: 0.3997 - accuracy: 0.8370
Epoch 71/100
800/800 [==============================] - 2s 2ms/step - loss: 0.3999 - accuracy: 0.8354
Epoch 72/100
800/800 [==============================] - 3s 3ms/step - loss: 0.3994 - accuracy: 0.8346
Epoch 73/100
800/800 [==============================] - 3s 3ms/step - loss: 0.3994 - accuracy: 0.8359
Epoch 74/100
800/800 [==============================] - 2s 2ms/step - loss: 0.3999 - accuracy: 0.8347
Epoch 75/100
800/800 [==============================] - 2s 2ms/step - loss: 0.3998 - accuracy: 0.8357
Epoch 76/100
800/800 [==============================] - 2s 2ms/step - loss: 0.3993 - accuracy: 0.8342
Epoch 77/100
800/800 [==============================] - 2s 2ms/step - loss: 0.3993 - accuracy: 0.8369
Epoch 78/100
800/800 [==============================] - 2s 2ms/step - loss: 0.3991 - accuracy: 0.8336
Epoch 79/100
800/800 [==============================] - 2s 3ms/step - loss: 0.3994 - accuracy: 0.8357
Epoch 80/100
800/800 [==============================] - 2s 3ms/step - loss: 0.3996 - accuracy: 0.8356
Epoch 81/100
800/800 [==============================] - 2s 2ms/step - loss: 0.3991 - accuracy: 0.8370
Epoch 82/100
800/800 [==============================] - 2s 2ms/step - loss: 0.3988 - accuracy: 0.8366
Epoch 83/100
800/800 [==============================] - 2s 2ms/step - loss: 0.3990 - accuracy: 0.8350
Epoch 84/100
800/800 [==============================] - 2s 2ms/step - loss: 0.3994 - accuracy: 0.8356
Epoch 85/100
800/800 [==============================] - 2s 2ms/step - loss: 0.3991 - accuracy: 0.8355
Epoch 86/100
800/800 [==============================] - 2s 3ms/step - loss: 0.3994 - accuracy: 0.8357
Epoch 87/100
800/800 [==============================] - 3s 3ms/step - loss: 0.3994 - accuracy: 0.8359
Epoch 88/100
800/800 [==============================] - 2s 2ms/step - loss: 0.3986 - accuracy: 0.8354
Epoch 89/100
800/800 [==============================] - 2s 2ms/step - loss: 0.3997 - accuracy: 0.8355
Epoch 90/100
800/800 [==============================] - 2s 2ms/step - loss: 0.3984 - accuracy: 0.8345
Epoch 91/100
800/800 [==============================] - 2s 2ms/step - loss: 0.3992 - accuracy: 0.8349
Epoch 92/100
800/800 [==============================] - 2s 2ms/step - loss: 0.3988 - accuracy: 0.8365
Epoch 93/100
800/800 [==============================] - 2s 3ms/step - loss: 0.3991 - accuracy: 0.8355
Epoch 94/100
800/800 [==============================] - 2s 3ms/step - loss: 0.3990 - accuracy: 0.8357
Epoch 95/100
800/800 [==============================] - 2s 2ms/step - loss: 0.3986 - accuracy: 0.8360
Epoch 96/100
800/800 [==============================] - 2s 2ms/step - loss: 0.3989 - accuracy: 0.8365
Epoch 97/100
800/800 [==============================] - 2s 2ms/step - loss: 0.3994 - accuracy: 0.8369
Epoch 98/100
800/800 [==============================] - 2s 2ms/step - loss: 0.3988 - accuracy: 0.8355
Epoch 99/100
800/800 [==============================] - 2s 2ms/step - loss: 0.3987 - accuracy: 0.8370
Epoch 100/100
800/800 [==============================] - 3s 3ms/step - loss: 0.3983 - accuracy: 0.8353
<keras.src.callbacks.History at 0x7cefc4ba7040>
[ ]
#Predicitng the test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred>0.5)
63/63 [==============================] - 0s 2ms/step
[ ]
print(y_pred)
[[False]
 [False]
 [False]
 ...
 [False]
 [False]
 [False]]
[ ]
#Make the Confusion Matrix
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
[ ]
print (cm)
[[1550   45]
 [ 271  134]]
[ ]
accuracy_score(y_test, y_pred)
0.842
```
## Conclusion
