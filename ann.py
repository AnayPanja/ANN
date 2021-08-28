# -*- coding: utf-8 -*-
"""
Created on Fri Aug 27 23:44:52 2021

@author: Anay Panja
"""
#------------------------------------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv('Churn_Modelling.csv')
x = data.iloc[:,3:13]
y = data.iloc[:,-1]

geography=pd.get_dummies(x["Geography"],drop_first=True)
gender=pd.get_dummies(x['Gender'],drop_first=True)

x=x.drop(['Geography','Gender'],axis=1)
x=pd.concat([x,geography,gender],axis=1)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)


from sklearn.preprocessing import StandardScaler 
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

#------------------------------------------------------------------------------------------------------

import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LeakyReLU,PReLU,ELU
from keras.layers import Dropout

classifier = Sequential()

classifier.add(Dense(units=6,activation='relu',kernel_initializer='he_uniform',input_dim=11))

classifier.add(Dense(units=6,activation='relu',kernel_initializer='he_uniform'))

classifier.add(Dense(units=1,activation='sigmoid',kernel_initializer='glorot_uniform'))


classifier.compile(optimizer = 'Adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

model_history=classifier.fit(x_train, y_train,validation_split=0.33, batch_size = 10, epochs = 100)


#-------------------------------------------------------------------------------------------------------

print(model_history.history.keys())


plt.plot(model_history.history['accuracy'])
plt.plot(model_history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(model_history.history['loss'])
plt.plot(model_history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

#---------------------------------------------------------------------------------------------------------

y_pred = classifier.predict(x_test)
y_pred = y_pred>.5
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

from sklearn.metrics import accuracy_score
score=accuracy_score(y_pred,y_test)


