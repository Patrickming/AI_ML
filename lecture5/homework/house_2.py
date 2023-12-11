import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier,GradientBoostingClassifier
from sklearn.svm import SVC,SVR
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,roc_auc_score,roc_curve
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from imblearn.over_sampling import RandomOverSampler,SMOTE

import seaborn as sns
import tensorflow as tf

import warnings

warnings.filterwarnings('ignore')

data = pd.read_csv('house/data.csv')

data['date'] = pd.to_datetime(data['date'])
data['date'] = data['date'] - data['date'].min()
data['date'] = data['date'].dt.days

data = data[data['price']<1500000]

data['rate'] = data['price'].apply(lambda x: '>500k' if x > 500000 else '<=500k')

for col in data.columns:
    if data[col].dtypes == 'object':
        data[col] = LabelEncoder().fit_transform(data[col])

x = data.drop('rate',axis=1)
y = data['rate']
x = (x-x.min())/(x.max()-x.min())
train_x,test_x,train_y,test_y = train_test_split(x,y,test_size=0.3,random_state=0)

train_x1,train_y1 = RandomOverSampler().fit_resample(train_x,train_y)
model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(40,activation='relu',input_shape=(train_x.shape[1],)))
model.add(tf.keras.layers.Dense(20,activation='relu'))
model.add(tf.keras.layers.Dense(1,activation='sigmoid'))
model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
model.summary()
history = model.fit(train_x1,train_y1,epochs=50,batch_size=32,validation_data=(test_x,test_y))
pred = model.predict(test_x)
y_pred = np.round(pred,0)
print(classification_report(test_y,y_pred,digits=4))
print(f'auc:{np.round(roc_auc_score(test_y,pred),4)}')

plt.figure(1,figsize=[10,5])
plt.subplot(121)
plt.plot(history.history['loss'],label='train_loss')
plt.plot(history.history['val_loss'],label='val_loss')
plt.legend()
plt.subplot(122)
plt.plot(history.history['accuracy'],label='train_acc')
plt.plot(history.history['val_accuracy'],label='val_acc')
plt.legend()
plt.show()


