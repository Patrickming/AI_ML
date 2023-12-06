import pandas as pd
import numpy as np
import sklearn.preprocessing as preprocessing
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve,roc_auc_score,f1_score,precision_score,recall_score,classification_report

import warnings
warnings.filterwarnings('ignore')

def LR_model(train_x,train_y,test_x,test_y):
    lr = LogisticRegression()
    lr.fit(train_x,train_y)
    pred = lr.predict(test_x)
    pred_proba = lr.predict_proba(test_x)[:,1]
    auc = roc_auc_score(test_y,pred_proba)
    print('auc:',auc)
    print(classification_report(test_y,pred))

data = pd.read_csv('data/adult.csv',index_col=0)
for col in data.columns:
    if data[col].dtypes == 'object':
        data[col] = preprocessing.LabelEncoder().fit_transform(data[col])

x = data.drop('income',axis=1)
y = data['income']
train_x,test_x,train_y,test_y = train_test_split(x,y,test_size=0.3,random_state=0)
LR_model(train_x,train_y,test_x,test_y)
corr = train_x.join(train_y).corr()['income'].iloc[:-1]
plt.figure(1,figsize=[10,5])
corr.plot(kind='bar')
plt.tight_layout()
plt.show()

select_cols = corr[corr.abs()>0.1].index.tolist()
LR_model(train_x[select_cols],train_y,test_x[select_cols],test_y)


lr = LogisticRegression()
lr.fit(train_x,train_y)
feature_im = pd.DataFrame({'importance':lr.coef_[0]},index=train_x.columns).abs().sort_values(by='importance',ascending=False)
select_cols = feature_im.index.tolist()[:10]
LR_model(train_x[select_cols],train_y,test_x[select_cols],test_y)

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier,GradientBoostingClassifier
model_names = ['DecisionTreeClassifier','RandomForestClassifier','AdaBoostClassifier','GradientBoostingClassifier']
model_list = [DecisionTreeClassifier(),RandomForestClassifier(),AdaBoostClassifier(),GradientBoostingClassifier()]

feature_impotance = pd.DataFrame(index=train_x.columns,columns=model_names)
for i,model in enumerate(model_list):
    model.fit(train_x,train_y)
    feature_impotance[model_names[i]] = model.feature_importances_

feature_impotance['corr'] = corr.abs()
feature_impotance['LR'] = np.abs(lr.coef_[0])
feature_impotance = (feature_impotance-feature_impotance.min())/(feature_impotance.max()-feature_impotance.min())
plt.figure(2,figsize=[10,5])
feature_impotance.plot(kind='bar')
plt.tight_layout()
plt.show()
