import pandas as pd
import warnings
warnings.filterwarnings('ignore')

data = pd.read_csv('data/adult.csv')
data = data.drop('Unnamed: 0', axis=1)
df = data.copy()
# 数据预处理
data['workclass'].value_counts()

# one-hot编码 职业 [0,0,0,...,1] 无偏见，稀疏矩阵
# label-encoder 职业：1 2 3 4 5 6 7 8 9 10 有偏见，稠密矩阵
# 1. 无偏见，稀疏矩阵
# l = pd.get_dummies(data['workclass'])
# data = pd.concat([data, l], axis=1)
# data.drop('workclass', axis=1, inplace=True)

from sklearn.preprocessing import OneHotEncoder,LabelEncoder

# 2. 有偏见，稠密矩阵
l1 = LabelEncoder().fit_transform(data['workclass'].values.reshape(-1, 1))

for col in data.columns:
    print(col, data[col].dtype)
    data[col] = LabelEncoder().fit_transform(data[col].values.reshape(-1, 1))

# 建模
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression,LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report,roc_auc_score
import numpy as np

def classifier_report(x,y,model):
    train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.3, random_state=0)
    clf = model.fit(train_x, train_y)
    pred = clf.predict(test_x)
    print(classification_report(test_y, pred,digits=4,labels=[0,1],target_names=['<=50K','>50K']))
    print(f'AUC: {np.round(roc_auc_score(test_y, pred),4)}')

models = [LogisticRegression(),DecisionTreeClassifier(),RandomForestClassifier(),GradientBoostingClassifier(),XGBClassifier()]
for model in models:
    print(f'***********{model}*************')
    classifier_report(data.drop('income', axis=1),data['income'],model)
# one-hot编码
# 1. 无偏见，稀疏矩阵
df['income'] = df['income'].map({' <=50K': 0, ' >50K': 1})
dis_col = []
for col in df.columns:
    if df[col].dtype != 'object':
        dis_col.append(col)

df_dis = df[dis_col]
# df_dis[col].fillna(df_dis[col].mean(), inplace=True)
# df_dis[col].fillna(df_dis[col].mode()[0], inplace=True)
# df_dis[col].fillna('unknown', inplace=True)

for col in df.columns:
    if df[col].dtype == 'object':
        l = OneHotEncoder().fit_transform(df[col].values.reshape(-1, 1)).toarray()
        l = pd.DataFrame(l, columns=[f'{col}_{i}' for i in range(l.shape[1])])
        df_dis = pd.concat([df_dis, l], axis=1)

models = [LogisticRegression(),DecisionTreeClassifier(),RandomForestClassifier(),GradientBoostingClassifier(),XGBClassifier()]
for model in models:
    print(f'***********{model}*************')
    classifier_report(df_dis.drop('income', axis=1),df_dis['income'],model)