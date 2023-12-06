import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
import warnings

warnings.filterwarnings('ignore')

data = pd.read_csv('data/italy/italy_housing_price_rent_clean.csv')

data.columns = ['region', 'city', 'quarter', 'price', 'datetime', 'parking_spaces',
       'bathrooms_per_room', 'bathrooms', 'rooms', 'top_floor', 'state',
       'energy_rating', 'sea_view', 'centralized_warming', 'surface',
       'furnished', 'balcony', 'tv_system', 'external_exposure', 'optic_fiber',
       'electric_gate', 'cellar', 'common_garden', 'private_garden',
       'alarm_system', 'doorman', 'pool', 'villa', 'entire_property',
       'apartment', 'attic', 'lofts', 'mansard']

data = data[data['price'].isnull() == False]
data = data[data['price']<5000]
data.isnull().sum()
data = data[data['region'].isnull() == False]
data = data[data['city'].isnull() == False]
data.isnull().sum()/data.shape[0]

data['quarter'].fillna('unknown',inplace=True)

data.isnull().sum()/data.shape[0]

data['bathrooms_per_room'].value_counts()
data['bathrooms_per_room'].fillna(0,inplace=True)

data['bathrooms'].value_counts()
data['bathrooms'].fillna(0,inplace=True)

data['rooms'].value_counts()
data['rooms'].fillna(0,inplace=True)

for col in data.columns:
    if data[col].isnull().sum() > 0:
        print(data[col].value_counts())
        print(data[col].isnull().sum()/data.shape[0])

data['state'].fillna('unknown',inplace=True)
data['energy_rating'].fillna(',',inplace=True)
data['sea_view'].fillna(data['sea_view'].mode()[0],inplace=True)
data['centralized_warming'].fillna(data['centralized_warming'].mode()[0],inplace=True)
data['surface'].fillna(data['surface'].mode()[0],inplace=True)

data.isnull().sum().sum()

data['datetime'] = pd.to_datetime(data['datetime'])
data['datetime'] = data['datetime'] - data['datetime'].min()
data['datetime'] = data['datetime'].dt.days

for col in data.columns:
    if data[col].dtype == 'object':
        data[col] = LabelEncoder().fit_transform(data[col])

x = data.drop(['price'],axis=1)
y = data['price']

import matplotlib.pyplot as plt
plt.figure(1,figsize=[10,5])
plt.plot(y.values)
plt.show()

import seaborn as sns
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 15
plt.figure(1,figsize=[10,5])
# 调色板
sns.set_palette('Set3')
sns.distplot(y)
plt.tight_layout()
plt.show()

def lr_model(x,y,model,model_name,num=1):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
    model.fit(x_train,y_train)
    y_pred = model.predict(x_test)
    print(model_name)
    mse = mean_squared_error(y_test,y_pred)
    mae = mean_absolute_error(y_test,y_pred)
    r2 = r2_score(y_test,y_pred)
    print('MSE:',np.round(mse,4))
    print('MAE:',np.round(mae,4))
    print('R2:',np.round(r2,4))

    plt.figure(num,figsize=[10,5])
    plt.plot(y_test.values[:100],label='true',color='r')
    plt.plot(y_pred[:100],label='pred',color='b')
    plt.title(model_name)
    plt.ylabel('price')
    plt.legend()
    plt.show()

    return mae,mse,r2

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor
from xgboost import XGBRegressor
import time
from lightgbm import LGBMRegressor

models_name = ['LinearRegression','DecisionTreeRegressor','RandomForestRegressor','GradientBoostingRegressor','XGBRegressor','LGBMRegressor']
models = [LinearRegression(),DecisionTreeRegressor(max_depth=5),RandomForestRegressor(n_estimators=50,max_depth=5),GradientBoostingRegressor(),XGBRegressor(),LGBMRegressor()]
result = pd.DataFrame(columns=['mae','mse','r2','time_cost'],index=models_name)
for i in range(len(models)):
    begin = time.time()
    mae,mse,r2 = lr_model(x,y,models[i],models_name[i],i+1)
    end = time.time()
    result['mae'][models_name[i]] = mae
    result['mse'][models_name[i]] = mse
    result['r2'][models_name[i]] = r2
    result['time_cost'][models_name[i]] = end-begin
