# pandas的基础应用
import pandas as pd
import matplotlib.pyplot as plt
## 1.1 pandas的数据结构
# Series
l = pd.Series([1, 2, 3, 4, 5])
l.index
k = l.values
l.index = ['a', 'b', 'c', 'd', 'e']

# DataFrame
df = pd.DataFrame({'a': [1, 2, 3, 4, 5], 'b': [2, 3, 4, 5, 6]})
df = pd.DataFrame([[1, 2, 3, 4, 5], [2, 3, 4, 5, 6]], columns=['a', 'b', 'c', 'd', 'e'], index=['a', 'b'])
df['a']['b'] # 先列后行
df.loc['a', 'b'] # 先列后行
df.iloc[0, 1] # 先列后行

# 1.2 pandas的数据读取
# 1.2.1 读取csv文件
df = pd.read_csv('data/adult.csv')
df = df.drop('Unnamed: 0', axis=1)
df.sort_values(by='age', ascending=True, inplace=True)
df.reset_index(drop=True, inplace=True)
df.head()
df.tail()
df.info()

df['age'].mean()
df['age'].median()
df['age'].max()
df['age'].min()
df['age'].quantile(q=0.75)
df['age'].describe()

df['age'].plot.hist()
plt.show()

df['age'].plot.box()
plt.show()

df['age'].plot.kde()
plt.show()

# 统计数据情况
vl = df['workclass'].value_counts()
len(df['workclass'].unique())
df['workclass'].nunique()
vl[vl.index[0]]
print(vl.index[0])
df.columns.dtype

df1 = pd.read_csv('data/italy/italy_housing_price_rent_clean.csv')
df1['regione'].fillna('unkown', inplace=True)
df1['regione'].isnull().sum()
df1.dropna(inplace=True)