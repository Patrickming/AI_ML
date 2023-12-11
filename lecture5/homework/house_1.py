import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# 1. 数据读取
data = pd.read_csv('lecture5/house/data.csv')

# 对全局的图画样式进行修改（此处是字体和字号）
plt.rcParams['font.sans-serif'] = ['Times New Roman']
plt.rcParams.update({'font.size': 20})
sns.set_palette('Set2')

data['date'] = pd.to_datetime(data['date'])
data['date'] = data['date'] - data['date'].min()
data['date'] = data['date'].dt.days

data = data[data['price']<1500000]

data['rate'] = data['price'].apply(lambda x: '>500k' if x > 500000 else '<=500k')

# 2. 数据可视化分析
# 对于字符型
def plot_categorical(data,col):
    fig, ax1 = plt.subplots(1,figsize=(15, 6))
    sns.countplot(data=data, x=col, hue='rate', ax=ax1)
    ax1.set_title('Count and Proportion of Price >500k by Category')
    ax1.set_ylabel('Count')

    # 创建右侧的y轴
    ax2 = ax1.twinx()

    # 计算并绘制每个类别中 >25k 收入的占比
    price_proportion = data.groupby(col)['rate'].apply(lambda x: np.mean(x == '>500k')).reset_index()
    # price_proportion的排序要根据data[col].unique()的字符自定义排序
    proportion = pd.DataFrame(index=data[col].unique(), columns=[col, 'rate'])
    for col_ in data[col].unique():
        proportion.loc[col_, col] = col_
        proportion.loc[col_, 'rate'] = price_proportion.loc[price_proportion[col] == col_, 'rate'].values[0]

    sns.lineplot(x=col, y='rate', data=proportion, marker='o', ax=ax2, color='red')
    ax2.set_ylabel('Proportion (>500k)')
    for label in ax1.get_xticklabels():
        label.set_rotation(45)
    plt.tight_layout()
    plt.show()

for col in data.columns:
    if data[col].dtypes == 'object':
        if col != 'rate':
            plot_categorical(data,col)

# 连续变量可视化分析
def plot_continuous(data,col,num=1):
    plt.figure(num, figsize=[10, 5])
    sns.distplot(data.loc[data['rate'] == '<=500k'][col], label='<=500k')
    sns.distplot(data.loc[data['rate'] == '>500k'][col], label='>500k')
    plt.legend()
    plt.tight_layout()
    plt.show()

num = 1
for col in data.columns:
    if data[col].dtypes != 'object':
        plot_continuous(data,col,num)
        num += 1

# 3. 数据预处理
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
for col in data.columns:
    if data[col].dtypes == 'object':
        data[col] = LabelEncoder().fit_transform(data[col])

corr = data.corr()
# 重新调整字号
plt.rcParams.update({'font.size': 12})
plt.figure(1,figsize=[10,8])
# 加注释，调颜色
sns.heatmap(corr,annot=True,cmap='RdBu')
plt.tight_layout()
plt.show()
