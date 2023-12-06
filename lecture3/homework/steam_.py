from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, SGDRegressor, Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier, export_graphviz
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import seaborn as sns


def knn_iris():
    """
    用KNN算法对鸢尾花进行分类
    :return:
    """

    # 1）获取数据
    steam = pd.read_csv('lecture3/homework/data/steam.csv')
    steam_description_data = pd.read_csv('lecture3/homework/data/steam_description_data.csv')
    steam_media_data = pd.read_csv('lecture3/homework/data/steam_media_data.csv')
    steam_requirements_data = pd.read_csv('lecture3/homework/data/steam_requirements_data.csv')
    steam_support_info = pd.read_csv('lecture3/homework/data/steam_support_info.csv')
    steamspy_tag_data = pd.read_csv('lecture3/homework/data/steamspy_tag_data.csv')
    # 将 release_date 列转换为日期时间类型
    steam['release_date'] = pd.to_datetime(steam['release_date'])

    tpy=steam.dtypes

    # 计算好评率并添加到数据集中
    steam['positive_rating'] = steam['positive_ratings'] / (steam['positive_ratings'] + steam['negative_ratings'])
    steam = steam.drop(['positive_ratings', 'negative_ratings'], axis=1)
    steam.to_csv('lecture3/homework/data/steam_final.csv', index=False)
    steam_final = pd.read_csv('lecture3/homework/data/steam_final.csv')

    # 统计"english"特征的值计数
    english_counts = steam_final['english'].value_counts()

    # 设置图形大小
    plt.figure(figsize=(6, 4))

    # 画出特征图
    english_counts.plot(kind='bar')

    # 设置x轴标签
    plt.xlabel('Value')

    # 设置y轴范围为0到最大值的1.1倍
    plt.ylim(0, english_counts.max() * 1.1)

    # 添加标题
    plt.title('English Feature')

    # 显示图形
    plt.show()

    # 统计"platforms"特征的值计数
    platforms_counts = steam_final['platforms'].value_counts()

    # 设置图形大小
    plt.figure(figsize=(6, 4))

    # 画出特征图
    platforms_counts.plot(kind='bar')

    # 设置x轴标签
    plt.xlabel('Value')

    # 设置y轴范围为0到最大值的1.1倍
    plt.ylim(0, platforms_counts.max() * 1.1)

    # 添加标题
    plt.title('Platforms Feature')

    # 显示图形
    plt.show()



    # 拆分 release_date 列为年、月、日
    steam_final['year'] = steam['release_date'].dt.year
    steam_final['month'] = steam['release_date'].dt.month
    steam_final['day'] = steam['release_date'].dt.day
    # 删除原始的release_date列
    steam.drop('release_date', axis=1, inplace=True)



    # 2）划分数据集
    x_train, x_test, y_train, y_test = train_test_split(steam.data, steam.target, random_state=22)

    # 3）特征工程：标准化
    transfer = StandardScaler()
    x_train = transfer.fit_transform(x_train)
    # 用训练集的均值和方差来转换测试集，使得测试集和训练集特征值具有同样的均值和方差
    x_test = transfer.transform(x_test)

    # 4）KNN算法预估器
    estimator = KNeighborsClassifier(n_neighbors=3)
    estimator.fit(x_train, y_train)

    # 5）模型评估
    # 方法1：直接比对真实值和预测值
    y_predict = estimator.predict(x_test) # 根据测试集的特征值数据预测目标值
    print("y_predict:\n", y_predict)
    print("直接比对真实值和预测值:\n", y_test == y_predict) # 直接拿预测出的目标值 和 测试集的目标值比对

    # 方法2：计算准确率
    score = estimator.score(x_test, y_test) # 拿训练好的estimator进行预测x_test -> y_test
    print("准确率为：\n", score)

    return None

