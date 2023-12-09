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


    steam_final = pd.read_csv('lecture4/homework/data/steam_final.csv')
    data_ = pd.read_csv('lecture4/homework/data/data.csv')

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

    # 统计每个开发商和发行商的人数
    developer_counts = steam_final['developer'].value_counts()

    # 选择出现次数最多的十家开发商
    top_developers = developer_counts.head(10)

    # 设置图形大小
    plt.figure(figsize=(10, 6))

    top_developers.plot(kind='bar')
    # 添加标题和轴标签
    plt.title('Top 10 Developers')
    plt.xlabel('Developer')
    plt.ylabel('Count')

    # 显示图形
    plt.show()

    # 统计每个发行商的数量
    publisher_counts = steam_final['publisher'].value_counts()

    # 选择出现次数最多的十家发行商
    top_publishers = publisher_counts.head(10)

    # 设置图形大小
    plt.figure(figsize=(10, 6))

    # 画出柱状图
    top_publishers.plot(kind='bar')

    # 添加标题和轴标签
    plt.title('Top 10 Publishers')
    plt.xlabel('Publisher')
    plt.ylabel('Count')

    # 显示图形
    plt.show()

    # 统计特征"platforms"中各个平台类别的数量
    platform_counts = steam_final['platforms'].value_counts()

    # 设置图形大小
    plt.figure(figsize=(10, 6))

    # 画出柱状图
    platform_counts.plot(kind='bar')

    # 添加标题和轴标签
    plt.title('Platform Counts')
    plt.xlabel('Platform')
    plt.ylabel('Count')

    # 设置x轴标签的斜体旋转角度为45度
    plt.xticks(rotation=25)

    # 显示图形
    plt.show()

    # 统计"categories"特征的值计数
    categories_counts = steam_final['categories'].value_counts().head(10)

    # 设置图形大小
    plt.figure(figsize=(12, 8))

    # 画出特征图
    categories_counts.plot(kind='bar')

    # 设置x轴标签斜体旋转角度为45度
    plt.xticks(rotation=15, ha='right')

    # 添加标题和轴标签
    plt.title('Categories Feature')
    plt.xlabel('Category')
    plt.ylabel('Count')

    # 显示图形
    plt.show()

    # 统计"genres"特征的值计数
    categories_counts = steam_final['genres'].value_counts().head(15)

    # 设置图形大小
    plt.figure(figsize=(12, 8))

    # 画出特征图
    categories_counts.plot(kind='bar')

    # 设置x轴标签斜体旋转角度为45度
    plt.xticks(rotation=15, ha='right')

    # 添加标题和轴标签
    plt.title('genres Feature')
    plt.xlabel('genres')
    plt.ylabel('Count')

    # 显示图形
    plt.show()

    # 统计"steamspy_tags"特征的值计数
    categories_counts = steam_final['steamspy_tags'].value_counts().head(15)

    # 设置图形大小
    plt.figure(figsize=(12, 8))

    # 画出特征图
    categories_counts.plot(kind='bar')

    # 设置x轴标签斜体旋转角度为45度
    plt.xticks(rotation=15, ha='right')

    # 添加标题和轴标签
    plt.title('steamspy_tags Feature')
    plt.xlabel('steamspy_tags')
    plt.ylabel('Count')

    # 显示图形
    plt.show()

    # 根据不同分类统计"achievements"特征的数量
    achievements_counts = pd.cut(steam_final['achievements'], bins=[0, 50, 100, 150, float('inf')],
                                 labels=['0-50', '51-100', '101-150', '>150']).value_counts()

    # 设置图形大小
    plt.figure(figsize=(10, 6))

    # 画出柱状图
    achievements_counts.plot(kind='bar')

    # 添加标题和轴标签
    plt.title('Achievements')
    plt.xlabel('Category')
    plt.ylabel('Count')
    plt.xticks(rotation=25, ha='right')

    # 显示图形
    plt.show()

    # 根据不同范围统计"positive_rating"特征的数量
    positive_rating_counts = pd.cut(steam_final['positive_rating'], bins=[0, 0.25, 0.5, 0.75, 1],
                                    labels=['0-25%', '26%-50%', '51%-75%', '76%-100%']).value_counts()

    # 设置图形大小
    plt.figure(figsize=(10, 6))

    # 画出柱状图
    positive_rating_counts.plot(kind='bar')

    # 添加标题和轴标签
    plt.title('Positive Rating')
    plt.xlabel('Range')
    plt.ylabel('Count')
    plt.xticks(rotation=30, ha='right')

    # 显示图形
    plt.show()

    # 根据不同范围统计"average_playtime"特征的数量
    playtime_counts = pd.cut(steam_final['average_playtime'], bins=[0, 100, 300, 600, float('inf')],
                             labels=['0-100', '101-300', '301-600', '>600']).value_counts()

    # 设置图形大小
    plt.figure(figsize=(10, 6))

    # 画出柱状图
    playtime_counts.plot(kind='bar')

    # 添加标题和轴标签
    plt.title('Average Playtime')
    plt.xlabel('Range')
    plt.ylabel('Count')

    # 显示图形
    plt.show()

    # 读取数据集
    df = pd.read_csv('your_dataset.csv')

    # 根据不同范围统计"median_playtime"特征的数量
    playtime_counts = pd.cut(steam_final['median_playtime'], bins=[0, 100, 300, 600, float('inf')],
                             labels=['0-100', '101-300', '301-600', '>600']).value_counts()

    # 设置图形大小
    plt.figure(figsize=(10, 6))

    # 画出柱状图
    playtime_counts.plot(kind='bar')

    # 添加标题和轴标签
    plt.title('Median Playtime')
    plt.xlabel('Range')
    plt.ylabel('Count')
    plt.xticks(rotation=30, ha='right')

    # 显示图形
    plt.show()

    # 统计每个owner的数量
    publisher_counts = steam_final['owners'].value_counts().head(15)


    # 设置图形大小
    plt.figure(figsize=(10, 6))

    # 画出柱状图
    publisher_counts.plot(kind='bar')

    # 添加标题和轴标签
    plt.title('Top 15 ')
    plt.xlabel('owner')
    plt.ylabel('Count')
    plt.xticks(rotation=30, ha='right')


    # 显示图形
    plt.show()

    # 定义不同范围的标签和范围
    labels = ['0-5', '6-10', '11-15', '16-20', '21-25', '26-30', '31-35', '36-40', '41-45', '46-50', '51-55', '56-60',
              '61-65', '66-70']
    ranges = [(0, 5), (6, 10), (11, 15), (16, 20), (21, 25), (26, 30), (31, 35), (36, 40), (41, 45), (46, 50), (51, 55),
              (56, 60), (61, 65), (66, 70)]

    # 根据不同范围统计"price"特征的数量
    price_counts = pd.cut(steam_final['price'], bins=len(labels), labels=labels).value_counts()

    # 设置图形大小
    plt.figure(figsize=(10, 6))

    # 画出柱状图
    price_counts.plot(kind='bar')

    # 添加标题和轴标签
    plt.title('Price')
    plt.xlabel('Range')
    plt.ylabel('Count')

    # 显示图形
    plt.show()
    return None

