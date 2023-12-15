import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

df = pd.read_csv(r'D:\sl\projects\AIandML\AI_ML\steamForecast\output_csv\oldsteam.csv')
# 选择除了'isravePositive'之外的所有特征
features = ['appid', 'release_date', 'english', 'developer', 'publisher',
            'platforms', 'required_age', 'categories', 'genres', 'steamspy_tags',
            'achievements', 'average_playtime', 'median_playtime', 'owners', 'price']

# 计算皮尔逊相关系数
correlation_with_target = df.corr()['isravePositive'].drop('isravePositive')

# # 绘制热力图
# plt.figure(figsize=(12, 10))
# heatmap = sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
# plt.title('Pearson Correlation Coefficients with isravePositive')
# plt.show()

# # 绘制柱状图
# plt.figure(figsize=(12, 6))
# correlation_with_target.plot(kind='bar', color='skyblue')
# plt.title('Pearson Correlation Coefficients with isravePositive')
# plt.xlabel('Features')
# plt.ylabel('Correlation with isravePositive')
# plt.xticks(rotation=45)
# plt.show()

# # 决策树
# clf = DecisionTreeClassifier()
# X = df.drop(columns=['isravePositive'])  # 特征矩阵
# y = df['isravePositive']  # 目标变量
#
# # 拟合模型
# clf.fit(X, y)
#
# # 获取特征重要性
# feature_importance = clf.feature_importances_
#
# # 创建柱状图
# plt.figure(figsize=(12, 6))
# plt.bar(X.columns, feature_importance, color='skyblue')
# plt.title('Feature Importance from Decision Tree')
# plt.xlabel('Features')
# plt.ylabel('Importance')
# plt.xticks(rotation=45)
# plt.show()

import matplotlib.pyplot as plt

# 模型名称
models = ['Logistic Regression', 'Decision Tree', 'SVM', 'Random Forest', 'XGBoost', 'LGBM', 'DNN', 'CNN']

# 模拟数据，这里使用随机数代替实际的AUC数据
auc_no_selection = [0.68, 0.56, 0.37, 0.74, 0.76, 0.77, 0.69, 0.70]
auc_pearson = [0.58, 0.56, 0.59, 0.68, 0.69, 0.69, 0.50, 0.50]
auc_tree_importance = [0.58, 0.56, 0.50, 0.69, 0.70, 0.71, 0.53, 0.58]

x = range(len(models))
bar_width = 0.25

# 绘制柱状图
plt.figure(figsize=(12, 8))
bars1 = plt.bar(x, auc_no_selection, width=bar_width, label='No Selection', color='skyblue')
bars2 = plt.bar([i + bar_width for i in x], auc_pearson, width=bar_width, label='Pearson Coefficient', color='orange')
bars3 = plt.bar([i + 2 * bar_width for i in x], auc_tree_importance, width=bar_width, label='Tree Importance', color='lightgreen')

plt.xlabel('Models')
plt.ylabel('AUC')
plt.title('Model AUC with Different Feature Selection Methods')
plt.xticks([i + bar_width for i in x], models)
plt.legend()

# 在每个柱子上方添加对应的百分数
def autolabel(bars):
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, height, f'{height:.2f}', ha='center', va='bottom')

autolabel(bars1)
autolabel(bars2)
autolabel(bars3)

plt.show()
