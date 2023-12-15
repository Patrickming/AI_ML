import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score, roc_auc_score
from xgboost import XGBClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from datetime import datetime
import matplotlib
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

matplotlib.use('TkAgg')
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
df = pd.read_csv(r'D:\sl\projects\AIandML\AI_ML\steamForecast\steam.csv')

df['release_date'] = (datetime.now() - pd.to_datetime(df['release_date'])).dt.days
df['positive_rate'] = df['positive_ratings'] / (df['positive_ratings'] + df['negative_ratings'])
df['owners'] = df['owners'].str.split('-').str[0].astype(int)

# columns_to_drop = ['positive_ratings', 'negative_ratings']
columns_to_drop = ['positive_ratings', 'negative_ratings',"appid","name"]
# columns_to_drop = ['positive_ratings', 'negative_ratings', "developer", "publisher","platforms",'median_playtime', 'owners']
# columns_to_drop = ['positive_ratings', 'negative_ratings',"english","appid","median_playtime"]
df = df.drop(columns=columns_to_drop)
# 导出数据集
# df.to_csv('D:\sl\projects\AIandML\AI_ML\steamForecast\output_csv\steamtest2.csv', index=False)

df['developer'] = LabelEncoder().fit_transform(df['developer'])
df['publisher'] = LabelEncoder().fit_transform(df['publisher'])
df['steamspy_tags'] = LabelEncoder().fit_transform(df['steamspy_tags'])
df['platforms'] = LabelEncoder().fit_transform(df['platforms'])
df['categories'] = LabelEncoder().fit_transform(df['categories'])
df['genres'] = LabelEncoder().fit_transform(df['genres'])
# df['appid'] = LabelEncoder().fit_transform(df['appid'])
# df['name'] = LabelEncoder().fit_transform(df['name'])

df['positive_rate'] = (df['positive_rate'] >= 0.90)
df['isravePositive'] = df['positive_rate']
df['isravePositive'] = LabelEncoder().fit_transform(df['isravePositive'])
drop2 = ['positive_rate']
df = df.drop(columns=drop2)

# df.to_csv('D:\sl\projects\AIandML\AI_ML\steamForecast\output_csv\oldsteam.csv', index=False)


y = df['isravePositive']
X = df.drop(columns=['isravePositive'])

# df.to_csv('D:\sl\projects\AIandML\AI_ML\steamForecast\output_csv\steamfinal.csv', index=False)


X = np.array(X)
y = np.array(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 用于存储每个模型指标的列表
models_metrics = []

# 逻辑回归
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
y_pred_logreg = logreg.predict(X_test)
metrics_logreg = accuracy_score(y_test, y_pred_logreg), precision_score(y_test, y_pred_logreg), recall_score(y_test, y_pred_logreg), f1_score(y_test, y_pred_logreg), roc_auc_score(y_test, logreg.predict_proba(X_test)[:, 1])
models_metrics.append(('逻辑回归', *metrics_logreg))

# 决策树
dt_classifier = DecisionTreeClassifier()
dt_classifier.fit(X_train, y_train)
y_pred_dt = dt_classifier.predict(X_test)
metrics_dt = accuracy_score(y_test, y_pred_dt), precision_score(y_test, y_pred_dt), recall_score(y_test, y_pred_dt), f1_score(y_test, y_pred_dt), roc_auc_score(y_test, dt_classifier.predict_proba(X_test)[:, 1])
models_metrics.append(('决策树', *metrics_dt))

# 支持向量机
svm_classifier = SVC(probability=True)
svm_classifier.fit(X_train, y_train)
y_pred_svm = svm_classifier.predict(X_test)
metrics_svm = accuracy_score(y_test, y_pred_svm), precision_score(y_test, y_pred_svm), recall_score(y_test, y_pred_svm), f1_score(y_test, y_pred_svm), roc_auc_score(y_test, svm_classifier.predict_proba(X_test)[:, 1])
models_metrics.append(('支持向量机', *metrics_svm))

# 随机森林
rf_classifier = RandomForestClassifier()
rf_classifier.fit(X_train, y_train)
y_pred_rf = rf_classifier.predict(X_test)
metrics_rf = accuracy_score(y_test, y_pred_rf), precision_score(y_test, y_pred_rf), recall_score(y_test, y_pred_rf), f1_score(y_test, y_pred_rf), roc_auc_score(y_test, rf_classifier.predict_proba(X_test)[:, 1])
models_metrics.append(('随机森林', *metrics_rf))

# XGBoost
xgb_classifier = XGBClassifier()
xgb_classifier.fit(X_train, y_train)
y_pred_xgb = xgb_classifier.predict(X_test)
metrics_xgb = accuracy_score(y_test, y_pred_xgb), precision_score(y_test, y_pred_xgb), recall_score(y_test, y_pred_xgb), f1_score(y_test, y_pred_xgb), roc_auc_score(y_test, xgb_classifier.predict_proba(X_test)[:, 1])
models_metrics.append(('XGBoost', *metrics_xgb))


# LGBM
lgbm_classifier = LGBMClassifier()
lgbm_classifier.fit(X_train, y_train)
y_pred_lgbm = lgbm_classifier.predict(X_test)
metrics_lgbm = accuracy_score(y_test, y_pred_lgbm), precision_score(y_test, y_pred_lgbm), recall_score(y_test, y_pred_lgbm), f1_score(y_test, y_pred_lgbm), roc_auc_score(y_test, lgbm_classifier.predict_proba(X_test)[:, 1])
models_metrics.append(('LGBM', *metrics_lgbm))


early_stopping = EarlyStopping(monitor='accuracy', patience=10)
# TensorFlow模型
def build_dnn_model(input_shape):
    model = Sequential([
        Dense(256, activation='relu', input_shape=(input_shape,)),  # 增加神经元
        Dense(128, activation='relu'),
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    return model


dnn_model = build_dnn_model(X_train.shape[1])
dnn_model.compile(optimizer=Adam(0.005), loss='binary_crossentropy', metrics=['accuracy'])
dnn_model.fit(X_train, y_train, epochs=100, batch_size=64, verbose=1, validation_split=0.2, callbacks=[early_stopping])
y_pred_dnn = dnn_model.predict(X_test)
y_pred_dnn_classes = (y_pred_dnn > 0.5).astype(int)
metrics_dnn = accuracy_score(y_test, y_pred_dnn_classes), precision_score(y_test, y_pred_dnn_classes), recall_score(y_test, y_pred_dnn_classes), f1_score(y_test, y_pred_dnn_classes), roc_auc_score(y_test, y_pred_dnn)
models_metrics.append(('DNN', *metrics_dnn))

def build_cnn_model(input_shape):
    model = Sequential([
        Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=(input_shape, 1), padding='same'),
        Conv1D(filters=64, kernel_size=3, activation='relu'),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(64, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    return model


cnn_model = build_cnn_model(X_train.shape[1])
cnn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
X_train_cnn = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
cnn_model.fit(X_train_cnn, y_train, epochs=100, batch_size=64, verbose=1, callbacks=[early_stopping], validation_split=0.2)
X_test_cnn = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
y_pred_cnn = cnn_model.predict(X_test_cnn)
y_pred_cnn_classes = (y_pred_cnn > 0.5).astype(int)
metrics_cnn = accuracy_score(y_test, y_pred_cnn_classes), precision_score(y_test, y_pred_cnn_classes), recall_score(y_test, y_pred_cnn_classes), f1_score(y_test, y_pred_cnn_classes), roc_auc_score(y_test, y_pred_cnn)
models_metrics.append(('CNN', *metrics_cnn))


metrics_df = pd.DataFrame(models_metrics, columns=['模型', '准确率', '精确度', '召回率', 'F1分数', 'AUC'])

fig, ax = plt.subplots(figsize=(10, 6))
metrics_df.set_index('模型')[['准确率', 'AUC']].plot(kind='bar', ax=ax)
ax.set_ylabel('得分')
ax.set_title('不同模型在准确率和AUC上的性能')

for p in ax.patches:
    ax.annotate(f'{p.get_height():.4f}', (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center', xytext=(0, 10), textcoords='offset points')

plt.xticks(rotation=0)
plt.show()

correlations = df.corr()['positive_rate'].drop('positive_rate')

fig, ax = plt.subplots(figsize=(10, 6))
correlations.plot(kind='bar', ax=ax)
ax.set_ylabel('Correlation')
ax.set_title('特征变量与标签的相关性')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()


