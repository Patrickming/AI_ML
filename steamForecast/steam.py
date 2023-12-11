import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score, roc_auc_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

# 数据预处理
df = pd.read_csv('steam.csv')
df['release_date'] = pd.to_datetime(df['release_date'])
df['days_since_release'] = (datetime.now() - df['release_date']).dt.days
df['positive_rate'] = df['positive_ratings'] / (df['positive_ratings'] + df['negative_ratings'])
df['year'] = df['release_date'].dt.year
df['month'] = df['release_date'].dt.month
df['owners'] = df['owners'].str.split('-').str[0].astype(int)

features = ['days_since_release', 'english', 'required_age', 'achievements',
            'positive_ratings', 'negative_ratings', 'average_playtime',
            'median_playtime', 'owners', 'price', 'year', 'month']
X = df[features]
y = df['positive_rate'] >= 0.8

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# 特征缩放
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 为了使用CNN，数据需要被重塑为3D数组
X_train_scaled_cnn = X_train_scaled.reshape(X_train_scaled.shape[0], X_train_scaled.shape[1], 1)
X_test_scaled_cnn = X_test_scaled.reshape(X_test_scaled.shape[0], X_test_scaled.shape[1], 1)


# 传统机器学习模型
log_reg = LogisticRegression()
log_reg.fit(X_train_scaled, y_train)

svm_model = SVC(probability=True)
svm_model.fit(X_train_scaled, y_train)

decision_tree = DecisionTreeClassifier(max_depth=3, min_samples_split=10, min_samples_leaf=5, random_state=42)
decision_tree.fit(X_train_scaled, y_train)

# 集成学习模型 - 参数调整以减少过拟合
random_forest = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
random_forest.fit(X_train_scaled, y_train)

xgboost_model = XGBClassifier(n_estimators=20, max_depth=3, gamma=0.1, use_label_encoder=False, eval_metric='logloss', random_state=42)
xgboost_model.fit(X_train_scaled, y_train)

lightgbm_model = LGBMClassifier(n_estimators=30, max_depth=3, num_leaves=8, random_state=42)
lightgbm_model.fit(X_train_scaled, y_train)

# TensorFlow模型
def build_dnn_model(input_shape):

    model = Sequential([
        Dense(128, activation='relu', input_shape=(input_shape,)),
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])
    return model

early_stopping = EarlyStopping(monitor='val_loss', patience=3)
dnn_model = build_dnn_model(X_train_scaled.shape[1])
dnn_model.fit(X_train_scaled, y_train, epochs=20, batch_size=32, verbose=1, callbacks=[early_stopping], validation_split=0.2)

def build_cnn_model(input_shape):
    model = Sequential([
        Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(input_shape, 1)),
        MaxPooling1D(pool_size=2),
        Flatten(),
        Dense(64, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

cnn_model = build_cnn_model(X_train_scaled.shape[1])
X_train_scaled_cnn = X_train_scaled.reshape((X_train_scaled.shape[0], X_train_scaled.shape[1], 1))
cnn_model.fit(X_train_scaled_cnn, y_train, epochs=20, batch_size=32, verbose=1, callbacks=[early_stopping], validation_split=0.2)

# 性能评估
models = [log_reg, svm_model, decision_tree, random_forest, xgboost_model, lightgbm_model, dnn_model, cnn_model]
model_names = ['Logistic Regression', 'SVM', 'Decision Tree', 'Random Forest', 'XGBoost', 'LightGBM', 'DNN', 'CNN']
accuracies, recalls, aucs = [], [], []

for model, name in zip(models, model_names):
    if name in ['DNN', 'CNN']:
        X_eval = X_test_scaled_cnn if name == 'CNN' else X_test_scaled
        predictions = model.predict(X_eval).flatten()
    else:
        predictions = model.predict(X_test_scaled)
        predictions_proba = model.predict_proba(X_test_scaled)[:, 1] if hasattr(model, "predict_proba") else np.nan

    accuracy = accuracy_score(y_test, predictions.round())
    recall = recall_score(y_test, predictions.round())
    auc = roc_auc_score(y_test, predictions_proba) if not np.isnan(predictions_proba).any() else np.nan
    accuracies.append(accuracy)
    recalls.append(recall)
    aucs.append(auc)

    print(f'{name} - Accuracy: {accuracy:.4f}, Recall: {recall:.4f}, AUC: {auc:.4f}')

# 可视化
x = np.arange(len(model_names))
width = 0.2

fig, ax = plt.subplots()
ax.bar(x - width, accuracies, width, label='Accuracy')
ax.bar(x, recalls, width, label='Recall')
ax.bar(x + width, aucs, width, label='AUC')

ax.set_ylabel('Scores')
ax.set_title('Model Performance Comparison')
ax.set_xticks(x)
ax.set_xticklabels(model_names, rotation=45)
ax.legend()

plt.tight_layout()  # Adjust the plot to ensure everything fits without overlapping
plt.savefig('model_performance.png')
plt.show()