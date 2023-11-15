import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
'''
y = bX + e
'''
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import random
RANDOM_SEED = 2333
# 归一化
# print(X)
X_normalized = (X - X.min()) / (X.max() - X.min())
#print(X_normalized)
# Y_normalized = (Y - Y.min()) / (Y.max() - Y.min())
# print(Y_normalized)

# 划分训练集和测试集
res = 0
for i in range(1000):
    RANDOM_SEED = (int)(random.random() * 1000)
    # print(RANDOM_SEED)
    X_train, X_test, Y_train, Y_test = train_test_split(X_normalized, Y, test_size = 0.2, random_state = RANDOM_SEED)
    # print(X_test)

    # 线性回归
    LR = LinearRegression()
    LR.fit(X_train, Y_train)
    Result = LR.predict(X_test)
    RMSE = np.sqrt(mean_squared_error(Y_test, Result))
    res = res + RMSE
print(res/1000)

# 误差分析
errors = Y_test - Result
plt.figure(figsize=(9, 6))
sns.scatterplot(x=Y_test, y=errors)
plt.axhline(y=0, color='red', linestyle='--')
plt.title('Error Scatter Plot')
plt.xlabel('Actual Score')
plt.ylabel('Prediction Error')
plt.show()

plt.figure(figsize=(9, 6))
sns.histplot(errors, kde=True)
plt.title('Error Distribution')
plt.xlabel('Prediction Error')
plt.ylabel('Frequency')
plt.show()

plt.figure(figsize=(9, 6))
sns.scatterplot(x=Result, y=errors)
plt.axhline(y=0, color='red', linestyle='--')
plt.title('Residual Plot')
plt.xlabel('Predicted Score')
plt.ylabel('Residuals')
plt.show()

data_df = pd.read_csv('./cwurData.csv')  # 读入 csv 文件为 pandas 的 DataFrame
data_df.head(3).T  # 观察前几列并转置方便观察
data_df = data_df.dropna()  # 舍去包含 NaN 的 row
len(data_df)
feature_cols = ['quality_of_faculty', 'publications', 'citations', 'alumni_employment', 
                'quality_of_education', 'patents'] # 'influence', 'broad_impact'
X = data_df[feature_cols]
Y = data_df['score']
print(Y)
# 绘制热图
correlation_matrix = data_df[feature_cols].corr()
plt.figure(figsize=(9, 6))
sns.heatmap(correlation_matrix, annot = True, cmap = 'coolwarm', fmt = '.2f', linewidths = .4)
plt.title("Correlation Matrix of Features and Target")
plt.show()
'''
y = bX + e
'''
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import random
import math
RANDOM_SEED = 2333
# 归一化
# print(X)
X_normalized = (X - X.min()) / (X.max() - X.min())
#print(X_normalized)
# Y_normalized = (Y - Y.min()) / (Y.max() - Y.min())
# print(Y_normalized)


# 划分训练集和测试集
res = 0
for i in range(1000):
    RANDOM_SEED = (int)(random.random() * 1000)
    # print(RANDOM_SEED)
    X_train, X_test, Y_train, Y_test = train_test_split(X_normalized, Y, test_size = 0.2, random_state = RANDOM_SEED)

    Judge = Y_train <= 70
    X_train_1 = X_train[Judge]
    Y_train_1 = Y_train[Judge]
    X_train_2 = X_train[~Judge]
    Y_train_2 = Y_train[~Judge]
    print(Y_train_2)

    Judge = Y_test <= 70
    X_test_1 = X_test[Judge]
    Y_test_1 = Y_test[Judge]
    X_test_2 = X_test[~Judge]
    Y_test_2 = Y_test[~Judge]

    LR1 = LinearRegression()
    LR1.fit(X_train_1, Y_train_1)
    Result1 = LR1.predict(X_test_1)
    RMSE = sum((Y_test_1 - Result1) ** 2)
    # print("QWQ:", type(Result1))
    LR2 = LinearRegression()
    LR2.fit(X_train_2, Y_train_2)
    Result2 = LR2.predict(X_test_2)
    RMSE = math.sqrt((RMSE + (sum((Y_test_2 - Result2)) ** 2)) / X_test.shape[0])
    res = res + RMSE
        
print(res / 1000)