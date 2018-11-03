# !/usr/bin/python
# coding:utf-8
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score
# %matplotlib inline
import matplotlib.pyplot as plt
import math
import pickle

cell_outage = pd.read_csv('D:\\receive_power_outage2.csv')
cell_outage['label'] = None
cell_outage['label'] = cell_outage.cell_location.apply(lambda x: 0 if x==1 else 1)
train_set,test_set = train_test_split(cell_outage, test_size=0.2, random_state=42)

X_train = train_set.iloc[0:, 4:8]
Y_train = train_set.iloc[0:, 8]
X_test = test_set.iloc[0:, 4:8]
Y_test = test_set.iloc[0:, 8]
std = StandardScaler()
X_train = std.fit_transform(X_train)
X_test = std.transform(X_test)
knn_clf = KNeighborsClassifier(n_neighbors=5, weights='distance', algorithm='auto')
knn_clf.fit(X_train,Y_train)
y_pre = knn_clf.predict(X_test)
y_pre
f1_score(Y_test,y_pre)

df1 = cell_outage[cell_outage['label']==0]
df2 = cell_outage[cell_outage['label']==1]
x1 = df1['x']
y1 = df1['y']
x2 = df2['x']
y2 = df2['y']
plt.scatter(x1, y1, s=5, c='red')
plt.scatter(x2, y2, s=5, c='blue')
# 中心六边形的中心点
hot_spot = [0, 0]
r = 1  # 正六边形对角线的一半
s = math.sqrt(3)  # 根3,保留小数点后3位
# 计算周边六边形中心点

# 读取代码计算出的点坐标
point_file = open("D:\\point.csv", "rb")
hot_end = pickle.load(point_file)
round_end = pickle.load(point_file)
roundfocu = pickle.load(point_file)
third_layer_focu = pickle.load(point_file)
# pointtt = pickle.load(point_file)


def display(hot_endpoint, round_endpoint, roundfocus, third_layer_focus):
    # 展示中心正六边形到坐标系上
    hot_x = []
    hot_y = []
    for i in range(6):
        hot_x.append(hot_endpoint[i][0])
        hot_y.append(hot_endpoint[i][1])
    hot_x.append(hot_endpoint[0][0])
    hot_y.append(hot_endpoint[0][1])
    plt.plot(hot_x, hot_y, '-')
    # 展示第2和第3层正六边形到坐标系上
    for u in range(len(round_endpoint)):
        endpoint_x = []
        endpoint_y = []
        for e in range(len(round_endpoint[u])):
            endpoint_x.append(round_endpoint[u][e][0])
            endpoint_y.append(round_endpoint[u][e][1])
        endpoint_x.append(round_endpoint[u][0][0])
        endpoint_y.append(round_endpoint[u][0][1])
        plt.plot(endpoint_x, endpoint_y, '-')

    #展示各正六边形的中心点到坐标系上
    focus_x = [0]
    focus_y = [0]
    roundfocus.extend(third_layer_focus)
    for rf in range(len(roundfocus)):
        focus_x.append(roundfocus[rf][0])
        focus_y.append(roundfocus[rf][1])
    plt.scatter(focus_x, focus_y, marker='*', color='r', s=10)

    # 展示UE_points到坐标系上
    # point_x = []
    # point_y = []
    # for p in range(len(points)):
    #     point_x.append(points[p][0])
    #     point_y.append(points[p][1])
    # plt.scatter(point_x, point_y, marker='+', color='b', s=10)
    plt.xlim(-6, 6)
    plt.ylim(-6, 6)
    plt.show()


display(hot_end, round_end, roundfocu, third_layer_focu)


