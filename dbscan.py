# !/usr/bin python
# coding:utf-8
import numpy as np
import matplotlib.pyplot as plt
## 模型数据
import pandas as pd
from sklearn.cluster import DBSCAN
import random
import pickle
from sklearn.preprocessing import StandardScaler


all_data = pd.read_csv("D:\\receive_power_outage2.csv")
# pointtt = list(all_data.iloc[:, 1:3])
singla = all_data.iloc[:, 4:]

std = StandardScaler()
singla = std.fit_transform(singla)
# X_test = std.transform(X_test)

# 归一化前参数组合为：eps=10, min_samples=6
y_si = DBSCAN(eps=1.2, min_samples=3).fit_predict(singla)  # , algorithm='ball_tree'
print(y_si)

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
    # plt.show()


display(hot_end, round_end, roundfocu, third_layer_focu)
coord = all_data.iloc[:, :4].values
# idd = [i for i, x in enumerate(coord) if x[-1] == 1]
label_si = list(set(y_si))
a = ["b", "r", "y", "g", "k", "m", "c", "coral"]
# ['Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds',
#     'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
#     'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn']
# b = random.sample(a, 1)[0]
for lsi in label_si:
    idd = [i for i, x in enumerate(y_si) if x == lsi]
    x_coord = [coord[i][1] for i in idd]
    y_coord = [coord[i][2] for i in idd]
    # plt.subplots(len(label_si), label_si.index(lsi))
    print(random.sample(a, 1)[0])
    plt.scatter(x_coord, y_coord, color=random.sample(a, 1)[0], marker='+', label="type"+str(label_si.index(lsi)))
    # color=' #054E9F'
    plt.legend(loc='upper left')
plt.show()
i = 0



