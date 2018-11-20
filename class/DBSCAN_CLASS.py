# !/usr/bin/python
# coding:utf-8
import matplotlib.pyplot as plt
## 模型数据
import pandas as pd
from sklearn.cluster import DBSCAN
import random
import pickle
import math
from UE_COD_CLASS import COD


all_data = pd.read_csv("D:\\receive_power.csv")
singla = all_data.iloc[:, 4:]
hot_spot = [0, 0]  # 中心六边形的中心点
r = 1  # 正六边形对角线的一半即边长
s = math.sqrt(3)

# 归一化前参数组合为：eps=10, min_samples=6
y_si = DBSCAN(eps=3.5, min_samples=20).fit_predict(singla)  # , algorithm='ball_tree'
print(y_si)

# 读取代码计算出的点坐标
point_file = open("D:\\point.csv", "rb")
hot_end = pickle.load(point_file)
round_end = pickle.load(point_file)
roundfocu = pickle.load(point_file)
third_layer_focu = pickle.load(point_file)
cod = COD(r, hot_spot, s)
setattr(cod, 'end_point', hot_end)
setattr(cod, 'round_end_points', round_end)
setattr(cod, 'round_focus', roundfocu)
setattr(cod, 'third_layer_focus', third_layer_focu)
coord = all_data.iloc[:, :4].values
# idd = [i for i, x in enumerate(coord) if x[-1] == 1]
label_si = list(set(y_si))
a = ["b", "r", "y", "g", "k", "m", "c", "coral"]
for lsi in label_si:
    idd = [i for i, x in enumerate(y_si) if x == lsi]
    x_coord = [coord[i][1] for i in idd]
    y_coord = [coord[i][2] for i in idd]
    plt.scatter(x_coord, y_coord, color=random.sample(a, 1)[0], marker='+', label="type"+str(label_si.index(lsi)))
    plt.legend(loc='upper left')
cod.display_regular_hexagon()
i = 0

