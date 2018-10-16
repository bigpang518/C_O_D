# !/usr/bin/python
# coding:utf-8
import random
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import copy


# 用matlib画出函数的图像
# 实现用Python画出正六边形，且坐标原点在六边形中心
def display(hot_endpoint, round_endpoint, roundfocus, third_layer_focus, points):
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
    # 计算第三场正六边形顶点坐标third_layer_endpoint,只是为了在坐标轴上展示
    third_layer_endpoint = []
    for tlf in third_layer_focuses:
        third_layer_endpoint.append(get_endpoint(tlf, r))
    round_endpoint.extend(third_layer_endpoint)
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
    point_x = []
    point_y = []
    for p in range(len(points)):
        point_x.append(points[p][0])
        point_y.append(points[p][1])
        # plt.annotate("%d" % p, xy=(points[p][0], points[p][1]), xytext=(points[p][0]-0.5, points[p][1]-0.5),
        #              arrowprops=dict(facecolor='black', shrink=0.1))
    plt.scatter(point_x, point_y, marker='+', color='b', s=10)
    plt.xlim(-6, 6)
    plt.ylim(-6, 6)
    plt.show()


# 判断点是否在中心正六边形内部，改进方法
def is_point_in_center(point, rangelist, focus, r):  # [[0,0],[1,1],[0,1],[0,0]] [1,0.8]
    point1 = rangelist[0]
    x = abs(point[0])
    y = abs(point[1])
    judge = 0
    # a = math.sqrt(3)
    # 先判断是否在顶点上
    for i in range(1, len(rangelist)):
        point2 = rangelist[i]
        if (point[0] == point1[0] and point[1] == point1[1]) or (point[0] == point2[0] and point[1] == point2[1]):
            print("zaidingdian")
    # 再判断在六边形边界或外部
    if (r - x) < y / s:
        print("buzai")
        judge = 2
    elif (r - x) == y / s:
        print("zaibianjie")
    # 后判断是否在中轴线上
    elif (point[0] == focus[0]) or (point[1] == focus[1]):
        print("zaizhongzhouxian")
        judge = 1
    # 若以上情况都为否，则说明在六边形内部
    else:
        print("zai")
        judge = 1
    return judge


def test_point_in_around(points, focuses):
    distance = 0
    x = points[0]
    y = points[1]
    for f in range(len(focuses)):
        focuse_x = focuses[f][0]
        focuse_y = focuses[f][1]
        d1 = abs(s*x - y + focuse_y + s*r - s * focuse_x) / 2
        d2 = abs(-s*x - y + focuse_y + s*r + s * focuse_x) / 2
        d3 = abs(y-focuse_y + s / 2)
        d4 = abs(s*x - y + focuse_y - s*r - s*focuse_x) / 2
        d5 = abs(-s*x - y + focuse_y - s*r + s*focuse_x) / 2
        d6 = abs(y-focuse_y - s / 2)
        if d1==0 or d2==0 or d3==0 or d4==0 or d5==0 or d6==0:
            break
        else:
            if round(d1+d2+d3, 5) == round(d4+d5+d6, 5) == round(3*s*r/2, 5):
                distance = 1
                print("该点在第%d个微蜂窝小区" % (f+2))
                points.append(f+2)
                break
    return distance, points


# 判断点到基站的距离并计算信号强度
def accumulate(point, focuses, third_focuses):
    # Free-space path loss formula in decibels
    #  FSPL(db) = 10log10((4pidf/c)^2)
    #  frequency, lte - KT from south korea, d(km), f(MHz)
    f = 2600  # MHz
     # focuses.extend(third_focuses)
    focuses.insert(0, [0, 0])
    for i in range(len(point)):
        Interference = 0
        Interference1 = 0
        ds = []
        for j in range(len(focuses)):
            d = math.sqrt(pow(point[i][0] - focuses[j][0], 2) + pow(point[i][1] - focuses[j][1], 2))
            ds.append(d)
        ds = sorted(ds)
        ds = ds[:7]
        FSPL = 20 * math.log10(ds[0]) + 20 * math.log10(f) + 32.45
        receive_power = 46 - FSPL #serving cell receive_power
        point[i].append(receive_power)
        dst = ds[1:7]
        for t in range(6):
            FSPLN = 20 * math.log10(dst[t]) + 20 * math.log10(f) + 32.45
            m = (46 - FSPLN) / 10  # 计算周围小区传送给在中心小区中的点UE的总能量
            mw = math.pow(10, m)  # 转换成mW
            Interference += mw
        rsrp = math.pow(10, receive_power / 10)  # 转换成mW
        noise = math.pow(10, -3.765)  # 把noise转换成mW
        sinr = rsrp / (Interference + noise)  #计算SINRs
        SINR = 10 * math.log10(sinr / 1000)  # 转换成db
        point[i].append(SINR)
        # 计算maximum neighboring RSRP 最近相邻小区收到的能量
        FSPL1 = 20 * math.log10(ds[1]) + 20 * math.log10(f) + 32.45
        receive_power1 = 46 - FSPL1
        point[i].append(receive_power1)
        # 计算maximum neighboring SINR 最近相邻小区收到的能量／（本小区收到能量+其他相邻5小区的能量+noise)
        dste = ds[2:7]
        for p in range(5):
            FSPLNE = 20 * math.log10(dste[p]) + 20 * math.log10(f) + 32.45
            q = (46 - FSPLNE) / 10
            Interference1 += math.pow(10, q)  # 转换成mW并叠加
        Interference1 += rsrp
        rsrp1 = math.pow(10, receive_power1 / 10)  # 从最近相邻小区收到的信号功率
        sinrn = rsrp1 / (Interference1 + noise)
        SINRn = 10 * math.log10(sinrn / 1000)  # 转换成db
        point[i].append(SINRn)
    print(len(point), "cols:", len(point[0]))
    return point


# 手动调整参数
def decrease_power(point,focuses,third_focuses):                 #下降能量之后下降基站能量调成6dbm
    # Free-space path loss formula in decibels
    #  FSPL(db) = 10log10((4pidf/c)^2)
    #  frequency, lte - KT from south korea, d(km), f(MHz)
    f = 2600  # MHz
    # focuses.extend(third_focuses)
    focuses.insert(0, [0, 0])
    for i in range(len(point)):
        sum = 0
        sum1 = 0
        ds = []
        for j in range(len(focuses)):
            d = math.sqrt(pow(point[i][0] - focuses[j][0], 2) + pow(point[i][1] - focuses[j][1], 2))
            ds.append(d)
        ds = sorted(ds)
        ds = ds[:7]
        FSPL = 20 * math.log10(ds[0]) + 20 * math.log10(f) + 32.45
        if point[i][2] == 1:
            receive_power = 6 - FSPL   # 小区1的信号能量下降单位是dbm
            point[i].append(receive_power)
            dst = ds[1:7]
            for t in range(6):
                FSPLN = 20 * math.log10(dst[t]) + 20 * math.log10(f) + 32.45
                m = (46 - FSPLN) / 10  # 计算周围小区传送给在中心小区中的点UE的总能量
                mw = math.pow(10, m)  # 转换成mW
                sum += mw
            Interference = sum
            rsrp = math.pow(10, receive_power / 10)  # 转换成mW
            noise = math.pow(10, -3.765)  # 把noise转换成mW
            sinr = rsrp / (Interference + noise)
            SINR = 10 * math.log10(sinr / 1000)  # 转换成db
            point[i].append(SINR)
            # 计算maximum neighboring RSRP 最近相邻小区收到的能量
            l = ds[1]
            FSPL1 = 20 * math.log10(l) + 20 * math.log10(f) + 32.45
            receive_power1 = 46 - FSPL1
            point[i].append(receive_power1)
            # 计算maximum neighboring SINR 最近相邻小区收到的能量／（本小区收到能量+其他相邻5小区的能量+noise)
            dste = ds[2:7]
            for p in range(5):
                FSPLNE = 20 * math.log10(dste[p]) + 20 * math.log10(f) + 32.45
                q = (46 - FSPLNE) / 10
                mwn = math.pow(10, q)  # 转换成mW
                sum1 += mwn
            Interference1 = sum1 + rsrp
            rsrp1 = math.pow(10, receive_power1 / 10)  # 从最近相邻小区收到的信号功率
            sinrn = rsrp1 / (Interference1 + noise)
            SINRn = 10 * math.log10(sinrn / 1000)  # 转换成db
            point[i].append(SINRn)
        else:
            receive_power = 46 - FSPL  # 正常的信号能量单位是dbm
            point[i].append(receive_power)
            dst = ds[1:7]
            #计算SINRs
            dist = math.sqrt(pow(point[i][0] - 0, 2) + pow(point[i][1] - 0, 2))
            FSPLC1 = 20 * math.log10(dist) + 20 * math.log10(f) + 32.45
            dst.remove(dist)
            for u in range(5):
                FSPLC = 20 * math.log10(dst[u]) + 20 * math.log10(f) + 32.45
                power = (46 - FSPLC) / 10
                sum += math.pow(10, power)
            Interference = sum + pow(10,(6 - FSPLC1)/10)  # 第二个数量为小区1的干扰
            rsrp = math.pow(10, receive_power / 10)  # 转换成mW
            noise = math.pow(10, -3.765)  # 把noise转换成mW
            sinr = rsrp / (Interference + noise)
            SINR = 10 * math.log10(sinr / 1000)  # 转换成db
            point[i].append(SINR)
            # 计算maximum neighboring RSRP 和 maximum neighboring SINR
            FSPL1 = 20 * math.log10(ds[1]) + 20 * math.log10(f) + 32.45
            if ds[1] == dist:
                receive_power1 = 6 - FSPL1
                point[i].append(receive_power1)
                dste = ds[2:7]
                for p in range(5):
                    FSPLNE = 20 * math.log10(dste[p]) + 20 * math.log10(f) + 32.45
                    q = (46 - FSPLNE) / 10
                    sum1 += math.pow(10, q)  # 转换成mW并叠加
                Interference1 = sum1 + rsrp
                rsrp1 = math.pow(10, receive_power1 / 10)  # 从最近相邻小区收到的信号功率
                sinrn = rsrp1 / (Interference1 + noise)
                SINRn = 10 * math.log10(sinrn / 1000)  # 转换成db
                point[i].append(SINRn)

            else:
                receive_power1 = 46 - FSPL1
                point[i].append(receive_power1)
                dste = ds[2:7]
                dste.remove(dist)
                for nm in range(4):
                    FSPLNE = 20 * math.log10(dste[nm]) + 20 * math.log10(f) + 32.45
                    q = (46 - FSPLNE) / 10
                    sum1 += math.pow(10, q)   # 转换成mW并叠加
                sum1 = sum1 + pow(10,(6 - FSPL1)/10)
                Interference1 = sum1 + rsrp
                rsrp1 = math.pow(10, receive_power1 / 10)  # 从最近相邻小区收到的信号功率
                sinrn = rsrp1 / (Interference1 + noise)
                SINRn = 10 * math.log10(sinrn / 1000)  # 转换成db
                point[i].append(SINRn)
    return point


def get_round_focus(hot_spot, r):
    x = hot_spot[0]
    y = hot_spot[1]
    # 按顺序依次给出上方正六边形的中心，右上方，右下方，下方，左下方，左上方正六边形中心点
    # round_focus = [up_spot, up_right_spot, down_right_spot, down_spot, down_left_spot, up_left_spot]
    round_focus = [[x, y + math.sqrt(3) * r], [x + r * 3 / 2, y + r * math.sqrt(3) / 2],
                   [x + r * 3 / 2, y - r * math.sqrt(3) / 2], [x, y - math.sqrt(3) * r],
                   [x - r * 3 / 2, y - r * math.sqrt(3) / 2], [x - r * 3 / 2, y + r * math.sqrt(3) / 2]]
    # 求第三层cell的中心点坐标list：third_layer_focus
    third_layer_focus = []
    for rf in round_focus:
        third_layer_focus.append([rf[0]*2, rf[1]*2])
    for tlf in range(9):
        if tlf % 2 == 0:
            third_layer_focus.insert(tlf+1, [(third_layer_focus[tlf][0]+third_layer_focus[tlf+1][0])/2,
                                             (third_layer_focus[tlf][1]+third_layer_focus[tlf+1][1])/2])
    third_layer_focus.insert(-1, [(third_layer_focus[0][0]+third_layer_focus[-1][0])/2,
                                     (third_layer_focus[0][1]+third_layer_focus[-1][1])/2])
    return round_focus, third_layer_focus


def get_endpoint(focus, r):
    x = focus[0]
    y = focus[1]
    # 按顺序依次给出中心点左侧，左上侧，右上侧，右侧，右下侧，左下侧的端点
    end_point = [[x - r, y], [x - r / 2, y + r * math.sqrt(3) / 2], [x + r / 2, y + r * math.sqrt(3) / 2], [x + r, y],
                 [x + r / 2, y - r * math.sqrt(3) / 2], [x - r / 2, y - r * math.sqrt(3) / 2]]
    return end_point


def get_point50(hot, endpoints):
    # 找出x轴的最大值，最小值方便后续产生随机点
    maxx = max(endpoints[1][3])
    minx = -maxx
    # y轴的最大值，最小值
    maxy = max(endpoints[0][1])  # 直接取上侧的正六边形的端点
    miny = -maxy
    p = -1
    point = []
    while p < 49:
        p += 1
        # 注：point最终的格式为[点P的x坐标，y, cell_location, RSRPs, SINRs, max_RSRPn, max_SINRn]
        # 随机产生一个点，并判断该点是否在正六边形内
        xxxxx = random.uniform(minx, maxx)
        yyyyy = random.uniform(miny, maxy)
        point.append([xxxxx, yyyyy])
        # print(point[p])
        if (hot[0] - r) < point[p][0] < (hot[0] + r) and \
                (hot[1] - r * s / 2) < point[p][1] < (hot[1] + r * s / 2):
            judge_result = is_point_in_center(point[p], hot_spot_endpoint, hot, r)
            if judge_result == 1:
                print("This UE_point is in center cell")
                point[p].append(1)
            # 若返回的结果是点落在六边形外部则判断六边形的编号
            elif judge_result == 2:
                if point[p][0] > 0:
                    if point[p][1] > 0:
                        print("This UE_point is in third cell")  # 即右上方
                        point[p].append(3)
                    elif point[p][1] < 0:
                        print("This UE_point is in fourth cell")  # 即右下方
                        point[p].append(4)
                elif point[p][0] < 0:
                    if point[p][1] > 0:
                        print("This UE_point is in seventh cell")  # 即左上方
                        point[p].append(7)
                    elif point[p][1] < 0:
                        print("This UE_point is in sixth cell")  # 即左下方
                        point[p].append(6)
            else:
                point.pop(-1)
                p = p - 1
        else:
            judge_result, point[p] = test_point_in_around(point[p], round_focuses)
            # judge_result, point[p] = is_point_in_around(point[p])
            # 若经以上判断过程后判断结果仍为0，说明该点没有落在7个正六边形任意一个
            print("judge_result", judge_result, "-"*20)
            print("point:", point[p], "="*20)
            if judge_result == 0:
                point.pop(-1)
                p = p - 1
                # print("This point is not in micro-cell")
        # print("what is p:", p)
    return point


if __name__ == '__main__':
    # 中心六边形的中心点
    hot_spot = [0, 0]
    r = 1  # 正六边形对角线的一半
    s = math.sqrt(3)  # 根3,保留小数点后3位
    # 计算周边六边形中心点
    round_focuses, third_layer_focuses = get_round_focus(hot_spot, r)
    hot_spot_endpoint = get_endpoint(hot_spot, r)  # 计算中心正六边形顶点坐标
    # 计算周边正六边形顶点坐标endpoint
    endpoint = []
    for rf in round_focuses:
        endpoint.append(get_endpoint(rf, r))
    point = get_point50(hot_spot, endpoint)
    point1 = copy.deepcopy(point)
    # 展示出设计的7个正六边形和UE点
    display(hot_spot_endpoint, endpoint, round_focuses, third_layer_focuses, point)
    # 判断点到基站的距离并计算信号强度
    signal_pow = accumulate(point, round_focuses, third_layer_focuses)
    signal_pow1 = decrease_power(point1,round_focuses, third_layer_focuses)
    # 把计算出的结果数据转成pandas的DataFrame格式并保存到csv文件print("rows:", len(point), "columns:", len(point[0]))
    power_df = pd.DataFrame(signal_pow)
    power_df1 = pd.DataFrame(signal_pow1)
    power_df.columns = ["x", "y", "cell_location", "RSRPs", "SINRs", "max_RSRPn", "max_SINRn"]
    power_df1.columns = ["x", "y", "cell_location", "RSRPs", "SINRs", "max_RSRPn", "max_SINRn"]
    power_df.to_csv("/Users/zhaominghao/Documents/Pycharm/COD/receive_power.csv")
    power_df1.to_csv("/Users/zhaominghao/Documents/Pycharm/COD/receive_power_outage.csv")