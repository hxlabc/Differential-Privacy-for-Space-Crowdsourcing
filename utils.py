import numpy as np
import math


# 计算第一网格粒度
def get_m1(N, epsilon):
    c1 = 10
    res = np.sqrt((N * epsilon) / c1) / 4
    res = math.ceil(res)
    return max(10, res)


# 计算第二网格粒度
def get_m2(N_noise, epsilon):
    c2 = np.sqrt(2)
    res = np.sqrt((N_noise * epsilon) / c2)
    res = math.ceil(res)
    return res


# 计算标准差圆半径
def get_r(x, y):
    n = len(x)
    if n == 1 or n == 2:       # 特殊情况处理
        return 1
    if n > 2:                  # 防止出现分母为0
        xn = np.mean(x)
        yn = np.mean(y)
        cnt = 0
        for i in range(n):
            dx = x[i] - xn
            dy = y[i] - yn
            cnt += dx**2 + dy**2
        cnt /= n-2
        r = np.sqrt(cnt)
    else:
        return -1      # 返回值为-1表示该网格人数为0
    return r


# 计算两点间距离
def get_distance(x1, y1, x2, y2):
    xd = x1 - x2
    yd = y1 - y2
    d = np.sqrt(xd**2 + yd**2)
    return d


# 工作者接受率
def worker_receive_rate(pmax, d, dmax):
    if dmax < d:
        return 0
    return pmax * (1 - d / dmax)


# 网格接受率
def area_receive_rate(n, p):
    return 1 - (1-p)**n


# 最小子网格面积
def min_area(p, p_gr, thres, N, square):
    S_min = -1

    if 0 <= p_gr < 1 and 0 < p < 1 and N > 0:     # 分母不能为0
        temp = (1 - thres) / (1 - p_gr)

        N_min = math.log(temp, 1 - p)
        S_min = (N_min / N) * square

    return S_min
