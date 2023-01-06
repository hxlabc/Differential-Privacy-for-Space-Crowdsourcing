"""
------------------------------------------------------------------------------------
Author: hxl
Create Date: 2023/01/05
Centralized Differential Privacy for Space Crowdsourcing Location Privacy Protection

Usage:
    $ python psd_ngga.py

    $ python psd_ngga.py --epsilon 0.8 --B 0.6 --thres 0.9 --dmax 45 --pmax 0.8


------------------------------------------------------------------------------------
"""

import argparse
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from My_worker import Area
from draw import draw_all_areas, draw_gr_area, show_detail
from utils import get_m1, get_m2, min_area


# 加载数据
def data_load(file_path):
    worker = pd.read_csv(file_path)
    # print(worker.head())
    Num = len(worker)  # 工作者总数

    x = list(worker['X'])
    y = list(worker['Y'])
    return x, y, Num


# 找相邻网格
def find_neighbor(X_area):
    n = len(X_area)
    for i in range(n):
        for j in range(n):
            if i != j:
                if X_area[i].is_neighbor(X_area[j].pos_up, X_area[j].pos_down):
                    X_area[i].add_neighbor(j)


# 分割第一层网格和第二层网格
def BWPSD(x, y, Num, epsilon, B):
    length = 1000  # 整个区域的边长
    sum1 = 0       # 用来记录第一级网格标准差半径总和
    sum2 = 0  # 用来记录第二级网格标准差半径总和
    epsilon1 = B * epsilon  # 第一部分隐私预算
    epsilon2 = epsilon - epsilon1  # 第二部分隐私预算
    A_area = []  # 第一级网格集合
    B_area = []  # 第二级网格集合

    m1 = get_m1(Num, epsilon)
    dis = length / m1     # 第一层小网格的边长
    vaild_num1 = 0     # 第一层有效网格数，即网格中工作者人数大于0
    vaild_num2 = 0  # 第二层有效网格数，即网格中工作者人数大于0

    real_num1 = 0
    real_num2 = 0

    # 第一层网格的划分
    for i in range(m1):
        for j in range(m1):
            x_up = j * dis
            y_up = i * dis + dis
            x_down = j * dis + dis
            y_down = i * dis

            area = Area([x_up, y_up], [x_down, y_down], 1)
            # 将属于该区域的点放入该区域，并返回没有划分进入该区域点的集合，提高算法效率
            x, y = area.divide(x, y)    # 更新x 和 y

            # 判断其是否为有效网格
            if area.r != -1:
                vaild_num1 += 1
                sum1 += area.r

            area.No = real_num1  # 记录该一级网格在A_area中的标号
            real_num1 += 1
            A_area.append(area)

    for area in A_area:
        flag = 1
        if area.r != -1:
            wt = area.get_wt(sum1)     # 计算该网格的隐私保护度
            e = wt * epsilon1        # 计算Ai分配的隐私预算
            area.add_noise(e, 1, vaild_num1)      # 添加噪声

            # 第二层网格的划分
            if area.N_noise > 0:       # 如果加噪后的人数少于等于0，直接跳过
                m2 = get_m2(area.N_noise, epsilon2)
                dis2 = dis / m2
                x2 = area.x
                y2 = area.y

                flag = 0   # 用来标记未被划分的一级网格

                for i in range(m2):
                    for j in range(m2):
                        x_up = area.pos_up[0] + j * dis2
                        y_up = area.pos_down[1] + i * dis2 + dis2
                        x_down = area.pos_up[0] + j * dis2 + dis2
                        y_down = area.pos_down[1] + i * dis2

                        area2 = Area([x_up, y_up], [x_down, y_down], 2)
                        # 将属于该区域的点放入该区域，并返回没有划分进入该区域点的集合，提高算法效率
                        x2, y2 = area2.divide(x2, y2)  # 更新x 和 y
                        # 判断其是否为有效网格
                        if area2.r != -1:
                            vaild_num2 += 1
                            sum2 += area2.r

                        area2.No = real_num2  # 记录该二级网格在B_area中的标号
                        area.include_parts.append(area2.No)  # 将该二级网格的标号归类到该一级网格中
                        real_num2 += 1
                        B_area.append(area2)

        # 将未被划分的一级网格列入二级网格，防止find_pointarea返回空
        if flag:
            temp = area.__copy__()
            temp.level = 2
            temp.r = -1
            temp.No = real_num2
            area.include_parts.append(temp.No)
            B_area.append(temp)
            real_num2 += 1

    for area2 in B_area:
        if area2.r != -1:
            wt2 = area2.get_wt(sum2)  # 计算该网格的隐私保护度
            e2 = wt2 * epsilon2  # 计算Bi分配的隐私预算
            area2.add_noise(e2, 1, vaild_num2)  # 添加噪声

    find_neighbor(B_area)

    return A_area, B_area


# 找到该任务点所属的二级网格
def find_pointarea(pos_x, pos_y, A_area, B_area):
    for xnt in A_area:
        if xnt.is_inArea(pos_x, pos_y):
            for i in xnt.include_parts:
                if B_area[i].is_inArea(pos_x, pos_y):
                    return B_area[i]
    return None


# 判断该网格是否在GR_max内
def is_inGRmax(area, GR_max):
    x1, y1 = area.pos_up
    x2, y2 = area.pos_down
    # 一个矩形的上下顶点都在GR_max内改矩形才在GR_max内
    return GR_max.is_inArea(x1, y1) and GR_max.is_inArea(x2, y2)


# 按照最小面积分割网格
def getarea_by_smin(original_area, neibor_area, S_min):
    # 新顶点的坐标
    x_up = y_up = x_down = y_down = 0

    # 首先判断 neibor_area网格在 original_area网格的哪一边
    x1, y1 = original_area.pos_up
    x2, y2 = original_area.pos_down
    x3, y3 = neibor_area.pos_up
    x4, y4 = neibor_area.pos_down

    # 分四种情况

    left = np.abs(x1 - x4) <= 0.00000001  # x1 == x4
    right = np.abs(x2 - x3) <= 0.00000001  # x2 == x3
    up = np.abs(y1 - y4) <= 0.00000001  # y1 == y4
    down = np.abs(y2 - y3) <= 0.00000001  # y2 == y3

    # neibor_area在 original_area的左边
    if left:
        dl = y3 - y4     # 左边的边长
        h = S_min / dl   # 面积除以边长得到高

        x_up = x4 - h
        y_up = y3
        x_down = x4
        y_down = y4

    # neibor_area在 original_area的右边
    elif right:
        dl = y3 - y4      # 右边的边长
        h = S_min / dl    # 面积除以边长得到高

        x_up = x3
        y_up = y3
        x_down = x3 + h
        y_down = y4

    # neibor_area在 original_area的上边
    elif up:
        dl = x4 - x3  # 左边的边长
        h = S_min / dl  # 面积除以边长得到高

        x_up = x3
        y_up = y4 + h
        x_down = x4
        y_down = y4

    # neibor_area在 original_area的下边
    elif down:
        dl = x4 - x3  # 左边的边长
        h = S_min / dl  # 面积除以边长得到高

        x_up = x3
        y_up = y3
        x_down = x4
        y_down = y3 - h

    new_area = Area([x_up, y_up], [x_down, y_down], 2)
    # 待补充：更新区域内点的信息
    return new_area


# 任务分配算法
def NGGA(tx, ty, thres, dmax, pmax, A_area, B_area):
    GR = []         # 初始化任务广播域
    begin_area = find_pointarea(tx, ty, A_area, B_area)     # 任务t所在区域网格区域

    # GR_max 任务t为中心，边长为 2 * dmax 的正方形区域
    lim_up = [tx - dmax, ty + dmax]
    lim_down = [tx + dmax, ty - dmax]
    GR_max = Area(lim_up, lim_down, -1)

    if begin_area is not None:
        _, p_gr = begin_area.calculate_p_rec(tx, ty, dmax, pmax)

        GR.append(begin_area)

        u = 0   # 用来记录GR中当前遍历到的网格
        visited = [begin_area.No]     # 用来存已经访问过且未经过分割的网格的编号（因为GR中可能包含切割后的网格）

        # 如果当前GR的接受率达不到阈值thres 且 任务t所在区域网格在GR_max内
        if p_gr < thres and is_inGRmax(begin_area, GR_max):
            while p_gr < thres:
                qlist = []  # 候选栈

                if u < len(GR):             # 防止越界
                    cur_area = GR[u]
                    u += 1
                else:
                    break

                for k in cur_area.neighbors:
                    if k not in visited:
                        visited.append(k)       # 标记已访问
                        temp_area = B_area[k]
                        if is_inGRmax(B_area[k], GR_max) is False:    # 取B_area[k]与GR_max的相交网格
                            temp_area = B_area[k].get_intersect_area(GR_max)

                        if temp_area is not None:     # 防止B_area[k]网格与GR_max不相交
                            p, p_rec = temp_area.calculate_p_rec(tx, ty, dmax, pmax)
                            qlist.append([temp_area, p_rec, p])

                # 排序，取网格接受率最高的先遍历
                qlist = sorted(qlist, key=lambda x: x[1], reverse=True)

                # area_i 为网格    prec_i 为该网格对应的接受率   p_i 为该网格的工作者接受率
                for area_i, prec_i, p_i in qlist:
                    if prec_i == 0:    # 如果网格接受率为0则不分配，直接跳过
                        continue

                    pgr_old = p_gr          # 记录更新前的变量
                    p_gr = 1 - (1 - prec_i) * (1 - p_gr)    # 更新GR的接受率

                    if p_gr < thres:
                        GR.append(area_i)
                    else:                   # 如果更新后的p_gr超过阈值，则划定最小面积区域
                        square = area_i.get_square()
                        S_min = min_area(p_i, pgr_old, thres, area_i.N, square)    # 计算最小面积

                        if S_min != -1:    # 如果该最小面积存在
                            small_area = getarea_by_smin(cur_area, area_i, S_min)
                            GR.append(small_area)
                            break

        # 如果当前GR的接受率已超过阈值thres 且 任务t所在区域网格在GR_max内
        elif p_gr > thres and is_inGRmax(begin_area, GR_max):
            p, _ = begin_area.calculate_p_rec(tx, ty, dmax, pmax)
            square = begin_area.get_square()
            S_min = min_area(p, 0, thres, begin_area.N, square)  # 计算最小面积

            # 构造以（tx, ty) 为中心，面积大小为S_min的正方形区域
            bian = np.sqrt(S_min) / 2
            small_area = Area([tx - bian, ty + bian], [tx + bian, ty - bian], 2)
            GR[0] = small_area
            # 待补充：更新区域内点的信息

        # 该区域超过最大广播域GR_max时，则其与GR_max的相交网格作为GR
        else:
            intersect_area = begin_area.get_intersect_area(GR_max)
            if intersect_area is not None:         # 防止出现None
                GR[0] = intersect_area

        return GR, GR_max


# 输入参数
def parse_opt():
    parser = argparse.ArgumentParser(description="Centralized Differential Privacy")
    parser.add_argument('--epsilon', type=float, default=1, help='Privacy budget')
    parser.add_argument('--B', type=float, default=0.5, help='Budget factor')
    parser.add_argument('--thres', type=float, default=0.95, help='threshold')
    parser.add_argument('--dmax', type=float, default=50, help='Maximum distance')
    parser.add_argument('--pmax', type=float, default=0.65, help='Maximum acceptance rate')
    parser.add_argument('--source', type=str, default='./data/worker.csv', help='Data path')
    parser.add_argument('--save', default=False, help='Whether to save the picture')

    opt = parser.parse_args()
    parser.print_help()
    print(opt)

    return opt


# 运行程序
def run(opt):
    epsilon = opt.epsilon        # 总共的隐私预算
    B = opt.B                    # 预算因子
    thres = opt.thres            # GR接受率阈值
    dmax = opt.dmax              # 工作者最大旅行距离
    pmax = opt.pmax              # 工作者最大接受率
    file_path = opt.source       # 数据的路径
    is_Save = opt.save           # 是否保存结果（图片）

    start = time.time()   # 开始时间

    x, y, Num = data_load(file_path)
    alist, blist = BWPSD(x, y, Num, epsilon, B)

    # show_detail(alist, 1, True)         # 打印每个网格的信息
    # show_detail(blist, 2, True)
    # print("总人数：", Num)

    ax = draw_all_areas(alist, blist)
    # plt.scatter(x, y, s=2, c='#f0a732', marker='*')  # 画上工作者

    # 随机的任务地点集合
    tlist = [[np.random.randint(50, 950), np.random.randint(50, 950)] for k in range(25)]
    gr_list = []
    for tl in tlist:
        gr, gr_max = NGGA(tl[0], tl[1], thres, dmax, pmax, alist, blist)
        gr_list.append([gr, gr_max])

    draw_gr_area(tlist, gr_list, ax, is_Save)

    print('Finish ! ! !')
    end = time.time()
    print("Time consumption:", end - start)

    plt.show()


if __name__ == '__main__':
    opt = parse_opt()
    run(opt)



