from matplotlib import pyplot as plt


# 打印网格信息
def show_detail(X_area, level, ok=False):
    res_noise = 0
    for xnt in X_area:
        if ok:
            xnt.output()
        res_noise += xnt.N_noise

    print("第{}级网格数：".format(level), len(X_area))
    print("加噪后总人数：", res_noise)


# 画一二级网格分割图
def draw_all_areas(alist,  # 第一层网格
                   blist   # 第二层网格
                   ):
    fig = plt.figure()

    ax = fig.add_subplot(111)
    plt.xlim(0, 1000)
    plt.ylim(0, 1000)

    for bnt in blist:
        rect = bnt.draw_rectangle(edgecolor='g')
        ax.add_patch(rect)

    for ant in alist:
        rect = ant.draw_rectangle()
        ax.add_patch(rect)

    return ax


# 画邻居
def draw_neighbors(alist,       # 第一层网格
                   blist,       # 第二层网格
                   test_pos,    # 所画的第二层网格的序号
                   ax,
                   save=False   # 是否保存为图片
                   ):

    for i in test_pos:
        target = blist[i]
        rect = target.draw_rectangle(False, color='r')
        ax.add_patch(rect)

        # print("***********************")
        # print("第{}个矩形".format(i))
        # print("上坐标：", target.pos_up)
        # print("下坐标：", target.pos_down)
        # print()

        for j in target.neighbors:
            rect = blist[j].draw_rectangle(False)
            ax.add_patch(rect)

            # print("上坐标：", blist[j].pos_up, "下坐标：", blist[j].pos_down)

    if save:
        plt.savefig('./run/Area_divide.jpg', dpi=600)

    return ax


# 画某任务t的分配区域
def draw_gr_area(tlist,        # 所有任务的坐标点   eg: [[x1,y1], ...]
                 gr_list,      # 所有任务的gr集合（附gr_max） eg: [[gr, gr_max], ...]
                 ax,
                 save=False    # 是否保存为图片
                 ):

    k = 0
    for gr, gr_max in gr_list:
        for gnt in gr:
            rect = gnt.draw_rectangle(False)
            ax.add_patch(rect)

        rect = gr_max.draw_rectangle(edgecolor='r')
        ax.add_patch(rect)

        plt.plot(tlist[k][0], tlist[k][1], 'y-^', ms=3)
        k += 1

    if save:
        plt.savefig('./run/distribution.jpg', dpi=600)

    return ax

