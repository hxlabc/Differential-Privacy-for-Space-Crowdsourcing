import csv
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# 控制总数为5000则第一层网格密度为3~5

# 获取worker.csv的绝对路径
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
ROOT = str(ROOT)
ROOT = ROOT.replace('\\', '/')
file_path = ROOT + '/worker.csv'


def gen_point(cx, cy, x_dis, y_dis, cnt):
    x = [np.random.normal(cx, x_dis) for i in range(cnt)]
    y = [np.random.normal(cy, y_dis) for i in range(cnt)]
    return [x, y]


def gen_all(times):
    dx = np.random.randint(100, 900, size=(times,))
    dy = np.random.randint(100, 900, size=(times,))
    x_dis = np.random.randint(60, 150, size=(times,))
    y_dis = np.random.randint(50, 150, size=(times,))
    x = []
    y = []
    for i in range(times):
        res = gen_point(dx[i], dy[i], x_dis[i], y_dis[i], 500)
        x += res[0]
        y += res[1]

    # 记录没超出范围的点
    x_new = []
    y_new = []
    # 把数据写入worker.csv
    with open(file_path, 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['X', 'Y'])    # 添加表头
        for i in range(len(x)):
            if x[i] != '' and y[i] != '':
                if 0 < x[i] < 1000 and 0 < y[i] < 1000:  # 去除超出范围的点
                    x_new.append(x[i])
                    y_new.append(y[i])
                    writer.writerow([x[i], y[i]])

    plt.xlim(0, 1000)
    plt.ylim(0, 1000)
    plt.scatter(x_new, y_new, s=5, c='b', marker='*')
    plt.grid()
    plt.savefig(ROOT + '/worker_area.jpg', dpi = 600)
    plt.show()


if __name__ == '__main__':
    gen_all(10)

