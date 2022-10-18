# 得点分布のグラフ

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

num = 1618
scores = np.array([[5, 1], [10, 4], [15, 19], [20, 51], [25, 76], [30, 91], [35, 99], [40, 110],
        [45, 120], [50, 137], [55, 120], [60, 117], [65, 124], [70, 116], [75, 103],
        [80, 114], [85, 85], [90, 59], [95, 49], [100, 23]], dtype=float)

data1 = []
for x, y in scores:
    data1.append([x-2.5, y])

data1 = np.array(data1)
data2 = []
for param in data1:
    for i in range(int(param[1])):
        data2.append(param[0])

ret = plt.hist(data2, density=True, ec='black', alpha=0.3, bins=[0,5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100])
plt.title('得点分布', fontname="MS Gothic")
plt.xlabel('得点（点）', fontname="MS Gothic")
plt.ylabel('人数（人）', fontname="MS Gothic")
plt.show()