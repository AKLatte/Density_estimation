# 頻度分布によるデータ

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

num = 1618
scores = np.array([[5, 1], [10, 4], [15, 19], [20, 51], [25, 76], [30, 91], [35, 99], [40, 110],
        [45, 120], [50, 137], [55, 120], [60, 117], [65, 124], [70, 116], [75, 103],
        [80, 114], [85, 85], [90, 59], [95, 49], [100, 23]], dtype=float)
# scores = np.array([[10, 16], [45, 9], [95, 21], [145, 249], [195, 1202], [245, 2902], [295, 5385], [345, 7552],
#         [395, 9138], [445, 10273], [495, 10872], [545, 11477], [595, 11116], [645, 9903], [695, 8683],
#         [745, 7535], [795, 6358], [845, 4492], [895, 3910]], dtype=float)
# scores = np.array([[10, 6], [45, 10], [95, 2], [145, 44], [195, 209], [245, 601], [295, 1162], [345, 1867],
#         [395, 2402], [445, 2904], [495, 3325], [545, 3612], [595, 3585], [645, 3407], [695, 3025],
#         [745, 2822], [795, 2243], [845, 1686], [895, 1470]], dtype=float)

per_scores = scores[:, 1] / num

# df = pd.DataFrame(scores, columns=['score', 'num'])

# score = []
# for i in scores:
#     score.append(str(i[0]))
    
data = []
for param in scores:
    for i in range(int(param[1])):
        data.append(param[0])


fig, axes = plt.subplots(1, 1)
# ret[0]:ヒストグラムの頻度分布の値，ret[1]:階級
ret = axes.hist(data, density=True, ec='black', alpha=0.3, bins=[2.5, 7.5, 12.5, 17.5, 22.5, 27.5, 32.5, 37.5, 42.5, 47.5, 52.5, 57.5, 62.5, 67.5, 72.5, 77.5, 82.5, 87.5, 92.5, 97.5])
# ret = axes.hist(data, density=True, ec='black', alpha=0.3, bins=[])
ret0 = ret[0].tolist()
ret1 = ret[1][1:].tolist()

dataset = []
for i, j in zip(ret0, ret1):
    dataset.append([i, j])
dataset = np.array(dataset)
# axes.scatter(ret[1][1:], ret[0])
plt.show()