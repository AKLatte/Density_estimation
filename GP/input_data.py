# 頻度分布によるデータ

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

num = 34382
scores = np.array([[10, 6], [45, 10], [95, 2], [145, 44], [195, 209], [245, 601], [295, 1162], [345, 1867],
        [395, 2402], [445, 2904], [495, 3325], [545, 3612], [595, 3585], [645, 3407], [695, 3025],
        [745, 2822], [795, 2243], [845, 1686], [895, 1470]], dtype=float)

per_scores = scores[:, 1] / num

# df = pd.DataFrame(scores, columns=['score', 'num'])

# score = []
# for i in scores:
#     score.append(str(i[0]))
    
data = []
for param in scores:
    for i in range(int(param[1])):
        data.append(param[0])


# if __name__ == "__main__":
fig, axes = plt.subplots(1, 1)
# axes.hist(data, ec='black', bins=[0, 10, 45, 95, 145, 195, 245, 295, 345, 395, 445, 495, 545, 595, 645, 695, 745, 795, 845, 895, 990])
# axes.grid(color='gray', linestyle='dotted', linewidth=1.2)
# axes.set_xticks([0, 10, 45, 95, 145, 195, 245, 295, 345, 395, 445, 495, 545, 595, 645, 695, 745, 795, 845, 895, 990])
# axes.tick_params(labelrotation=45)
# ret[0]:ヒストグラムの頻度分布の値，ret[1]:階級
ret = axes.hist(data, density=True, ec='black', alpha=0.3, bins=[0, 10, 45, 95, 145, 195, 245, 295, 345, 395, 445, 495, 545, 595, 645, 695, 745, 795, 845, 895, 990])
print('ret[0]: {}, ret[1]:{}'.format(ret[0], ret[1]))
ret0 = ret[0].tolist()
ret1 = ret[1][1:].tolist()

dataset = []
for i, j in zip(ret0, ret1):
    dataset.append([i, j])
dataset = np.array(dataset)
# axes.scatter(ret[1][1:], ret[0])
# plt.show()