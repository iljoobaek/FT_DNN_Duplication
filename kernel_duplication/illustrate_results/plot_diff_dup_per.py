import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns; sns.set()

sns.set_style('whitegrid')

path = "results_diff_per.txt"
# x = [0.1, 0.3, 0.5, 0.7, 0.9, 0.92, 0.94, 0.96, 0.98, 1]
# x = [0.1, 0.3, 0.5, 0.7, 0.9, 1]
width = 0
# error = 0.9

labels = []
results = np.zeros((4, 10))
cnt = 0
with open(path, 'r') as f:
    for line in f:
        arr = line.strip().split("|")
        r = cnt // 10
        c = cnt % 10
        results[r][c] = float(arr[3])
        cnt += 1

n_plots = 1
plt.figure(figsize=(4.8 * n_plots, 4.2))
# x = np.arange(1, width + 1)
x = np.arange(1, 11, 1)
# print(ori.shape, imp.shape, sco.shape, rad.shape)

for i in range(n_plots):
    plt.subplot(1, n_plots, i + 1)
    data = {'x': np.concatenate((np.array(x), np.array(x), np.array(x), np.array(x))),
            'y': np.concatenate((results[0], results[1], results[2], results[3])),
            'Duplication Type': ["no dup"] * len(x) + ["entropy"] * len(x) + ["uniform"] * len(x) + ["full dup"] * len(x)}

    df = pd.DataFrame(data)

    ax = sns.lineplot(x='x', y='y', hue='Duplication Type', data=df)
    ax.spines['top'].set_color('black')
    ax.spines['bottom'].set_color('black')
    ax.spines['left'].set_color('black')
    ax.spines['right'].set_color('black')

    plt.xlabel('Error Rate (%)', fontsize=18, fontweight='bold')
    plt.ylabel('Accuracy', fontsize=18, fontweight='bold')
    plt.title('duplication=10%\n weight error=1%', fontsize=18, fontweight='bold')
    plt.tick_params(labelsize=18)
    plt.ylim(0, 0.7)

plt.tight_layout()
#plt.show()
plt.savefig('results_diff_per.pdf')
