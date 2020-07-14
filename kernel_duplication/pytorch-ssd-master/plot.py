import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns; sns.set()

sns.set_style('whitegrid')

path = "results_1.txt"
# x = [0.1, 0.3, 0.5, 0.7, 0.9, 0.92, 0.94, 0.96, 0.98, 1]
x = [0.1, 0.3, 0.5, 0.7, 0.9, 1]

y_att_all = []
y_imp_all = []
y_sco_all = []
labels = []
y_ori = []
y_att = []
y_imp = []
y_sco = []
with open(path, 'r') as f:
    for line in f:
        if "#duplication" in line:
            y_att_all.append(y_att)
            y_imp_all.append(y_imp)
            y_sco_all.append(y_sco)
            y_att = []
            y_imp = []
            y_sco = []
            labels.append(line.strip()[13:])
            continue
        arr = line.strip().split("|")
        if arr[1] == "False":
            y_ori.append(float(arr[3]))
        else:
            if arr[2] == "attention":
                y_att.append(float(arr[3]))
            elif arr[2] == "importance":
                y_imp.append(float(arr[3]))
            else:
                y_sco.append(float(arr[3]))

n_plots = len(labels)
plt.figure(figsize=(4.8*n_plots, 4.2))

for i in range(n_plots):
    plt.subplot(1, n_plots, i + 1)
    data = {'x': np.concatenate((np.array(x), np.array(x), np.array(x), np.array(x))),
            'y': np.concatenate((np.array(y_ori), np.array(y_att_all[i]), np.array(y_imp_all[i]), np.array(y_sco_all[i]))),
            'method': ["original"] * len(x) + ["attention"] * len(x) + ["importance"] * len(x) + ["d2nn"] * len(x)}
    df = pd.DataFrame(data)

    ax = sns.lineplot(x='x', y='y', hue='method', data=df)
    ax.spines['top'].set_color('black')
    ax.spines['bottom'].set_color('black')
    ax.spines['left'].set_color('black')
    ax.spines['right'].set_color('black')

    plt.xlabel('Error rate (%)', fontsize=18, fontweight='bold')
    plt.ylabel('Accuracy (%)', fontsize=18, fontweight='bold')
    plt.title(str(labels[i]) + ' duplications', fontsize=24, fontweight='bold')
    plt.tick_params(labelsize=18)
    plt.ylim(0, 0.8)

plt.tight_layout()
#plt.show()
plt.savefig('result_touch_1.pdf')
