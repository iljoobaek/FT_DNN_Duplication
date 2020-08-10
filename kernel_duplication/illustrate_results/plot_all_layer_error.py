import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns; sns.set()

sns.set_style('whitegrid')

path = "results_all_layer_1.txt"
# x = [0.1, 0.3, 0.5, 0.7, 0.9, 0.92, 0.94, 0.96, 0.98, 1]
# x = [0.1, 0.3, 0.5, 0.7, 0.9, 1]
width = 0
# error = 0.9

# y_att_all = []
y_ori_all = []
y_imp_all = []
y_sco_all = []
y_rad_all = []
labels = []
y_ori = []
# y_att = []
y_imp = []
y_sco = []
y_rad = []
with open(path, 'r') as f:
    for line in f:
        if "error" in line:
            # y_att_all.append(y_att)
            y_ori_all.append(y_ori)
            y_imp_all.append(y_imp)
            y_sco_all.append(y_sco)
            y_rad_all.append(y_rad)
            # y_att = []
            y_ori = []
            y_imp = []
            y_sco = []
            y_rad = []
            width += 1
            continue
        arr = line.strip().split("|")
        # if arr[0] == str(error):
        if arr[1] == "False":
            y_ori.append(float(arr[3]))
        else:
            # if arr[2] == "attention":
            #     y_att.append(float(arr[3]))
            if arr[2] == "importance":
                y_imp.append(float(arr[3]))
            elif arr[2] == "random":
                y_rad.append(float(arr[3]))
            elif arr[2] == "d2nn":
                y_sco.append(float(arr[3]))

n_plots = 3
plt.figure(figsize=(4.8 * n_plots, 4.2))
# x = np.arange(1, width + 1)
x = np.arange(1, 11, 1)
idx = {0: 0, 1: 1, 2: 2}
ori = np.array(y_ori_all)
imp = np.array(y_imp_all)
sco = np.array(y_sco_all)
rad = np.array(y_rad_all)
# print(ori.shape, imp.shape, sco.shape, rad.shape)

for i in range(n_plots):
    plt.subplot(1, n_plots, i + 1)
    data = {'x': np.concatenate((np.array(x), np.array(x), np.array(x), np.array(x))),
            'y': np.concatenate((ori[idx[i]], imp[idx[i]], sco[idx[i]], rad[idx[i]])),
            'Duplication Type': ["none"] * len(x) + ["2nd derivative"] * len(x) + ["weight sum"] * len(x) + ["random"] * len(x)}

    df = pd.DataFrame(data)

    ax = sns.lineplot(x='x', y='y', hue='Duplication Type', data=df)
    ax.spines['top'].set_color('black')
    ax.spines['bottom'].set_color('black')
    ax.spines['left'].set_color('black')
    ax.spines['right'].set_color('black')

    plt.xlabel('Error Rate (%)', fontsize=18, fontweight='bold')
    plt.ylabel('Accuracy', fontsize=18, fontweight='bold')
    plt.title('Varied Feature Map Error\n Weight Error=' + str((idx[i] + 1)) + '%', fontsize=18, fontweight='bold')
    plt.tick_params(labelsize=18)
    plt.ylim(0, 0.7)

plt.tight_layout()
#plt.show()
plt.savefig('results_all_layer.pdf')
