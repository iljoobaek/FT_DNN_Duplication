import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns; sns.set()

sns.set_style('whitegrid')

path = "results_map_vs_layer.txt"
# x = [0.1, 0.3, 0.5, 0.7, 0.9, 0.92, 0.94, 0.96, 0.98, 1]
# x = [0.1, 0.3, 0.5, 0.7, 0.9, 1]
width = 0
error = 0.9

# y_att_all = []
# y_imp_all = []
# y_sco_all = []
labels = []
y_ori = []
y_att = []
y_imp = []
y_sco = []
y_rad = []
with open(path, 'r') as f:
    for line in f:
        if "Layer" in line:
            width += 1
            continue
        arr = line.strip().split("|")
        if arr[0] == str(error):
            if arr[1] == "False":
                y_ori.append(float(arr[3]))
            else:
                if arr[2] == "attention":
                    y_att.append(float(arr[3]))
                elif arr[2] == "importance":
                    y_imp.append(float(arr[3]))
                elif arr[2] == "random":
                    y_rad.append(float(arr[3]))
                else:
                    y_sco.append(float(arr[3]))

n_plots = 1
plt.figure(figsize=(4.8, 4.2))
x = np.arange(1, width + 1)

for i in range(n_plots):
    plt.subplot(1, n_plots, i + 1)
    data = {'x': np.concatenate((np.array(x), np.array(x), np.array(x), np.array(x), np.array(x))),
            'y': np.concatenate((np.array(y_ori), np.array(y_att), np.array(y_imp), np.array(y_sco), np.array(y_rad))),
            'method': ["original"] * len(x) + ["attention"] * len(x) + ["importance"] * len(x) + ["d2nn"] * len(x) + ["random"] * len(x)}
    # data = {'x': np.concatenate((np.array(x), np.array(x), np.array(x), np.array(x))),
    #         'y': np.concatenate((np.array(y_ori), np.array(y_imp), np.array(y_sco), np.array(y_rad))),
    #         'method': ["original"] * len(x) + ["importance"] * len(x) + ["d2nn"] * len(x) + ["random"] * len(x)}

    df = pd.DataFrame(data)

    ax = sns.lineplot(x='x', y='y', hue='method', data=df)
    ax.spines['top'].set_color('black')
    ax.spines['bottom'].set_color('black')
    ax.spines['left'].set_color('black')
    ax.spines['right'].set_color('black')

    plt.xlabel('Layer Index', fontsize=18, fontweight='bold')
    plt.ylabel('Accuracy (%)', fontsize=18, fontweight='bold')
    plt.title('Feature Map Error=' + str(error), fontsize=24, fontweight='bold')
    plt.tick_params(labelsize=18)
    plt.ylim(0, 0.8)

plt.tight_layout()
#plt.show()
plt.savefig('results_map_vs_layer.pdf')
