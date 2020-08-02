import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns; sns.set()

sns.set_style('whitegrid')

path = "results_weight_feature.txt"
# x = [0.1, 0.3, 0.5, 0.7, 0.9, 0.92, 0.94, 0.96, 0.98, 1]
# x = [0.1, 0.3, 0.5, 0.7, 0.9, 1]
width = 0
error = 0.9

y_fea_all = []
y_wei_all = []
# y_sco_all = []
labels = []
y_fea = []
y_wei = []
# y_ori = []
# y_att = []
# y_imp = []
# y_sco = []
# y_rad = []
with open(path, 'r') as f:
    for line in f:
        if "type" in line:
            y_fea_all.append(y_fea)
            y_wei_all.append(y_wei)
            y_wei = []
            y_fea = []
            width += 1
            continue
        arr = line.strip().split("|")
        if arr[1] == "False":
            y_fea.append(float(arr[3]))
        else:
            y_wei.append(float(arr[3]))

n_plots = 2
plt.figure(figsize=(4.8 * n_plots, 4.2))
# x = np.arange(1, width + 1)
x = np.arange(0, 1.1, 0.1)
error_type = ["zero error", "random error"]

for i in range(n_plots):
    plt.subplot(1, n_plots, i + 1)
    data = {'x': np.concatenate((np.array(x), np.array(x), np.array(x))),
            'y': np.concatenate((np.array(y_fea_all[i]), np.array(y_wei_all[i]), np.array([y_wei_all[i][0]] * len(x)))),
            'type': ["error at feature map"] * len(x) + ["error at weight"] * len(x) + ["no error"] * len(x)}
    # data = {'x': np.concatenate((np.array(x), np.array(x), np.array(x), np.array(x))),
    #         'y': np.concatenate((np.array(y_ori), np.array(y_imp), np.array(y_sco), np.array(y_rad))),
    #         'method': ["original"] * len(x) + ["importance"] * len(x) + ["d2nn"] * len(x) + ["random"] * len(x)}

    df = pd.DataFrame(data)

    ax = sns.lineplot(x='x', y='y', hue='type', data=df)
    ax.spines['top'].set_color('black')
    ax.spines['bottom'].set_color('black')
    ax.spines['left'].set_color('black')
    ax.spines['right'].set_color('black')

    plt.xlabel('Error Rate (%)', fontsize=18, fontweight='bold')
    plt.ylabel('Accuracy (%)', fontsize=18, fontweight='bold')
    plt.title('Error Type: ' + error_type[i], fontsize=24, fontweight='bold')
    plt.tick_params(labelsize=18)
    plt.ylim(0, 0.8)

plt.tight_layout()
#plt.show()
plt.savefig('results_weight_feature.pdf')
