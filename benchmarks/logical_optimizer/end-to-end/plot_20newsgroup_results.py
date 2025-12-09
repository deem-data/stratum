import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

base_path = "benchmarks/logical_optimizer/end-to-end/"

data = pd.read_csv(base_path + 'bench_cse_tfidf_gridsearch.csv')
data["total"] = data["total"].apply(np.round, decimals=2)

labels = ["skrub-njobs=1", "skrub-njobs=-1", "stratum-njobs=1"]
exp_names = (100, 500, 1000, 1000)

# Publication-quality colorblind-friendly colors
# Using a 4-color palette: blue, orange, green, red (ColorBrewer inspired)
pub_colors = ['#F18F01', '#C73E1D', '#6A994E']  # Blue, Orange, Red, Green
exp_names = (100, 500, 1000, 1000)
x = np.arange(len(exp_names)) # the label locations
width = 0.85  # the width of the bars
x = x* width*(len(labels)+1)
multiplier = 0

fig, ax = plt.subplots(figsize=(9, 5), dpi=100)
for scheduler, group in data.groupby("scheduler"):
    offset = width * multiplier
    rects = ax.bar(x + offset, group["total"], width=width, label=scheduler, color=pub_colors[multiplier])
    ax.bar_label(rects, padding=3)
    multiplier += 1

ax.set_xticks(x + width, exp_names)
ax.set_yscale("log")
ax.set_ylabel("Time (s)")
ax.legend(loc="upper right", ncols=len(labels))
plt.ylim(0.1, 300)
ax.grid(axis='y', alpha=0.3, linestyle='--')
plt.tight_layout()
plt.savefig(base_path + "20newsgroup_results_bar_plot.pdf")