import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

base_path = "benchmarks/logical_optimizer/end-to-end/california-housing/"
data = pd.read_csv(base_path + "california_housing_pipelines_benchmark.csv", sep=";")
data["time"] = data["time"].apply(np.round, decimals=2)

# Prepare data in desired order: non-optimized, optimized
non_optimized = data.iloc[:3]["time"].values
optimized = data.iloc[3:6]["time"].values

labels = ["skrub-njobs=1", "skrub-njobs=-1", "stratum-njobs=1"]
data = pd.DataFrame({"non_optimized": non_optimized, "optimized": optimized, "labels": labels})

# Publication-quality colorblind-friendly colors
# Using a 4-color palette: blue, orange, green, red (ColorBrewer inspired)
pub_colors = ['#F18F01', '#C73E1D', '#6A994E']  # Blue, Orange, Red, Green
exp_names = ("w/o Logical Rewrites", "w/ Logical Rewrites")
x = np.arange(len(exp_names)) # the label locations
width = 0.25  # the width of the bars

multiplier = 0

fig, ax = plt.subplots(figsize=(5, 5), dpi=100, layout='constrained')
for i, row in data.iterrows():
    offset = width * multiplier
    rects = ax.bar(x + offset, row[:2], width=width, label=labels[i], color=pub_colors[i])
    ax.bar_label(rects, padding=3)
    multiplier += 1

ax.set_xticks(x + width, exp_names)
ax.set_yscale("log")
ax.set_ylabel("Time (s)")
ax.legend(loc="upper right", ncols=2)
plt.ylim(0.01, 30)
ax.grid(axis='y', alpha=0.3, linestyle='--')
plt.tight_layout()
plt.savefig(base_path + "california_housing_pipelines_benchmark_bar_plot.pdf")