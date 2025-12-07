import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

data = pd.read_csv("benchmarks/logical_optimizer/end-to-end/california-housing/california_housing_pipelines_benchmark.csv")

# make an bar plot example
plt.figure(figsize=(10, 4))

# Prepare data in desired order: baseline, non-optimized, optimized
baseline = data.tail(1)
non_optimized = data.iloc[:3]
optimized = data.iloc[3:6]

# Combine in order for plotting
plot_data = pd.concat([baseline, non_optimized, optimized], ignore_index=True)

# Publication-quality colorblind-friendly colors
# Using a 4-color palette: blue, orange, green, red (ColorBrewer inspired)
pub_colors = ['#2E86AB', '#F18F01', '#C73E1D', '#6A994E']  # Blue, Orange, Red, Green

# Assign colors: baseline gets first color, then cycle through for the rest
colors_list = [pub_colors[0]]  # baseline
colors_list.extend([pub_colors[1], pub_colors[2], pub_colors[3]])  # non-optimized (3 bars)
colors_list.extend([pub_colors[1], pub_colors[2], pub_colors[3]])  # optimized (3 bars)

# Plot all bars
bars = []
for i, row in plot_data.iterrows():
    bar = plt.bar(row['impl'], row['time'], color=colors_list[i])
    bars.append(bar)
    # Add time label on top of bar
    plt.text(row['impl'], row['time'], 
             f"{row['time']:.3f}", 
             ha='center', va='bottom', fontsize=9)

# Get x-axis positions for separators
ax = plt.gca()
patches = ax.patches

# Indices: baseline (0), non-optimized (1-3), optimized (4-6)
baseline_rect = patches[0]
non_opt_end_rect = patches[3]  # Last non-optimized bar
opt_start_rect = patches[4]    # First optimized bar

# Draw separator after baseline (between baseline and non-optimized)
baseline_right = baseline_rect.get_x() + baseline_rect.get_width()
non_opt_start_left = patches[1].get_x()
separator_x1 = (baseline_right + non_opt_start_left) / 2
ax.axvline(x=separator_x1, color='gray', linestyle='--', linewidth=1, alpha=0.5)

# Draw separator after non-optimized (between non-optimized and optimized)
non_opt_end_right = non_opt_end_rect.get_x() + non_opt_end_rect.get_width()
opt_start_left = opt_start_rect.get_x()
separator_x2 = (non_opt_end_right + opt_start_left) / 2
ax.axvline(x=separator_x2, color='gray', linestyle='--', linewidth=1, alpha=0.5)

# Add group labels below each section
# For log scale, position text at a small value below the visible range
y_bottom = 0.01  # Position text below the bars (small value for log scale)

# Baseline label (centered under baseline bar)
baseline_center_x = baseline_rect.get_x() + baseline_rect.get_width() / 2
plt.text(baseline_center_x, y_bottom, 'baseline', 
         ha='center', va='top', fontsize=10, fontweight='bold')

# Non-optimized label (centered under non-optimized bars)
non_opt_start_rect = patches[1]
non_opt_center_x = (non_opt_start_rect.get_x() + non_opt_end_rect.get_x() + non_opt_end_rect.get_width()) / 2
plt.text(non_opt_center_x, y_bottom, 'w/out logical rewrites', 
         ha='center', va='top', fontsize=10, fontweight='bold')

# Optimized label (centered under optimized bars)
opt_end_rect = patches[6]
opt_center_x = (opt_start_rect.get_x() + opt_end_rect.get_x() + opt_end_rect.get_width()) / 2
plt.text(opt_center_x, y_bottom, 'w/ logical rewrites', 
         ha='center', va='top', fontsize=10, fontweight='bold')

# Ensure x-axis labels are visible
plt.xticks(rotation=15, ha='right')
plt.xlabel('Scheduler', labelpad=50)
plt.ylabel('Time in seconds')
plt.yscale('log')
plt.ylim(0.000, 6)
plt.tight_layout()

plt.savefig('california_housing_pipelines_benchmark_bar_plot.pdf')