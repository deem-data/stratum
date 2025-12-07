import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read the CSV file
df = pd.read_csv('bench_cse_tfidf_gridsearch.csv')

# Get unique schedulers and n_rows
schedulers = df['scheduler'].unique()
n_rows_values = sorted(df['n_rows'].unique())

# Set up the plot
fig, ax = plt.subplots(figsize=(12, 6))

# Define colors for each scheduler (consistent across groups)
colors = {
    'skrub-single': '#1f77b4',  # blue
    'skrub-multi': '#ff7f0e',   # orange
    'stratum-single': '#2ca02c'  # green
}

# Set up bar positions
x = np.arange(len(n_rows_values))
width = 0.25  # Width of bars
spacing = 0.05  # Spacing between groups

# Plot bars for each scheduler
for i, scheduler in enumerate(schedulers):
    values = []
    for n_rows in n_rows_values:
        row = df[(df['n_rows'] == n_rows) & (df['scheduler'] == scheduler)]
        if not row.empty:
            values.append(row['total'].iloc[0])
        else:
            values.append(0)
    
    # Calculate x positions for this scheduler's bars
    offset = (i - len(schedulers) / 2 + 0.5) * (width + spacing)
    bars = ax.bar(x + offset, values, width, label=scheduler, color=colors[scheduler])
    
    # Add value labels on top of bars
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height,
                f'{value:.3f}',
                ha='center', va='bottom', fontsize=9)

# Customize the plot
ax.set_xlabel('Number of Rows', fontsize=12)
ax.set_ylabel('Total Time (seconds)', fontsize=12)
ax.set_title('Benchmark Results: CSE TF-IDF Grid Search', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(n_rows_values)
ax.legend(title='Scheduler', fontsize=10)
ax.grid(axis='y', alpha=0.3, linestyle='--')

plt.tight_layout()
plt.savefig('bench_cse_tfidf_gridsearch_bar_plot.pdf', dpi=300, bbox_inches='tight')
plt.show()

