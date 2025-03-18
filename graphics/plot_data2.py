import matplotlib.pyplot as plt
import numpy as np

# Data from the table
years = np.arange(2024, 2012, -1)  # Years from 2024 to 2013

percentile_30 = [-4.830426,-1.276497,10.653436,0.246698,-1.69032,-2.110252,-13.225405,-0.618189,-3.560843,0.01761,-3.824895,-8.532096]
percentile_50 = [-3.115717,-0.821465,94.917032,12.814071,-0.99106,-0.566782,-8.223784,-0.399078,-1.728179,1.108181,-2.394891,-5.172715]
percentile_70 = [-1.662384,-0.410098,216.213027,30.403752,-0.159463,-0.004252,-3.446802,-0.217441,-0.38331,6.509843,-0.917961,-1.744052]

# Define y-axis limits
y_max_limit = 50  # Set limit based on data (adjust as needed)
y_min_limit = -15
# Create figure
plt.figure(figsize=(12, 6))


# Plot only within the limit
plt.plot(years, percentile_30, marker='o', linestyle='-', label="30th Percentile", color='orange')
plt.plot(years, percentile_50, marker='o', linestyle='-', label="50th Percentile", color='green')
plt.plot(years, percentile_70, marker='o', linestyle='-', label="70th Percentile", color='red')

# Adjust y-axis limit
plt.ylim(y_min_limit, y_max_limit)

# Labels, legend, and formatting
plt.xlabel("Year")
plt.ylabel("Sortino Ratio (1e-15)")
plt.xticks(rotation=45)
plt.grid(True)
plt.legend()

# Save the figure
plot_path = 'graphics/percentile_trends.png'
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
plt.close()
