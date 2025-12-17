import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 1. Load your new data
df = pd.read_csv("Premier_League_Shots_14_24_Direct.csv")

# 2. Calculate Shot Distance (Pitch is 105m x 68m)
# X is 0-1, Y is 0-1.
df['distance_m'] = np.sqrt( ((1 - df['X']) * 105)**2 + ((0.5 - df['Y']) * 68)**2 )

# 3. Define "Long Range" (> 24 meters approx 26 yards)
df['is_long_range'] = df['distance_m'] > 24

# 4. Group by Season to find the % of Long Range Shots
trends = df.groupby('season')['is_long_range'].mean() * 100
trends = trends.reset_index()

print("--- Long Range Shot Percentage by Season ---")
print(trends)

# 5. Plot the Decline
plt.figure(figsize=(12, 6))
sns.lineplot(data=trends, x='season', y='is_long_range', marker='o', linewidth=3, color='red')
plt.title('The Decline of Long-Range Shots in the Premier League (2014-2024)', fontsize=16)
plt.ylabel('Percentage of Total Shots (%)', fontsize=12)
plt.xlabel('Season', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.xticks(rotation=45)
plt.tight_layout()

# Save the chart
plt.savefig("Dissertation_Chart_1_Decline.png", dpi=300)
print("\nChart saved as 'Dissertation_Chart_1_Decline.png'. Open it to see the proof!")