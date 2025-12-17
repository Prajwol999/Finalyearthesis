import pandas as pd
import matplotlib.pyplot as plt
from mplsoccer import VerticalPitch
import seaborn as sns

# 1. Load Data
print("Loading shot data...")
df = pd.read_csv("Premier_League_Shots_14_24_Direct.csv")

# 2. Filter for the Start vs End of the Decade
df_2014 = df[df['season'] == '2014/2015']
df_2023 = df[df['season'] == '2023/2024']

print(f"2014/15 Shots: {len(df_2014)}")
print(f"2023/24 Shots: {len(df_2023)}")

# 3. Setup the Pitch
# We create a simple figure with 2 subplots side-by-side
pitch = VerticalPitch(
    pitch_type='opta', 
    pitch_color='white', 
    line_color='black',
    line_zorder=2
)

# grid() returns a dictionary of axes. 'pitch' contains the plotting areas.
fig, axs = pitch.grid(nrows=1, ncols=2, title_height=0.05, figheight=10)

# --- Plot 1: 2014/15 (Red) ---
# We use the 'pitch' axes list directly
pitch.kdeplot(
    df_2014['X'] * 100, 
    df_2014['Y'] * 100, 
    ax=axs['pitch'][0],  # Left Plot
    fill=True,           # Updated from 'shade'
    levels=100, 
    cmap='Reds', 
    alpha=0.8,
    thresh=0.05
)
# Set title directly on the pitch axis
axs['pitch'][0].set_title("2014/15 Season\n(More Long Range Attempts)", fontsize=18, fontweight='bold', pad=15)

# --- Plot 2: 2023/24 (Blue) ---
pitch.kdeplot(
    df_2023['X'] * 100, 
    df_2023['Y'] * 100, 
    ax=axs['pitch'][1],  # Right Plot
    fill=True,           # Updated from 'shade'
    levels=100, 
    cmap='Blues', 
    alpha=0.8,
    thresh=0.05
)
# Set title directly on the pitch axis
axs['pitch'][1].set_title("2023/24 Season\n(Concentrated in the Box)", fontsize=18, fontweight='bold', pad=15)

# 4. Save
plt.savefig("Dissertation_Chart_4_HeatmapComparison.png", dpi=300, bbox_inches='tight')
print("\nSUCCESS! Heatmap saved as 'Dissertation_Chart_4_HeatmapComparison.png'")