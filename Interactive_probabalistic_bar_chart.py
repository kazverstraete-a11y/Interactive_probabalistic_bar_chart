import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
import matplotlib.colors as mcolors
from matplotlib.widgets import Slider
import scipy.stats as stats
from mpl_toolkits.axes_grid1 import make_axes_locatable 
import streamlit as st

#--- Magic Numbers ---
CONFIDENCE_LEVEL = 0.95
INITIAL_THRESHOLD = 40000
RANDOM_SEED = 12345

#data 
np.random.seed(RANDOM_SEED) #SET SEED: USE THE SAME RANDOM SEQUENCE FOR ALL SAMPLES

df = pd.DataFrame([np.random.normal(32000,200000,3650), 
                   np.random.normal(43000,100000,3650), 
                   np.random.normal(43500,140000,3650), 
                   np.random.normal(48000,70000,3650)], 
                  index=[1992,1993,1994,1995])


df['y_mean'] = df.mean(axis=1)
df['y_sem'] = df.sem(axis=1)


df['ci_val'] = df.apply(lambda row: stats.norm.interval(
    CONFIDENCE_LEVEL, 
    loc=row.iloc[:3650].mean(),
    scale=stats.sem(row.iloc[:3650])), 
    axis=1
)


yerr_low = df['y_mean'] - df['ci_val'].str[0]
yerr_upper = df['ci_val'].str[1] - df['y_mean']
asymmetric_error = [yerr_low.values, yerr_upper.values]

#--- Plot Setup ---
fig, ax = plt.subplots()
plt.subplots_adjust(left = 0.1, bottom=0.15, right=0.9, top=0.9)
ax.set_xlim(1991.5, 1995.5) 
threshold_line = ax.axhline(y=INITIAL_THRESHOLD, color='gray', linestyle='-', linewidth=2, alpha=0.8)
threshold_label = ax.text(
    1991.8, INITIAL_THRESHOLD, f'{INITIAL_THRESHOLD:.0f}', 
    ha='right', va='bottom', color='gray', fontsize=10,
    transform=ax.get_yaxis_transform(), clip_on=True
)

#initial plot
bars = ax.bar(df.index.values, df['y_mean'], yerr=asymmetric_error, capsize=10, color='grey')
plt.xticks(df.index.values, fontsize=11)
plt.tick_params(axis='x', bottom=False)
plt.yticks(fontsize=12)

ax.set_ylabel('')
ax.set_title('Annual Means vs. Threshold Probability', fontsize='x-large')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

#--- Slider Setup ---
ax_slider = plt.axes([0.15, 0.05, 0.65, 0.03])
threshold_slider = Slider(
    ax = ax_slider,
    label = 'Threshold Y',
    valmin = df['y_mean'].min() - df['y_sem'].max() * 3,
    valmax = df['y_mean'].max() + df['y_sem'].max() * 1.2,
    valinit = INITIAL_THRESHOLD
)

#--- Colormap Setup ---
#mpl.colormaps returns a colormap object (i.e. a function that for any given value between 0.0 and 1.0 returns an rgba)
cmap = mpl.colormaps['RdBu_r']
norm_prob = mcolors.Normalize(vmin=0, vmax=1)

def compute_probabilities(threshold, means, sems):
    return stats.norm.sf(threshold, loc=means, scale=sems)

def update(val):
    threshold_value = threshold_slider.val
    #probability that distribution value is greater than threshold value
    probabilities_greater = compute_probabilities(threshold_value, df['y_mean'], df['y_sem'])
    #normalize probabilities, search for corresponding rgba
    new_colors = cmap(norm_prob(probabilities_greater))
    #implement color for each bar
    for bar, color, in zip(bars, new_colors):
        bar.set_color(color)
    #update threshold label
    threshold_line.set_ydata([threshold_value, threshold_value])
    threshold_label.set_text(f'{threshold_value:.0f}')
    threshold_label.set_position((0.05, threshold_value))
    
    fig.canvas.draw_idle()

threshold_slider.on_changed(update)
update(threshold_slider.valinit)

#add colorbar
sm = ScalarMappable(norm = norm_prob, cmap=cmap)
sm.set_array([])

divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="2%", pad=0.15)

cb = fig.colorbar(sm, cax=cax,label='Distribution > Threshold probability (%)')
cb.ax.tick_params(labelsize=8)
cb.ax.set_ylabel('Distribution > Threshold probability (%)', fontsize=12, labelpad=10)

#--- Show ---
plt.show()
st.pyplot(fig)

