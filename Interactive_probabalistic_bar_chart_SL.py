import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
import matplotlib.colors as mcolors
import scipy.stats as stats
from mpl_toolkits.axes_grid1 import make_axes_locatable
import streamlit as st


#--- Markdown ---
st.markdown(
"""
This visualization shows annual means with 95% confidence intervals.
Bar color encodes the probability that the underlying distribution
exceeds a user-defined threshold.
"""
)

# --- Magic Numbers ---
CONFIDENCE_LEVEL = 0.95
INITIAL_THRESHOLD = 40000
RANDOM_SEED = 12345

#---Set
st.set_page_config(page_title="Probabilistic Bar Chart", layout="centered")
st.title("Annual Means vs. Threshold Probability")

# --- Data ---
@st.cache_data
def generate_data(confidence_level, random_seed):
    np.random.seed(random_seed)

    df = pd.DataFrame(
        [
            np.random.normal(32000, 200000, 3650),
            np.random.normal(43000, 100000, 3650),
            np.random.normal(43500, 140000, 3650),
            np.random.normal(48000, 70000, 3650),
        ],
        index=[1992, 1993, 1994, 1995],
    )

    df["y_mean"] = df.mean(axis=1)
    df["y_sem"] = df.sem(axis=1)
    
    df["ci_val"] = df.apply(
        lambda row: stats.norm.interval(
            CONFIDENCE_LEVEL,
            loc=row.iloc[:3650].mean(),
            scale=stats.sem(row.iloc[:3650]),
        ),
        axis=1,
    )
    return df

df = generate_data(CONFIDENCE_LEVEL, RANDOM_SEED)

yerr_low = df["y_mean"] - df["ci_val"].str[0]
yerr_upper = df["ci_val"].str[1] - df["y_mean"]
asymmetric_error = [yerr_low.values, yerr_upper.values]

# --- Streamlit slider (replaces matplotlib Slider) ---
valmin = float(df["y_mean"].min() - df["y_sem"].max() * 3)
valmax = float(df["y_mean"].max() + df["y_sem"].max() * 1.2)

threshold = st.sidebar.slider(
    "Threshold Y",
    min_value=valmin,
    max_value=valmax,
    value=float(INITIAL_THRESHOLD),
    step=100.0,
)

# --- Colormap ---
cmap = mpl.colormaps["RdBu_r"]
norm_prob = mcolors.Normalize(vmin=0, vmax=1)

def compute_probabilities(threshold_value, means, sems):
    # survival function = P(X > threshold)
    return stats.norm.sf(threshold_value, loc=means, scale=sems)

probabilities_greater = compute_probabilities(threshold, df["y_mean"], df["y_sem"])
bar_colors = cmap(norm_prob(probabilities_greater))

# --- Plot ---
fig, ax = plt.subplots()
plt.subplots_adjust(left=0.1, bottom=0.15, right=0.9, top=0.9)

ax.set_xlim(1991.5, 1995.5)
ax.axhline(y=threshold, color="gray", linestyle="-", linewidth=2, alpha=0.8)
ax.text(
    0.05, threshold, f"{threshold:.0f}",
    transform=ax.get_yaxis_transform(),
    ha="left", va="bottom", color="gray", fontsize=10, clip_on=True
)

bars = ax.bar(
    df.index.values,
    df["y_mean"],
    yerr=asymmetric_error,
    capsize=10,
    color=bar_colors
)

ax.set_title("Annual Means vs. Threshold Probability", fontsize="x-large")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.tick_params(axis="x", bottom=False)
plt.xticks(df.index.values, fontsize=11)
plt.yticks(fontsize=12)

# Colorbar
sm = ScalarMappable(norm=norm_prob, cmap=cmap)
sm.set_array([])

divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="2%", pad=0.15)
cb = fig.colorbar(sm, cax=cax, label="Distribution > Threshold probability (%)")
cb.ax.tick_params(labelsize=8)
cb.ax.set_ylabel("Distribution > Threshold probability (%)", fontsize=12, labelpad=10)

st.pyplot(fig, clear_figure=True)


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

# Colorbar
sm = ScalarMappable(norm=norm_prob, cmap=cmap)
sm.set_array([])

divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="2%", pad=0.15)
cb = fig.colorbar(sm, cax=cax, label="Distribution > Threshold probability (%)")
cb.ax.tick_params(labelsize=8)
cb.ax.set_ylabel("Distribution > Threshold probability (%)", fontsize=12, labelpad=10)

st.pyplot(fig, clear_figure=True)
