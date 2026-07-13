import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# --- Compute correlation matrix on numeric columns ---
_numeric_df = car_df.select_dtypes(include=["int64", "float64"])
_corr = _numeric_df.corr()

# Zerve dark-theme colours
_BG   = "#1D1D20"
_TEXT = "#fbfbff"

# --- Heatmap figure ---
corr_heatmap_fig, ax = plt.subplots(figsize=(12, 9))
corr_heatmap_fig.patch.set_facecolor(_BG)
ax.set_facecolor(_BG)

sns.heatmap(
    _corr,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    linewidths=0.5,
    linecolor="#333336",
    ax=ax,
    annot_kws={"size": 8, "color": _TEXT},
    cbar_kws={"shrink": 0.8},
)

# Style colorbar
_cbar = ax.collections[0].colorbar
_cbar.ax.yaxis.set_tick_params(color=_TEXT)
_cbar.outline.set_edgecolor(_TEXT)
plt.setp(_cbar.ax.yaxis.get_ticklabels(), color=_TEXT)

ax.set_title("Feature Correlation Heatmap", color=_TEXT, fontsize=14, pad=14)
ax.tick_params(axis="x", colors=_TEXT, labelsize=9, rotation=45)
ax.tick_params(axis="y", colors=_TEXT, labelsize=9, rotation=0)

plt.tight_layout()
plt.close("all")

# --- Price correlations summary ---
_price_corr = _corr["Price"].drop("Price").sort_values(ascending=False)
print("Correlations with Price (sorted):")
print(_price_corr.to_string())
