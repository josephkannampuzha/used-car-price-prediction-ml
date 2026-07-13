import matplotlib.pyplot as plt

# Mean price by transmission type
_trans_mean = car_df.groupby("Transmission")["Price"].mean().sort_values(ascending=False)
print("Mean Price by Transmission Type:")
print(_trans_mean.apply(lambda x: f"${x:,.0f}").to_string())

# Sample for performance
_SAMPLE_N = 50_000
_sample = car_df.dropna(subset=["Transmission"]).sample(n=_SAMPLE_N, random_state=42)

# Zerve dark-theme colours
_BG = "#1D1D20"
_TEXT = "#fbfbff"
_PALETTE = ["#A1C9F4", "#FFB482", "#8DE5A1", "#FF9F9B", "#D0BBFF"]

_trans_categories = sorted(_sample["Transmission"].unique())
_color_map = {t: _PALETTE[i % len(_PALETTE)] for i, t in enumerate(_trans_categories)}

transmission_price_fig, ax = plt.subplots(figsize=(9, 6))
transmission_price_fig.patch.set_facecolor(_BG)
ax.set_facecolor(_BG)

_grouped_data = [_sample[_sample["Transmission"] == t]["Price"].values for t in _trans_categories]
_bp = ax.boxplot(
    _grouped_data,
    patch_artist=True,
    medianprops=dict(color="#ffd400", linewidth=2),
    whiskerprops=dict(color=_TEXT, linewidth=1.2),
    capprops=dict(color=_TEXT, linewidth=1.2),
    flierprops=dict(marker="o", markersize=2, alpha=0.3, markeredgewidth=0),
)

for patch, t in zip(_bp["boxes"], _trans_categories):
    patch.set_facecolor(_color_map[t])
    patch.set_alpha(0.75)
    patch.set_edgecolor(_TEXT)

# Scale Y axis to millions
_yticks = ax.get_yticks()
ax.set_yticklabels([f"₹{v/1e6:.1f}M" if v >= 0 else "" for v in _yticks], color=_TEXT, fontsize=10)
ax.set_xticks(range(1, len(_trans_categories) + 1))
ax.set_xticklabels(_trans_categories, color=_TEXT, fontsize=12)

ax.set_title("Price Distribution by Transmission Type", color=_TEXT, fontsize=14, pad=14)
ax.set_xlabel("Transmission", color=_TEXT, fontsize=11, labelpad=8)
ax.set_ylabel("Price", color=_TEXT, fontsize=11, labelpad=8)

for _spine in ax.spines.values():
    _spine.set_edgecolor("#909094")

ax.tick_params(colors=_TEXT)
ax.yaxis.grid(True, color="#909094", linestyle="--", alpha=0.4)
ax.set_axisbelow(True)

plt.tight_layout()
plt.close("all")
print(f"\n(Boxplot based on {_SAMPLE_N:,} randomly sampled records)")
