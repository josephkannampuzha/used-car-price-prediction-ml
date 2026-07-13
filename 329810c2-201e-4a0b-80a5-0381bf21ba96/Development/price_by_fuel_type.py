import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# Normalize Fuel_Type: standardize casing and fix known typos
_fuel_map = {
    "diesel": "Diesel", " diesel": "Diesel",
    "petrol": "Petrol", " petrol": "Petrol", "petrol": "Petrol", "PETROL": "Petrol",
    "electric": "Electric", "electrik": "Electric",
    "hybrid": "Hybrid", "hybridd": "Hybrid",
    "cng": "CNG", "CNG": "CNG",
}
_fuel_clean = car_df["Fuel_Type"].str.strip().str.title().replace({
    "Petrol": "Petrol", "Diesel": "Diesel", "Electric": "Electric",
    "Hybrid": "Hybrid", "Cng": "CNG",
    # catch remaining raw oddities via direct map
    "Electrik": "Electric", "Hybridd": "Hybrid", "Petrol": "Petrol",
})

_fuel_df = car_df.copy()
_fuel_df["Fuel_Type_Clean"] = _fuel_clean

# Drop rows with missing fuel type for analysis
_fuel_df = _fuel_df.dropna(subset=["Fuel_Type_Clean"])

# Mean price by fuel type (printed summary)
mean_prices = _fuel_df.groupby("Fuel_Type_Clean")["Price"].mean().sort_values(ascending=False)
print("Mean Price by Fuel Type:")
print(mean_prices.apply(lambda x: f"${x:,.0f}").to_string())

# Boxplot — sample for performance (1M rows is heavy for boxplot rendering)
_SAMPLE_N = 50_000
_sample = _fuel_df.sample(n=_SAMPLE_N, random_state=42)

# Zerve dark-theme colours
_BG = "#1D1D20"
_TEXT = "#fbfbff"
_PALETTE = ["#A1C9F4", "#FFB482", "#8DE5A1", "#FF9F9B", "#D0BBFF", "#C49C94"]

_fuel_categories = sorted(_sample["Fuel_Type_Clean"].unique())
_color_map = {ft: _PALETTE[i % len(_PALETTE)] for i, ft in enumerate(_fuel_categories)}

fuel_price_fig, ax = plt.subplots(figsize=(11, 6))
fuel_price_fig.patch.set_facecolor(_BG)
ax.set_facecolor(_BG)

# Build per-group data lists for boxplot
_grouped_data = [_sample[_sample["Fuel_Type_Clean"] == ft]["Price"].values for ft in _fuel_categories]
_bp = ax.boxplot(
    _grouped_data,
    patch_artist=True,
    medianprops=dict(color="#ffd400", linewidth=2),
    whiskerprops=dict(color=_TEXT, linewidth=1.2),
    capprops=dict(color=_TEXT, linewidth=1.2),
    flierprops=dict(marker="o", markersize=2, alpha=0.3, markeredgewidth=0),
)

for patch, ft in zip(_bp["boxes"], _fuel_categories):
    patch.set_facecolor(_color_map[ft])
    patch.set_alpha(0.75)
    patch.set_edgecolor(_TEXT)

# Scale Y axis to millions for readability
_yticks = ax.get_yticks()
ax.set_yticklabels([f"₹{v/1e6:.1f}M" if v >= 0 else "" for v in _yticks], color=_TEXT, fontsize=10)
ax.set_xticks(range(1, len(_fuel_categories) + 1))
ax.set_xticklabels(_fuel_categories, color=_TEXT, fontsize=11)

ax.set_title("Price Distribution by Fuel Type", color=_TEXT, fontsize=14, pad=14)
ax.set_xlabel("Fuel Type", color=_TEXT, fontsize=11, labelpad=8)
ax.set_ylabel("Price", color=_TEXT, fontsize=11, labelpad=8)

for _spine in ax.spines.values():
    _spine.set_edgecolor("#909094")

ax.tick_params(colors=_TEXT)
ax.yaxis.grid(True, color="#909094", linestyle="--", alpha=0.4)
ax.set_axisbelow(True)

plt.tight_layout()
plt.show()
print(f"\n(Boxplot based on {_SAMPLE_N:,} randomly sampled records)")
