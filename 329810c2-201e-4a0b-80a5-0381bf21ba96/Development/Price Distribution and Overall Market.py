import warnings
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

warnings.filterwarnings("ignore")

# Use car_df from upstream block (already loaded and cleaned)
price_series = car_df["Price"].dropna()

# Compute stats
_mean = price_series.mean()
_median = price_series.median()
_std = price_series.std()
_p5, _p95 = np.percentile(price_series, [5, 95])

# Plot histogram manually to avoid seaborn/pandas compatibility issue
fig, ax = plt.subplots(figsize=(11, 6))
fig.patch.set_facecolor("#1D1D20")
ax.set_facecolor("#1D1D20")

_counts, _bins, _patches = ax.hist(
    price_series,
    bins=60,
    color="#A1C9F4",
    edgecolor="#1D1D20",
    linewidth=0.4,
    alpha=0.9,
)

# Overlay a KDE curve
from scipy.stats import gaussian_kde
_kde = gaussian_kde(price_series, bw_method=0.15)
_x = np.linspace(price_series.min(), price_series.max(), 500)
_scale = _counts.max() / _kde(_x).max()
ax.plot(_x, _kde(_x) * _scale, color="#FFB482", linewidth=2, label="KDE")

# Reference lines
ax.axvline(_mean, color="#ffd400", linewidth=1.5, linestyle="--", label=f"Mean ${_mean:,.0f}")
ax.axvline(_median, color="#8DE5A1", linewidth=1.5, linestyle="--", label=f"Median ${_median:,.0f}")

ax.set_title("Distribution of Used Car Prices", color="#fbfbff", fontsize=15, pad=12)
ax.set_xlabel("Price (USD)", color="#909094", fontsize=12)
ax.set_ylabel("Count", color="#909094", fontsize=12)
ax.tick_params(colors="#909094")
for spine in ax.spines.values():
    spine.set_edgecolor("#909094")

ax.legend(facecolor="#2a2a2e", labelcolor="#fbfbff", fontsize=10)

plt.tight_layout()

print("Price Distribution Summary")
print("=" * 40)
print(f"Count   : {len(price_series):>12,.0f}")
print(f"Mean    : ${_mean:>12,.2f}")
print(f"Median  : ${_median:>12,.2f}")
print(f"Std Dev : ${_std:>12,.2f}")
print(f"5th pct : ${_p5:>12,.2f}")
print(f"95th pct: ${_p95:>12,.2f}")
print(f"Min     : ${price_series.min():>12,.2f}")
print(f"Max     : ${price_series.max():>12,.2f}")
