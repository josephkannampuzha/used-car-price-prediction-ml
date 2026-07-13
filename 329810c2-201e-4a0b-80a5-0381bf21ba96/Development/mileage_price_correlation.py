import matplotlib.pyplot as plt
import numpy as np

# Sample for scatter plot performance
_sample = car_df.sample(n=5000, random_state=42)

# Compute correlation
_corr = car_df["Kms_Driven"].corr(car_df["Price"])
print(f"Pearson correlation (Kms_Driven vs Price): {_corr:.4f}")

# Plot
mileage_price_fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(
    _sample["Kms_Driven"],
    _sample["Price"],
    alpha=0.3,
    color="#A1C9F4",
    edgecolors="none",
    s=12,
)

# Trend line
_z = np.polyfit(_sample["Kms_Driven"], _sample["Price"], 1)
_p = np.poly1d(_z)
_x_line = np.linspace(_sample["Kms_Driven"].min(), _sample["Kms_Driven"].max(), 200)
ax.plot(_x_line, _p(_x_line), color="#FFB482", linewidth=1.5, label=f"Trend  (r = {_corr:.3f})")

ax.set_title("Mileage vs Price", color="#fbfbff", fontsize=14, pad=12)
ax.set_xlabel("Kms Driven", color="#909094")
ax.set_ylabel("Price", color="#909094")
ax.tick_params(colors="#909094")
ax.set_facecolor("#1D1D20")
mileage_price_fig.patch.set_facecolor("#1D1D20")
for spine in ax.spines.values():
    spine.set_edgecolor("#909094")
ax.legend(facecolor="#1D1D20", labelcolor="#fbfbff")

plt.tight_layout()
