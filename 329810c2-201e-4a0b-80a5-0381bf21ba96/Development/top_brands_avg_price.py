import matplotlib.pyplot as plt
import numpy as np

brand_value = (
    car_df.groupby("Brand")["Price"]
    .mean()
    .sort_values(ascending=False)
    .head(10)
)

# Strip leading/trailing whitespace from brand names (deduplicate variants like " BMW" vs "BMW")
brand_value.index = brand_value.index.str.strip()
brand_value = brand_value.groupby(level=0).mean().sort_values(ascending=False).head(10)

print("Top 10 Brands by Average Price:")
print(brand_value.apply(lambda x: f"${x:,.0f}").to_string())

# Scale to millions for readability
values_m = brand_value.values / 1e6
labels = brand_value.index.tolist()

brand_avg_fig, ax = plt.subplots(figsize=(10, 6))
brand_avg_fig.patch.set_facecolor("#1D1D20")
ax.set_facecolor("#1D1D20")

colors = ["#A1C9F4"] * len(labels)
bars = ax.barh(labels[::-1], values_m[::-1], color=colors, height=0.6)

# Add value labels inside bars
for bar, val in zip(bars, values_m[::-1]):
    ax.text(
        bar.get_width() - 0.05, bar.get_y() + bar.get_height() / 2,
        f"${val:.2f}M",
        va="center", ha="right", color="#1D1D20", fontsize=9, fontweight="bold"
    )

ax.set_title("Top 10 Brands by Average Price", color="#fbfbff", fontsize=13, pad=12)
ax.set_xlabel("Average Price (Millions)", color="#909094", fontsize=10)
ax.tick_params(colors="#fbfbff", labelsize=9)
ax.xaxis.label.set_color("#909094")

for spine in ax.spines.values():
    spine.set_visible(False)
ax.tick_params(left=False, bottom=False)
ax.xaxis.set_ticks_position("none")

plt.tight_layout()
plt.show()
