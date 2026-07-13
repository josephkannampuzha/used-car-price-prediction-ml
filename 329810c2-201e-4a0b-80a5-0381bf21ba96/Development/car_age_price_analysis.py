import matplotlib.pyplot as plt
import numpy as np

_sample = car_df.sample(10000, random_state=42)

# --- Scatter: Car Age vs Price ---
age_price_scatter, ax = plt.subplots(figsize=(10, 6))
age_price_scatter.patch.set_facecolor("#1D1D20")
ax.set_facecolor("#1D1D20")

ax.scatter(
    _sample["Car_Age"], _sample["Price"],
    alpha=0.25, s=8, color="#A1C9F4", label="Individual cars"
)

# Average price per age
_avg_by_age = car_df.groupby("Car_Age")["Price"].mean().reset_index()
_avg_by_age.columns = ["Car_Age", "Avg_Price"]

ax.plot(
    _avg_by_age["Car_Age"], _avg_by_age["Avg_Price"],
    color="#FFB482", linewidth=2.5, label="Average price"
)

ax.set_title("Car Age vs Price", color="#fbfbff", fontsize=14, pad=12)
ax.set_xlabel("Car Age (years)", color="#909094", fontsize=11)
ax.set_ylabel("Price", color="#909094", fontsize=11)
ax.tick_params(colors="#909094")
for spine in ax.spines.values():
    spine.set_edgecolor("#909094")
ax.legend(facecolor="#1D1D20", labelcolor="#fbfbff")
plt.tight_layout()

# --- Average price by age group ---
_bins = [0, 3, 6, 10, 15, 20, 100]
_labels = ["0–3 yrs", "4–6 yrs", "7–10 yrs", "11–15 yrs", "16–20 yrs", "20+ yrs"]
_avg_by_age["Age_Group"] = np.searchsorted(_bins, _avg_by_age["Car_Age"].values, side="right") - 1
_avg_by_age["Age_Group"] = [_labels[min(i, len(_labels)-1)] for i in _avg_by_age["Age_Group"]]

avg_price_by_age_group = (
    car_df.assign(Age_Group=np.where(
        car_df["Car_Age"] <= 3, "0–3 yrs",
        np.where(car_df["Car_Age"] <= 6, "4–6 yrs",
        np.where(car_df["Car_Age"] <= 10, "7–10 yrs",
        np.where(car_df["Car_Age"] <= 15, "11–15 yrs",
        np.where(car_df["Car_Age"] <= 20, "16–20 yrs", "20+ yrs"))))
    ))
    .groupby("Age_Group")["Price"]
    .mean()
    .reindex(_labels)
    .reset_index()
)
avg_price_by_age_group.columns = ["Age_Group", "Avg_Price"]
avg_price_by_age_group["Avg_Price"] = avg_price_by_age_group["Avg_Price"].round(2)

print("Average Price by Age Group:")
print(avg_price_by_age_group.to_string(index=False))
