
# ==========================
# Feature Importance
# ==========================

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Get feature names after preprocessing
feature_names = tree_model.named_steps["preprocessor"].get_feature_names_out()

# Get importance scores
importance = tree_model.named_steps["model"].feature_importances_

# Create DataFrame
importance_df = (
    pd.DataFrame({
        "Feature": feature_names,
        "Importance": importance
    })
    .sort_values("Importance", ascending=False)
    .head(10)
    .reset_index(drop=True)
)

print(importance_df.to_string(index=False))

# Plot
feature_importance_fig, ax = plt.subplots(figsize=(10, 6))
feature_importance_fig.patch.set_facecolor("#1D1D20")
ax.set_facecolor("#1D1D20")

_colors = plt.cm.Blues_r(np.linspace(0.3, 0.8, len(importance_df)))

ax.barh(
    importance_df["Feature"],
    importance_df["Importance"],
    color=_colors,
    edgecolor="none"
)

ax.invert_yaxis()
ax.set_title("Top 10 Most Important Features", color="#fbfbff", fontsize=14, pad=12)
ax.set_xlabel("Importance", color="#fbfbff")
ax.tick_params(colors="#fbfbff")
for spine in ax.spines.values():
    spine.set_edgecolor("#909094")

plt.tight_layout()
plt.close("all")
