import pandas as pd

# ==========================
# Model Comparison
# ==========================

results = pd.DataFrame(
    {
        "Model": [
            "Linear Regression",
            "Decision Tree",
        ],
        "MAE": [
            linear_mae,
            tree_mae,
        ],
        "RMSE": [
            linear_rmse,
            tree_rmse,
        ],
        "R²": [
            linear_r2,
            tree_r2,
        ],
    }
)

print(results.to_string(index=False))