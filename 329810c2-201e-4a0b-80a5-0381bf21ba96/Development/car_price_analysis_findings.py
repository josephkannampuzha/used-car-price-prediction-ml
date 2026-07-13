import numpy as np

# ==========================
# Key Findings
# ==========================

print("=" * 50)
print("KEY FINDINGS")
print("=" * 50)

# Recompute correlation matrix (upstream uses private _corr variable)
_numeric_df = car_df.select_dtypes(include=["int64", "float64"])
_corr = _numeric_df.corr()

# Strongest correlations with price
top_corr = (
    _corr["Price"]
    .drop("Price")
    .sort_values(key=abs, ascending=False)
    .head(5)
)

print("\nTop Features Correlated with Price:")
print(top_corr.round(3))

# Highest-value brand
_brand_avg = car_df.groupby("Brand")["Price"].mean()
top_brand = _brand_avg.idxmax()
top_brand_price = _brand_avg.max()

print(f"\nHighest Average Resale Value:")
print(f"• {top_brand}: ${top_brand_price:,.0f}")

# Transmission comparison
transmission_mean = (
    car_df.groupby("Transmission")["Price"]
    .mean()
    .sort_values(ascending=False)
)

print("\nAverage Price by Transmission:")
print(transmission_mean.apply(lambda x: f"₹{x:,.0f}"))

# Fuel type comparison — use cleaned column if available
_fuel_col = "Fuel_Type_Clean" if "Fuel_Type_Clean" in car_df.columns else "Fuel_Type"
fuel_mean = (
    car_df.groupby(_fuel_col)["Price"]
    .mean()
    .sort_values(ascending=False)
)

print("\nAverage Price by Fuel Type:")
print(fuel_mean.apply(lambda x: f"${x:,.0f}"))

print("\nBusiness Insights")
print("-" * 50)
print("• Vehicle age is one of the strongest factors influencing resale price.")
print("• Premium brands consistently command higher resale values.")
print("• Fuel type and transmission influence price, but less than age and brand.")
print("• These findings will guide feature selection for the predictive pricing model.")
