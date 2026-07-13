# ==========================
# Business Conclusions
# ==========================

best_model = results.loc[results["R²"].idxmax(), "Model"]

print("=" * 60)
print("BUSINESS CONCLUSIONS")
print("=" * 60)

print(f"\nBest Performing Model: {best_model}")
print(f"R² Score: {results['R²'].max():.3f}")

print("""
Key Findings:
• The Decision Tree model significantly outperformed Linear Regression,
  improving prediction accuracy and explaining approximately 67% of the
  variation in used car prices.

• Vehicle mileage (Kms_Driven) was the most important factor affecting
  resale value, followed by vehicle brand and age.

• Luxury brands such as Mercedes, Audi, and BMW consistently maintained
  higher resale values than other brands.

• Older vehicles and vehicles with higher mileage generally experienced
  lower resale values.

Business Impact:
• Dealerships can use predictive models to estimate fair market values
  for used vehicles.

• Buyers can identify potentially overpriced or undervalued vehicles
  by comparing listed prices with model predictions.

• Sellers can use data-driven pricing strategies to remain competitive
  in the used car market.

Project Outcome:
• Successfully cleaned and analyzed over 1 million vehicle records.
• Built and evaluated multiple machine learning models.
• Developed a predictive pricing solution capable of supporting
  data-driven decision making in the automotive resale market.
""")