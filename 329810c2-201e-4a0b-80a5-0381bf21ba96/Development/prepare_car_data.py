# ==========================
# Prepare Data
# ==========================

target = "Price"

features = [
    "Brand",
    "Model",
    "Car_Age",
    "Mileage_kmpl",
    "Engine_CC",
    "Horsepower",
    "Fuel_Type",
    "Transmission",
    "Owner_Type",
    "Kms_Driven",
    "Insurance_Valid",
    "Service_History",
    "Accidents",
    "Tax_Paid",
    "Number_of_Doors",
    "Seats",
]

X = car_df[features]
y = car_df[target]

categorical_features = X.select_dtypes(include="object").columns.tolist()
numerical_features = X.select_dtypes(exclude="object").columns.tolist()

print(f"Training Features: {len(features)}")
print(f"Categorical Features: {len(categorical_features)}")
print(f"Numerical Features: {len(numerical_features)}")