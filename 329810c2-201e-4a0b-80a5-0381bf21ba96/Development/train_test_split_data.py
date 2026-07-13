# ==========================
# Train/Test Split
# ==========================
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print(f"Training Samples: {len(X_train):,}")
print(f"Testing Samples: {len(X_test):,}")