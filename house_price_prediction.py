import os
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Keep generated files in the project's outputs folder.
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(PROJECT_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def main():
    # Load the California Housing dataset from scikit-learn.
    housing = fetch_california_housing(as_frame=True)
    X = housing.data
    y = housing.target

    print("Dataset shape:", X.shape)
    print("\nFirst five rows:")
    print(X.head())
    print("\nMissing values by feature:")
    print(X.isnull().sum())

    # Split before fitting the scaler to avoid data leakage.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LinearRegression()
    model.fit(X_train_scaled, y_train)
    predictions = model.predict(X_test_scaled)

    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    print(f"\nMean Squared Error (MSE): {mse:.4f}")
    print(f"R² Score: {r2:.4f}")

    with open(os.path.join(OUTPUT_DIR, "model_metrics.txt"), "w", encoding="utf-8") as f:
        f.write(f"Mean Squared Error (MSE): {mse:.4f}\n")
        f.write(f"R² Score: {r2:.4f}\n")

    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, predictions, alpha=0.45)
    plt.xlabel("Actual House Value (in $100,000s)")
    plt.ylabel("Predicted House Value (in $100,000s)")
    plt.title("Actual vs Predicted House Values")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "actual_vs_predicted.png"), dpi=150)
    plt.close()

    print("\nSaved graph and metrics in:", OUTPUT_DIR)

if __name__ == "__main__":
    main()
