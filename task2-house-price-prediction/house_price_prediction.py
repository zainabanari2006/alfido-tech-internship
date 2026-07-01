# TASK 2: HOUSE PRICE PREDICTION — Alfido Tech Internship
# Author: Zainab Ansari
# GitHub: github.com/zainabanari2006

# ── 1. IMPORTS ───
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib

print("✅ All libraries imported successfully!")


# ── 2. LOAD DATASET ───
# Download from: https://www.kaggle.com/datasets/bhanupratapbiswas/house-price-prediction
# Place the CSV file in the same folder as this script

df = pd.read_csv("data.csv")

print("\n📊 Shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())
print("\nColumn Names:", df.columns.tolist())
print("\nData Types:\n", df.dtypes)
print("\nMissing Values:\n", df.isnull().sum())


# ── 3. EXPLORATORY DATA ANALYSIS (EDA) ──────────────────────
print("\n📐 Statistical Summary:")
print(df.describe())

# Identify target column (price)
# Common column names for price
price_col = None
for col in df.columns:
    if 'price' in col.lower():
        price_col = col
        break

if price_col is None:
    price_col = df.columns[-1]  # assume last column is target

print(f"\n🎯 Target column: {price_col}")

# --- Plot 1: Price Distribution ---
plt.figure(figsize=(8, 5))
sns.histplot(df[price_col], kde=True, color='#457b9d', bins=40)
plt.title("House Price Distribution", fontsize=14, fontweight='bold')
plt.xlabel("Price")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("plot_price_distribution.png", dpi=150)
plt.show()
print("✅ Saved: plot_price_distribution.png")

# --- Plot 2: Correlation Heatmap ---
plt.figure(figsize=(12, 8))
numeric_df = df.select_dtypes(include=np.number)
sns.heatmap(numeric_df.corr(), annot=True, fmt=".2f",
            cmap='coolwarm', linewidths=0.5)
plt.title("Feature Correlation Heatmap", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig("plot_heatmap.png", dpi=150)
plt.show()
print("✅ Saved: plot_heatmap.png")

# --- Plot 3: Top correlations with price ---
corr_with_price = numeric_df.corr()[price_col].drop(price_col).sort_values(ascending=False)
plt.figure(figsize=(10, 6))
corr_with_price.plot(kind='bar', color=['#2a9d8f' if x > 0 else '#e63946' for x in corr_with_price])
plt.title(f"Feature Correlation with {price_col}", fontsize=14, fontweight='bold')
plt.xlabel("Features")
plt.ylabel("Correlation")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig("plot_correlation_with_price.png", dpi=150)
plt.show()
print("✅ Saved: plot_correlation_with_price.png")


# ── 4. PREPROCESSING ────────────────────────────────────────

# Handle missing values
for col in df.columns:
    if df[col].dtype == 'object':
        df[col].fillna(df[col].mode()[0], inplace=True)
    else:
        df[col].fillna(df[col].median(), inplace=True)

print("\n✅ Missing values handled!")
print("Remaining missing values:", df.isnull().sum().sum())

# Encode categorical columns
le = LabelEncoder()
cat_cols = df.select_dtypes(include='object').columns.tolist()
for col in cat_cols:
    df[col] = le.fit_transform(df[col].astype(str))
    print(f"  Encoded: {col}")

# Log transform the target (reduces skewness)
df[f'log_{price_col}'] = np.log1p(df[price_col])

plt.figure(figsize=(8, 5))
sns.histplot(df[f'log_{price_col}'], kde=True, color='#e63946', bins=40)
plt.title("Log-Transformed Price Distribution", fontsize=14, fontweight='bold')
plt.xlabel("Log(Price)")
plt.tight_layout()
plt.savefig("plot_log_price_distribution.png", dpi=150)
plt.show()
print("✅ Saved: plot_log_price_distribution.png")

# Features and target
X = df.drop([price_col, f'log_{price_col}'], axis=1)
y = df[f'log_{price_col}']  # use log-transformed target

print(f"\n📂 Features: {X.shape[1]}  |  Samples: {X.shape[0]}")

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"Train size: {X_train.shape[0]}  |  Test size: {X_test.shape[0]}")

# Feature Scaling
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)


# ── 5. TRAIN & EVALUATE MODELS ──────────────────────────────
models = {
    "Linear Regression":    LinearRegression(),
    "Random Forest":        RandomForestRegressor(n_estimators=100, random_state=42),
    "Gradient Boosting":    GradientBoostingRegressor(n_estimators=100, random_state=42)
}

results = {}

for name, model in models.items():
    model.fit(X_train_sc, y_train)
    y_pred = model.predict(X_test_sc)

    # Convert back from log scale
    y_pred_actual = np.expm1(y_pred)
    y_test_actual = np.expm1(y_test)

    rmse = np.sqrt(mean_squared_error(y_test_actual, y_pred_actual))
    mae  = mean_absolute_error(y_test_actual, y_pred_actual)
    r2   = r2_score(y_test_actual, y_pred_actual)

    results[name] = {
        "model":         model,
        "rmse":          rmse,
        "mae":           mae,
        "r2":            r2,
        "y_pred":        y_pred_actual,
        "y_test":        y_test_actual
    }

    print(f"\n{'='*55}")
    print(f"🔷 Model: {name}")
    print(f"   RMSE : {rmse:,.2f}")
    print(f"   MAE  : {mae:,.2f}")
    print(f"   R²   : {r2:.4f} ({r2*100:.2f}%)")


# ── 6. RESIDUAL ANALYSIS PLOTS ───────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for ax, (name, res) in zip(axes, results.items()):
    residuals = res['y_test'] - res['y_pred']
    ax.scatter(res['y_pred'], residuals, alpha=0.5, color='#457b9d', s=20)
    ax.axhline(y=0, color='red', linestyle='--')
    ax.set_title(f"{name}\nResidual Plot", fontweight='bold')
    ax.set_xlabel("Predicted Price")
    ax.set_ylabel("Residual")
plt.suptitle("Residual Analysis — All Models", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig("plot_residuals.png", dpi=150)
plt.show()
print("✅ Saved: plot_residuals.png")


# --- Actual vs Predicted ---
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for ax, (name, res) in zip(axes, results.items()):
    ax.scatter(res['y_test'], res['y_pred'], alpha=0.5, color='#2a9d8f', s=20)
    min_val = min(res['y_test'].min(), res['y_pred'].min())
    max_val = max(res['y_test'].max(), res['y_pred'].max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--')
    ax.set_title(f"{name}\nR²: {res['r2']:.3f}", fontweight='bold')
    ax.set_xlabel("Actual Price")
    ax.set_ylabel("Predicted Price")
plt.suptitle("Actual vs Predicted Price", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig("plot_actual_vs_predicted.png", dpi=150)
plt.show()
print("✅ Saved: plot_actual_vs_predicted.png")


# ── 7. MODEL COMPARISON CHART ───────────────────────────────
model_names = list(results.keys())
r2_scores   = [results[m]['r2']   for m in model_names]
rmse_scores = [results[m]['rmse'] for m in model_names]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# R² comparison
bars = ax1.bar(model_names, r2_scores, color=['#457b9d','#2a9d8f','#e63946'], edgecolor='black')
ax1.set_title("R² Score Comparison", fontsize=13, fontweight='bold')
ax1.set_ylabel("R² Score")
ax1.set_ylim(0, 1.1)
for bar in bars:
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
             f"{bar.get_height():.3f}", ha='center', fontsize=11)

# RMSE comparison
bars2 = ax2.bar(model_names, rmse_scores, color=['#457b9d','#2a9d8f','#e63946'], edgecolor='black')
ax2.set_title("RMSE Comparison (Lower = Better)", fontsize=13, fontweight='bold')
ax2.set_ylabel("RMSE")
for bar in bars2:
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
             f"{bar.get_height():,.0f}", ha='center', fontsize=11)

plt.suptitle("Model Comparison", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig("plot_model_comparison.png", dpi=150)
plt.show()
print("✅ Saved: plot_model_comparison.png")


# ── 8. FEATURE IMPORTANCE ────────────────────────────────────
rf_model = results["Random Forest"]["model"]
importances = pd.Series(rf_model.feature_importances_, index=X.columns)
importances = importances.sort_values(ascending=False).head(15)

plt.figure(figsize=(10, 6))
importances.plot(kind='bar', color='#457b9d', edgecolor='black')
plt.title("Top 15 Feature Importances (Random Forest)", fontsize=14, fontweight='bold')
plt.xlabel("Features")
plt.ylabel("Importance")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig("plot_feature_importance.png", dpi=150)
plt.show()
print("✅ Saved: plot_feature_importance.png")


# ── 9. SELECT BEST MODEL & SAVE ─────────────────────────────
best_name  = max(results, key=lambda m: results[m]['r2'])
best_model = results[best_name]['model']
best_r2    = results[best_name]['r2']

print(f"\n🏆 Best Model: {best_name}  (R²: {best_r2*100:.2f}%)")

joblib.dump(best_model, "best_house_price_model.joblib")
joblib.dump(scaler,     "scaler_house.joblib")
print("✅ Saved: best_house_price_model.joblib, scaler_house.joblib")


# ── 10. INFERENCE EXAMPLE ───────────────────────────────────
print("\n" + "="*55)
print("🔍 INFERENCE EXAMPLE")
print("="*55)

loaded_model  = joblib.load("best_house_price_model.joblib")
loaded_scaler = joblib.load("scaler_house.joblib")

# Use first row of test set as example
sample = X_test.iloc[[0]]
sample_sc = loaded_scaler.transform(sample)
log_pred  = loaded_model.predict(sample_sc)
pred_price = np.expm1(log_pred)[0]
actual_price = np.expm1(y_test.iloc[0])

print(f"\n  Predicted Price : ${pred_price:,.2f}")
print(f"  Actual Price    : ${actual_price:,.2f}")
print(f"  Difference      : ${abs(pred_price - actual_price):,.2f}")

print("\n🎉 All done! Submit your notebook + saved files to Alfido Tech.")
