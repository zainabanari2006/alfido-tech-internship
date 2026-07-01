# ============================================================
# TASK 1: IRIS CLASSIFICATION — Alfido Tech Internship
# Author: Zainab Ansari
# GitHub: github.com/zainabansari2006
# ============================================================

# ── 1. IMPORTS ──────────────────────────────────────────────
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             classification_report, ConfusionMatrixDisplay)
import joblib

print("✅ All libraries imported successfully!")


# ── 2. LOAD DATASET ─────────────────────────────────────────
# Download from: https://www.kaggle.com/datasets/bhanupratapbiswas/iris-classification-dataset
# Place iris.csv in the same folder as this script

df = pd.read_csv("iris.csv")

print("\n📊 Shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())

print("\nColumn Names:", df.columns.tolist())
print("\nData Types:\n", df.dtypes)
print("\nMissing Values:\n", df.isnull().sum())


# ── 3. EXPLORATORY DATA ANALYSIS (EDA) ──────────────────────
print("\n📈 Class Distribution:")
print(df['Species'].value_counts())

print("\n📐 Statistical Summary:")
print(df.describe())

# --- Plot 1: Class Distribution Bar Chart ---
plt.figure(figsize=(6, 4))
df['Species'].value_counts().plot(kind='bar', color=['#e63946','#457b9d','#2a9d8f'], edgecolor='black')
plt.title("Class Distribution of Iris Species", fontsize=14, fontweight='bold')
plt.xlabel("Species")
plt.ylabel("Count")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("plot_class_distribution.png", dpi=150)
plt.show()
print("✅ Saved: plot_class_distribution.png")

# --- Plot 2: Pairplot (Class Separability) ---
plt.figure()
sns.pairplot(df, hue='Species', palette='Set2', diag_kind='kde')
plt.suptitle("Pairplot — Iris Feature Relationships", y=1.02, fontsize=13, fontweight='bold')
plt.savefig("plot_pairplot.png", dpi=150, bbox_inches='tight')
plt.show()
print("✅ Saved: plot_pairplot.png")

# --- Plot 3: Correlation Heatmap ---
plt.figure(figsize=(7, 5))
numeric_df = df.select_dtypes(include=np.number)
sns.heatmap(numeric_df.corr(), annot=True, fmt=".2f", cmap='coolwarm', linewidths=0.5)
plt.title("Feature Correlation Heatmap", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig("plot_heatmap.png", dpi=150)
plt.show()
print("✅ Saved: plot_heatmap.png")

# --- Plot 4: Boxplots per Feature ---
features = [c for c in df.columns if c != 'Species']
fig, axes = plt.subplots(1, len(features), figsize=(16, 5))
for ax, feat in zip(axes, features):
    df.boxplot(column=feat, by='Species', ax=ax)
    ax.set_title(feat, fontsize=11)
    ax.set_xlabel("")
plt.suptitle("Boxplots: Feature Distribution by Species", fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig("plot_boxplots.png", dpi=150)
plt.show()
print("✅ Saved: plot_boxplots.png")


# ── 4. PREPROCESSING ────────────────────────────────────────
X = df.drop('Species', axis=1)
y = df['Species']

# Encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)
print("\n🔤 Label Encoding:")
for cls, code in zip(le.classes_, le.transform(le.classes_)):
    print(f"  {cls} → {code}")

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)
print(f"\n📂 Train size: {X_train.shape[0]}  |  Test size: {X_test.shape[0]}")

# Feature Scaling
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)


# ── 5. TRAIN & EVALUATE MODELS ──────────────────────────────
models = {
    "k-NN (k=5)":            KNeighborsClassifier(n_neighbors=5),
    "Logistic Regression":   LogisticRegression(max_iter=200, random_state=42),
    "Decision Tree":         DecisionTreeClassifier(max_depth=4, random_state=42)
}

results = {}

for name, model in models.items():
    model.fit(X_train_sc, y_train)
    y_pred = model.predict(X_test_sc)

    acc    = accuracy_score(y_test, y_pred)
    cv_acc = cross_val_score(model, scaler.transform(X), y_encoded, cv=5).mean()
    cm     = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=le.classes_)

    results[name] = {
        "model":    model,
        "accuracy": acc,
        "cv_acc":   cv_acc,
        "cm":       cm,
        "report":   report,
        "y_pred":   y_pred
    }

    print(f"\n{'='*55}")
    print(f"🔷 Model: {name}")
    print(f"   Test Accuracy  : {acc:.4f} ({acc*100:.2f}%)")
    print(f"   CV Accuracy    : {cv_acc:.4f}")
    print(f"\n   Classification Report:\n{report}")


# ── 6. CONFUSION MATRICES ────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for ax, (name, res) in zip(axes, results.items()):
    disp = ConfusionMatrixDisplay(confusion_matrix=res['cm'],
                                  display_labels=le.classes_)
    disp.plot(ax=ax, colorbar=False, cmap='Blues')
    ax.set_title(f"{name}\nAcc: {res['accuracy']*100:.1f}%", fontweight='bold')
plt.suptitle("Confusion Matrices — All Models", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig("plot_confusion_matrices.png", dpi=150)
plt.show()
print("✅ Saved: plot_confusion_matrices.png")


# ── 7. MODEL COMPARISON CHART ───────────────────────────────
model_names = list(results.keys())
test_accs   = [results[m]['accuracy'] for m in model_names]
cv_accs     = [results[m]['cv_acc']   for m in model_names]

x = np.arange(len(model_names))
width = 0.35

fig, ax = plt.subplots(figsize=(9, 5))
bars1 = ax.bar(x - width/2, test_accs, width, label='Test Accuracy',  color='#457b9d', edgecolor='black')
bars2 = ax.bar(x + width/2, cv_accs,   width, label='CV Accuracy (5-fold)', color='#2a9d8f', edgecolor='black')

ax.set_title("Model Comparison — Accuracy", fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(model_names, fontsize=11)
ax.set_ylim(0.85, 1.02)
ax.set_ylabel("Accuracy")
ax.legend()

for bar in bars1:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
            f"{bar.get_height():.3f}", ha='center', va='bottom', fontsize=10)
for bar in bars2:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
            f"{bar.get_height():.3f}", ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.savefig("plot_model_comparison.png", dpi=150)
plt.show()
print("✅ Saved: plot_model_comparison.png")


# ── 8. DECISION TREE VISUALIZATION ──────────────────────────
dt_model = results["Decision Tree"]["model"]
plt.figure(figsize=(18, 8))
plot_tree(dt_model, feature_names=X.columns.tolist(),
          class_names=le.classes_, filled=True, rounded=True, fontsize=10)
plt.title("Decision Tree Structure", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig("plot_decision_tree.png", dpi=150)
plt.show()
print("✅ Saved: plot_decision_tree.png")


# ── 9. SELECT BEST MODEL & SAVE ─────────────────────────────
best_name  = max(results, key=lambda m: results[m]['cv_acc'])
best_model = results[best_name]['model']
best_acc   = results[best_name]['cv_acc']

print(f"\n🏆 Best Model: {best_name}  (CV Accuracy: {best_acc*100:.2f}%)")

# Save model and scaler
joblib.dump(best_model, "best_iris_model.joblib")
joblib.dump(scaler,     "scaler.joblib")
joblib.dump(le,         "label_encoder.joblib")
print("✅ Saved: best_iris_model.joblib, scaler.joblib, label_encoder.joblib")


# ── 10. INFERENCE EXAMPLE ───────────────────────────────────
print("\n" + "="*55)
print("🔍 INFERENCE EXAMPLE")
print("="*55)

# Load saved model
loaded_model = joblib.load("best_iris_model.joblib")
loaded_scaler = joblib.load("scaler.joblib")
loaded_le     = joblib.load("label_encoder.joblib")

# Example: new flower measurements
# [SepalLengthCm, SepalWidthCm, PetalLengthCm, PetalWidthCm]
new_flower = np.array([[5.1, 3.5, 1.4, 0.2]])   # Should be Iris-setosa

new_flower_sc  = loaded_scaler.transform(new_flower)
prediction_enc = loaded_model.predict(new_flower_sc)
prediction     = loaded_le.inverse_transform(prediction_enc)

print(f"\n  Input features : {new_flower[0].tolist()}")
print(f"  Predicted class: {prediction[0]}")

# Try all three species examples
examples = {
    "Setosa":     [5.1, 3.5, 1.4, 0.2],
    "Versicolor": [6.0, 2.9, 4.5, 1.5],
    "Virginica":  [6.7, 3.1, 5.6, 2.4],
}
print("\n  Batch inference:")
for true_label, feats in examples.items():
    inp = np.array([feats])
    sc_inp = loaded_scaler.transform(inp)
    pred   = loaded_le.inverse_transform(loaded_model.predict(sc_inp))[0]
    status = "✅" if true_label.lower() in pred.lower() else "❌"
    print(f"  {status}  True: {true_label:<12}  Predicted: {pred}")

print("\n🎉 All done! Submit your notebook + saved files to Alfido Tech.")
