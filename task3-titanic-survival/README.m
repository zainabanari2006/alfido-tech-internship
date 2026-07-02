# TASK 3: TITANIC SURVIVAL PREDICTION — Alfido Tech Internship
# Author: Zainab Ansari

# ── 1. IMPORTS ──────────────────────────────────────────────
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             classification_report, ConfusionMatrixDisplay)
import joblib

print("✅ All libraries imported successfully!")


# ── 2. LOAD DATASET ─────────────────────────────────────────
# Download from: https://www.kaggle.com/datasets/bhanupratapbiswas/titanic-survival-datasets
# Place the CSV file in the same folder as this script

df = pd.read_csv("titanic.csv")

print("\n📊 Shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())
print("\nColumn Names:", df.columns.tolist())
print("\nMissing Values:\n", df.isnull().sum())


# ── 3. FEATURE ENGINEERING ──────────────────────────────────
print("\n🔧 Engineering new features...")

# Extract Title from Name
if 'Name' in df.columns:
    df['Title'] = df['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
    df['Title'] = df['Title'].replace(
        ['Lady','Countess','Capt','Col','Don','Dr','Major','Rev','Sir','Jonkheer','Dona'], 'Rare')
    df['Title'] = df['Title'].replace('Mlle', 'Miss')
    df['Title'] = df['Title'].replace('Ms', 'Miss')
    df['Title'] = df['Title'].replace('Mme', 'Mrs')
    print("  ✅ Extracted Title from Name")
    print("  Title counts:\n", df['Title'].value_counts())

# Family Size
if 'SibSp' in df.columns and 'Parch' in df.columns:
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
    print("  ✅ Created FamilySize and IsAlone features")

# Cabin Presence
if 'Cabin' in df.columns:
    df['HasCabin'] = df['Cabin'].notna().astype(int)
    print("  ✅ Created HasCabin feature")


# ── 4. EDA PLOTS ────────────────────────────────────────────

# Find target column
target_col = 'Survived' if 'Survived' in df.columns else df.columns[1]
print(f"\n🎯 Target column: {target_col}")

# --- Plot 1: Survival Count ---
plt.figure(figsize=(6, 4))
df[target_col].value_counts().plot(kind='bar',
    color=['#e63946','#2a9d8f'], edgecolor='black')
plt.title("Survival Count (0=Died, 1=Survived)", fontsize=14, fontweight='bold')
plt.xlabel("Survived")
plt.ylabel("Count")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("plot_survival_count.png", dpi=150)
plt.show()
print("✅ Saved: plot_survival_count.png")

# --- Plot 2: Survival by Gender ---
if 'Sex' in df.columns:
    plt.figure(figsize=(7, 5))
    sns.countplot(data=df, x='Sex', hue=target_col,
                  palette=['#e63946','#2a9d8f'])
    plt.title("Survival by Gender", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig("plot_survival_by_gender.png", dpi=150)
    plt.show()
    print("✅ Saved: plot_survival_by_gender.png")

# --- Plot 3: Survival by Passenger Class ---
if 'Pclass' in df.columns:
    plt.figure(figsize=(7, 5))
    sns.countplot(data=df, x='Pclass', hue=target_col,
                  palette=['#e63946','#2a9d8f'])
    plt.title("Survival by Passenger Class", fontsize=14, fontweight='bold')
    plt.xlabel("Passenger Class (1=First, 2=Second, 3=Third)")
    plt.tight_layout()
    plt.savefig("plot_survival_by_class.png", dpi=150)
    plt.show()
    print("✅ Saved: plot_survival_by_class.png")

# --- Plot 4: Age Distribution ---
if 'Age' in df.columns:
    plt.figure(figsize=(8, 5))
    df[df[target_col]==1]['Age'].hist(alpha=0.6, color='#2a9d8f',
                                       bins=30, label='Survived')
    df[df[target_col]==0]['Age'].hist(alpha=0.6, color='#e63946',
                                       bins=30, label='Died')
    plt.title("Age Distribution by Survival", fontsize=14, fontweight='bold')
    plt.xlabel("Age")
    plt.legend()
    plt.tight_layout()
    plt.savefig("plot_age_distribution.png", dpi=150)
    plt.show()
    print("✅ Saved: plot_age_distribution.png")

# --- Plot 5: Family Size vs Survival ---
if 'FamilySize' in df.columns:
    plt.figure(figsize=(8, 5))
    sns.countplot(data=df, x='FamilySize', hue=target_col,
                  palette=['#e63946','#2a9d8f'])
    plt.title("Survival by Family Size", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig("plot_family_size.png", dpi=150)
    plt.show()
    print("✅ Saved: plot_family_size.png")

# --- Plot 6: Correlation Heatmap ---
plt.figure(figsize=(12, 8))
numeric_df = df.select_dtypes(include=np.number)
sns.heatmap(numeric_df.corr(), annot=True, fmt=".2f",
            cmap='coolwarm', linewidths=0.5)
plt.title("Feature Correlation Heatmap", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig("plot_heatmap.png", dpi=150)
plt.show()
print("✅ Saved: plot_heatmap.png")


# ── 5. PREPROCESSING ────────────────────────────────────────
print("\n⚙️ Preprocessing data...")

# Handle missing Age with median
if 'Age' in df.columns:
    df['Age'].fillna(df['Age'].median(), inplace=True)

# Handle missing Embarked with mode
if 'Embarked' in df.columns:
    df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

# Handle missing Fare with median
if 'Fare' in df.columns:
    df['Fare'].fillna(df['Fare'].median(), inplace=True)

print("✅ Missing values handled!")

# Encode categorical columns
le = LabelEncoder()
cat_cols = ['Sex', 'Embarked', 'Title']
for col in cat_cols:
    if col in df.columns:
        df[col] = le.fit_transform(df[col].astype(str))
        print(f"  Encoded: {col}")

# Drop irrelevant columns
drop_cols = ['Name', 'Ticket', 'Cabin', 'PassengerId']
drop_cols = [c for c in drop_cols if c in df.columns]
df.drop(columns=drop_cols, inplace=True)

# Features and target
X = df.drop(target_col, axis=1)
y = df[target_col]

print(f"\n📂 Features: {X.shape[1]}  |  Samples: {X.shape[0]}")
print(f"Features used: {X.columns.tolist()}")

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Train size: {X_train.shape[0]}  |  Test size: {X_test.shape[0]}")

# Feature Scaling
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)


# ── 6. TRAIN & EVALUATE MODELS ──────────────────────────────
models = {
    "Logistic Regression":   LogisticRegression(max_iter=200, random_state=42),
    "Random Forest":         RandomForestClassifier(n_estimators=100, random_state=42),
    "Gradient Boosting":     GradientBoostingClassifier(n_estimators=100, random_state=42)
}

results = {}

for name, model in models.items():
    model.fit(X_train_sc, y_train)
    y_pred = model.predict(X_test_sc)

    acc    = accuracy_score(y_test, y_pred)
    cv_acc = cross_val_score(model, scaler.transform(X), y, cv=5).mean()
    cm     = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred,
                                   target_names=['Died', 'Survived'])

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
    print(f"   Test Accuracy : {acc:.4f} ({acc*100:.2f}%)")
    print(f"   CV Accuracy   : {cv_acc:.4f}")
    print(f"\n   Classification Report:\n{report}")


# ── 7. CONFUSION MATRICES ────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for ax, (name, res) in zip(axes, results.items()):
    disp = ConfusionMatrixDisplay(confusion_matrix=res['cm'],
                                  display_labels=['Died', 'Survived'])
    disp.plot(ax=ax, colorbar=False, cmap='Blues')
    ax.set_title(f"{name}\nAcc: {res['accuracy']*100:.1f}%",
                 fontweight='bold')
plt.suptitle("Confusion Matrices — All Models", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig("plot_confusion_matrices.png", dpi=150)
plt.show()
print("✅ Saved: plot_confusion_matrices.png")


# ── 8. MODEL COMPARISON ──────────────────────────────────────
model_names = list(results.keys())
test_accs   = [results[m]['accuracy'] for m in model_names]
cv_accs     = [results[m]['cv_acc']   for m in model_names]

x = np.arange(len(model_names))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 6))
bars1 = ax.bar(x - width/2, test_accs, width,
               label='Test Accuracy', color='#457b9d', edgecolor='black')
bars2 = ax.bar(x + width/2, cv_accs, width,
               label='CV Accuracy (5-fold)', color='#2a9d8f', edgecolor='black')

ax.set_title("Model Comparison — Accuracy", fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(model_names, fontsize=11)
ax.set_ylim(0.7, 1.0)
ax.set_ylabel("Accuracy")
ax.legend()

for bar in bars1:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
            f"{bar.get_height():.3f}", ha='center', fontsize=10)
for bar in bars2:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
            f"{bar.get_height():.3f}", ha='center', fontsize=10)

plt.tight_layout()
plt.savefig("plot_model_comparison.png", dpi=150)
plt.show()
print("✅ Saved: plot_model_comparison.png")


# ── 9. FEATURE IMPORTANCE ────────────────────────────────────
rf_model = results["Random Forest"]["model"]
importances = pd.Series(rf_model.feature_importances_, index=X.columns)
importances = importances.sort_values(ascending=False)

plt.figure(figsize=(10, 6))
importances.plot(kind='bar', color='#457b9d', edgecolor='black')
plt.title("Feature Importances (Random Forest)", fontsize=14, fontweight='bold')
plt.xlabel("Features")
plt.ylabel("Importance")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig("plot_feature_importance.png", dpi=150)
plt.show()
print("✅ Saved: plot_feature_importance.png")


# ── 10. SAVE BEST MODEL ──────────────────────────────────────
best_name  = max(results, key=lambda m: results[m]['cv_acc'])
best_model = results[best_name]['model']
best_acc   = results[best_name]['cv_acc']

print(f"\n🏆 Best Model: {best_name}  (CV Accuracy: {best_acc*100:.2f}%)")

joblib.dump(best_model, "best_titanic_model.joblib")
joblib.dump(scaler,     "scaler_titanic.joblib")
print("✅ Saved: best_titanic_model.joblib, scaler_titanic.joblib")


# ── 11. INFERENCE EXAMPLE ───────────────────────────────────
print("\n" + "="*55)
print("🔍 INFERENCE EXAMPLE")
print("="*55)

loaded_model  = joblib.load("best_titanic_model.joblib")
loaded_scaler = joblib.load("scaler_titanic.joblib")

# Use first row of test set
sample    = X_test.iloc[[0]]
sample_sc = loaded_scaler.transform(sample)
pred      = loaded_model.predict(sample_sc)[0]
actual    = y_test.iloc[0]

print(f"\n  Predicted : {'Survived ✅' if pred == 1 else 'Died ❌'}")
print(f"  Actual    : {'Survived ✅' if actual == 1 else 'Died ❌'}")
print(f"  Correct   : {'Yes ✅' if pred == actual else 'No ❌'}")

print("\n🎉 All done! Submit your notebook + saved files to Alfido Tech.")
