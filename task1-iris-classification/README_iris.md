# 🌸 Task 1: Iris Classification
**Alfido Tech Internship | Zainab Ansari**  
GitHub: [github.com/zainabansari2006](https://github.com/zainabansari2006)

---

## 📌 Goal
Build a classification model to predict Iris species (Setosa, Versicolor, Virginica)
using sepal and petal measurements.

## 📂 Dataset
[Kaggle – Iris Classification Dataset](https://www.kaggle.com/datasets/bhanupratapbiswas/iris-classification-dataset)

---

## 🧠 Models Trained
| Model               | Test Accuracy | CV Accuracy (5-fold) |
|---------------------|:-------------:|:--------------------:|
| k-NN (k=5)          | ~97%          | ~97%                 |
| Logistic Regression | ~97%          | ~97%                 |
| Decision Tree       | ~97%          | ~96%                 |

---

## 📁 Files
| File | Description |
|------|-------------|
| `iris_classification.py` | Full pipeline: EDA → Train → Evaluate → Save |
| `best_iris_model.joblib` | Saved best model |
| `scaler.joblib` | Fitted StandardScaler |
| `label_encoder.joblib` | Fitted LabelEncoder |
| `plot_*.png` | All EDA and evaluation plots |

---

## 🚀 How to Run

### 1. Install dependencies
```bash
pip install numpy pandas matplotlib seaborn scikit-learn joblib
```

### 2. Download dataset
Download `iris.csv` from Kaggle and place it in this folder.

### 3. Run the script
```bash
python iris_classification.py
```

---

## 🔍 Inference Example
```python
import numpy as np
import joblib

model  = joblib.load("best_iris_model.joblib")
scaler = joblib.load("scaler.joblib")
le     = joblib.load("label_encoder.joblib")

# [SepalLength, SepalWidth, PetalLength, PetalWidth]
new_flower = np.array([[5.1, 3.5, 1.4, 0.2]])
prediction = le.inverse_transform(model.predict(scaler.transform(new_flower)))
print("Predicted species:", prediction[0])
# Output: Iris-setosa
```

---

## 📊 Plots Generated
- `plot_class_distribution.png` — Bar chart of species counts
- `plot_pairplot.png` — Pairplot showing class separability
- `plot_heatmap.png` — Feature correlation heatmap
- `plot_boxplots.png` — Per-feature boxplots by species
- `plot_confusion_matrices.png` — Confusion matrix for all 3 models
- `plot_model_comparison.png` — Accuracy comparison bar chart
- `plot_decision_tree.png` — Visual of trained Decision Tree

---

*Submitted as part of Alfido Tech ML Internship — June 2026*
