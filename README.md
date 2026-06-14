# 🫀 Heart Disease Risk Predictor

> 🌐 **Project URL:** [STREAMLIT WEB APP](https://heart-disease-prediction-brkakyldz.streamlit.app)

An machine learning project that predicts heart disease risk using the UCI Heart Disease (Cleveland) dataset. The project covers the full ML pipeline: exploratory data analysis, baseline model comparison, hyperparameter tuning, and deployment via a Streamlit web application.

---

## 📁 Project Structure

```
heart_disease/
├── datas/
│   └── heart.csv                  # Raw dataset
├── notebooks/
│   └── eda_and_modeling.ipynb     # EDA, baseline comparison, hyperparameter tuning
├── src/
│   ├── __init__.py
│   ├── preprocessing.py           # Data loading, cleaning, ColumnTransformer
│   ├── train.py                   # Model training, evaluation, saving
│   └── predict.py                 # Inference helper function
├── models/
│   ├── disease_lgbm_model.pkl     # Trained pipeline (preprocessing + model)
│   └── metrics.pkl                # Test set performance metrics
├── app.py                         # Streamlit web application                  
├── requirements.txt
└── README.md
```

---

## 🚀 Setup & Usage

### 1. Clone the repository

```bash
git clone https://github.com/brkakyldz/heart-disease-predictor.git
cd heart-disease-predictor
```

### 2. Create and activate virtual environment

```bash
python3 -m venv venv

# macOS / Linux
source venv/bin/activate

# Windows
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Train the model (optional — pre-trained model already in `models/`)

```bash
python src/train.py
```

### 5. Run the Streamlit app

```bash
streamlit run app.py
```

---

## 📊 Dataset

- **Source:** UCI Heart Disease Dataset (Cleveland)
- **Original size:** 1,025 rows
- **After removing duplicates:** 303 unique rows
- **Features:** 13 clinical attributes (age, sex, chest pain type, cholesterol, etc.)
- **Target:** Binary — `1` = Heart Disease, `0` = No Heart Disease

> ⚠️ The dataset contained 700+ duplicate rows, which were removed before training to prevent data leakage and inflated scores.

---

## 🔬 Project Workflow

### 1. Exploratory Data Analysis (`notebook/`)
- Checked for missing values, duplicates, and class balance
- Visualized distributions (age, cholesterol, max heart rate)
- Analyzed feature-target relationships via boxplots and histograms
- Computed correlation matrix

### 2. Baseline Model Comparison
Four models were evaluated on the same preprocessing pipeline (StandardScaler on numerical features):

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Logistic Regression | 0.787 | 0.722 | 0.897 | 0.800 |
| XGBoost | 0.803 | 0.758 | 0.862 | 0.806 |
| LightGBM | 0.803 | 0.758 | 0.862 | 0.806 |
| **Random Forest** | **0.836** | **0.788** | **0.897** | **0.839** |

> Note: Tree-based models (XGBoost, RandomForest, LightGBM) showed overfitting on train set (accuracy = 1.0), which is expected on a small dataset. Test scores were used for model selection.

### 3. Hyperparameter Tuning
`RandomizedSearchCV` was applied to LightGBM with `scoring='recall'`, since **missing a real heart disease case (false negative) is more costly than a false alarm**.

Best parameters found:

```python
{
    'num_leaves': 63,
    'max_depth': 5,
    'min_child_samples': 10,
    'n_estimators': 50,
    'learning_rate': 0.01,
    'reg_alpha': 1.0,
    'reg_lambda': 5.0,
    'subsample': 0.8,
    'colsample_bytree': 0.8
}
```

### 4. Final Model Evaluation (Test Set)

| Metric | Score |
|---|---|
| Accuracy | 0.754 |
| Precision | 0.675 |
| Recall | **0.931** |
| F1 Score | 0.783 |

> **Recall was prioritized** — the model correctly identifies 96.6% of actual heart disease cases. The trade-off is lower precision (some healthy patients may be flagged), which is acceptable in a medical screening context.

---

## 🛠️ Tech Stack

| | |
|---|---|
| **Language** | Python 3 |
| **ML Library** | scikit-learn, LightGBM |
| **Data** | pandas, numpy |
| **Visualization** | matplotlib, seaborn |
| **Deployment** | Streamlit |
| **Model Serialization** | joblib |

---
