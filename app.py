"""
Heart Disease Risk Predictor - Streamlit App
"""
import sys
import joblib
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

BASE_DIR = Path(__file__).resolve().parent
sys.path.append(str(BASE_DIR / "src"))

MODEL_PATH = BASE_DIR / "models" / "disease_lgbm_model.pkl"
METRICS_PATH = BASE_DIR / "models" / "metrics.pkl"
DATA_PATH = BASE_DIR / "datas" / "heart.csv"

# -----------------------------------------------------------------------
# Categorical mappings
# -----------------------------------------------------------------------
CP_OPTIONS = {
    "Typical Angina": 0,
    "Atypical Angina": 1,
    "Non-Anginal Pain": 2,
    "Asymptomatic": 3,
}
RESTECG_OPTIONS = {
    "Normal": 0,
    "ST-T Wave Abnormality": 1,
    "Left Ventricular Hypertrophy": 2,
}
SLOPE_OPTIONS = {
    "Upsloping": 0,
    "Flat": 1,
    "Downsloping": 2,
}
THAL_OPTIONS = {
    "Normal": 1,
    "Fixed Defect": 2,
    "Reversible Defect": 3,
}

# -----------------------------------------------------------------------
# Loaders
# -----------------------------------------------------------------------
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

@st.cache_resource
def load_metrics():
    return joblib.load(METRICS_PATH)

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    return df.drop_duplicates()

# -----------------------------------------------------------------------
# Page config
# -----------------------------------------------------------------------
st.set_page_config(
    page_title="Heart Disease Risk Predictor",
    page_icon="🫀",
    layout="wide",
)

st.title("🫀 Heart Disease Risk Predictor")
st.markdown(
    "Enter patient information below to predict heart disease risk using a "
    "**LightGBM** model trained on the UCI Heart Disease dataset."
)

model = load_model()
metrics = load_metrics()
df = load_data()

tab1, tab2, tab3 = st.tabs(["🔮 Prediction", "📊 Data Explorer", "📈 Model Performance"])

# -----------------------------------------------------------------------
# TAB 1: PREDICTION
# -----------------------------------------------------------------------
with tab1:
    st.header("Patient Information")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Demographics")
        age = st.slider("Age", 20, 80, 50)
        sex = st.selectbox("Sex", options=["Female", "Male"])
        sex_val = 0 if sex == "Female" else 1

        st.subheader("Symptoms")
        cp = st.selectbox("Chest Pain Type", options=list(CP_OPTIONS.keys()))
        exang = st.selectbox("Exercise Induced Angina", options=["No", "Yes"])
        exang_val = 0 if exang == "No" else 1

    with col2:
        st.subheader("Vitals")
        trestbps = st.slider("Resting Blood Pressure (mmHg)", 80, 200, 120)
        chol = st.slider("Cholesterol (mg/dl)", 100, 600, 240)
        thalach = st.slider("Max Heart Rate Achieved", 60, 220, 150)
        oldpeak = st.slider("ST Depression (Oldpeak)", 0.0, 6.5, 1.0, 0.1)

    with col3:
        st.subheader("Clinical Tests")
        fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", options=["No", "Yes"])
        fbs_val = 0 if fbs == "No" else 1
        restecg = st.selectbox("Resting ECG Result", options=list(RESTECG_OPTIONS.keys()))
        slope = st.selectbox("Slope of Peak Exercise ST", options=list(SLOPE_OPTIONS.keys()))
        ca = st.selectbox("Number of Major Vessels (Fluoroscopy)", options=[0, 1, 2, 3])
        thal = st.selectbox("Thalassemia", options=list(THAL_OPTIONS.keys()))

    st.divider()

    if st.button("Predict Risk", type="primary", use_container_width=True):
        input_dict = {
            "age": age,
            "sex": sex_val,
            "cp": CP_OPTIONS[cp],
            "trestbps": trestbps,
            "chol": chol,
            "fbs": fbs_val,
            "restecg": RESTECG_OPTIONS[restecg],
            "thalach": thalach,
            "exang": exang_val,
            "oldpeak": oldpeak,
            "slope": SLOPE_OPTIONS[slope],
            "ca": ca,
            "thal": THAL_OPTIONS[thal],
        }

        input_df = pd.DataFrame([input_dict])
        pred = model.predict(input_df)[0]
        proba = model.predict_proba(input_df)[0][1]

        if pred == 1:
            st.error(f"### ⚠️ High Risk — Heart Disease Detected")
            st.metric("Risk Probability", f"{proba:.1%}")
        else:
            st.success(f"### ✅ Low Risk — No Heart Disease Detected")
            st.metric("Risk Probability", f"{proba:.1%}")

        st.caption(
            f"Model achieves {metrics['recall']:.1%} recall on the test set — "
            f"meaning it correctly identifies {metrics['recall']:.1%} of actual heart disease cases."
        )

        st.progress(float(proba), text=f"Disease probability: {proba:.1%}")

# -----------------------------------------------------------------------
# TAB 2: DATA EXPLORER
# -----------------------------------------------------------------------
with tab2:
    sns.set_style('darkgrid')
    st.header("Dataset Overview")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Records", len(df))
    c2.metric("Heart Disease Cases", int(df["target"].sum()))
    c3.metric("Healthy Cases", int((df["target"] == 0).sum()))
    c4.metric("Disease Rate", f"{df['target'].mean():.1%}")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Target Distribution")
        fig, ax = plt.subplots()
        df["target"].value_counts().rename({0: "No Disease", 1: "Disease"}).plot(
            kind="bar", ax=ax, color=["#4CAF50", "#F44336"], edgecolor="white"
        )
        ax.set_xticklabels(["No Disease", "Disease"], rotation=0)
        ax.set_ylabel("Count")
        st.pyplot(fig)
        plt.close()

    with col2:
        st.subheader("Age Distribution by Target")
        fig, ax = plt.subplots()
        sns.histplot(data=df, x="age", hue="target", multiple="stack", 
                     bins=20, ax=ax, palette={0: "#4CAF50", 1: "#F44336"})
        ax.set_xlabel("Age")
        st.pyplot(fig)
        plt.close()

    col3, col4 = st.columns(2)

    with col3:
        st.subheader("Cholesterol by Target")
        fig, ax = plt.subplots()
        sns.boxplot(data=df, x="target", y="chol", ax=ax,
                    palette={'0': "#4CAF50", '1': "#F44336"})
        ax.set_xticklabels(["No Disease", "Disease"])
        st.pyplot(fig)
        plt.close()

with col4:
    st.subheader("Feature Importance")
    importances = model.named_steps['model'].feature_importances_
    feature_names = model.named_steps['preprocessor'].get_feature_names_out()

    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False)

    importance_df['feature'] = importance_df['feature'].str.replace(
        r'^(num__|remainder__)', '', regex=True
    )

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.barplot(data=importance_df, x='importance', y='feature',
                palette='magma', ax=ax)
    ax.set_xlabel("Importance Score")
    ax.set_ylabel("")
    ax.set_title("Feature Importance", fontweight='bold')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()
# -----------------------------------------------------------------------
# TAB 3: MODEL PERFORMANCE
# -----------------------------------------------------------------------
with tab3:
    st.header("Model Performance")

    st.subheader("Test Set Metrics")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Accuracy", f"{metrics['accuracy']:.3f}")
    m2.metric("Precision", f"{metrics['precision']:.3f}")
    m3.metric("Recall", f"{metrics['recall']:.3f}")
    m4.metric("F1 Score", f"{metrics['f1']:.3f}")

    st.info(
        "**Recall** is the most critical metric in medical diagnosis — "
        "it measures how many actual heart disease cases the model correctly identifies. "
        f"This model achieves **{metrics['recall']:.1%} recall**, meaning it misses "
        f"{1 - metrics['recall']:.1%} of real cases (false negatives)."
    )

    st.subheader("About the Model")
    st.markdown("""
    | | |
    |---|---|
    | **Algorithm** | LightGBM Classifier |
    | **Hyperparameter Tuning** | RandomizedSearchCV (recall-optimized) |
    | **Preprocessing** | StandardScaler (numerical) + passthrough (categorical) |
    | **Train/Test Split** | 80% / 20%, random_state=42 |
    | **Dataset** | UCI Heart Disease (Cleveland), duplicates removed |
    """)

    st.subheader("Why LightGBM?")
    st.markdown("""
    In baseline comparison (XGBoost, Logistic Regression, Random Forest, LightGBM):
    - Tree-based models outperformed Logistic Regression on this dataset
    - LightGBM achieved the best **recall** score after hyperparameter tuning
    - Recall was prioritized over accuracy because **missing a heart disease case is more dangerous** than a false alarm
    """)

# -----------------------------------------------------------------------
# Sidebar
# -----------------------------------------------------------------------
st.sidebar.title("ℹ️ About")
st.sidebar.markdown("""
**Heart Disease Risk Predictor**

ML project trained on the UCI Heart Disease (Cleveland) dataset.

**Project Structure:**
- `notebooks/` — EDA & model selection
- `src/preprocessing.py` — data loading & feature pipeline
- `src/train.py` — model training & evaluation
- `src/predict.py` — inference helper
- `app/app.py` — this Streamlit app

""")