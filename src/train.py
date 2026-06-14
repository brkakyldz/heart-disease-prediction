from pathlib import Path
import warnings
from preprocessing import load_data, build_preprocessor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score
from lightgbm import LGBMClassifier
import joblib

warnings.filterwarnings('ignore')

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / 'datas' / 'heart.csv'
MODEL_PATH = BASE_DIR / 'models' / 'disease_lgbm_model.pkl'

df = load_data(DATA_PATH)
X = df.drop(columns='target')
y = df['target']

pipeline = Pipeline([
    ('preprocessor', build_preprocessor()),
    ('model', LGBMClassifier(
            learning_rate=0.01,
            n_estimators=50,
            max_depth=3,
            num_leaves=63,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_samples=30,
            reg_alpha=1.0,
            reg_lambda=1.0,
            random_state=42
))
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)

metrics = {
    'accuracy': accuracy_score(y_test, y_pred),
    'precision': precision_score(y_test, y_pred),
    'recall': recall_score(y_test, y_pred),
    'f1': f1_score(y_test, y_pred),
}
print(metrics)

joblib.dump(metrics, BASE_DIR / 'models' / 'metrics.pkl')

pipeline.fit(X, y)
joblib.dump(pipeline, MODEL_PATH)