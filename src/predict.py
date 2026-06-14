import joblib
import pandas as pd

def predict(data):
    model = joblib.load('models/disease_lgbm_model.pkl')
    df = pd.DataFrame([data])
    pred = model.predict(df)[0]
    pred_proba = model.predict_proba(df)[0][1]

    return pred, pred_proba