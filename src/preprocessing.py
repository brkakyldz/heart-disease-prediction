import pandas as pd
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NUM_FEATURES = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']

def load_data(path):
    df = pd.read_csv (path)
    df = df.drop_duplicates()
    return df

def build_preprocessor():
    return ColumnTransformer([
    ('num', StandardScaler(), NUM_FEATURES),
],  remainder='passthrough')
    