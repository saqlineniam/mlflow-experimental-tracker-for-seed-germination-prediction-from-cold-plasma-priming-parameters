import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import PowerTransformer, PolynomialFeatures, StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline

from .config import POLY_COLS, CATEGORICAL_COLS

# Helper: Build preprocessor

def make_preprocessor(numeric_cols, categorical_cols=None, poly_cols=None):
    if categorical_cols is None:
        categorical_cols = []
    if poly_cols is None:
        poly_cols = []
    num_cols = [c for c in numeric_cols if c not in poly_cols]
    transformers = []
    if poly_cols:
        poly_pipe = Pipeline([
            ("power", PowerTransformer(method="yeo-johnson", standardize=False)),
            ("poly", PolynomialFeatures(degree=2, include_bias=False)),
            ("scaler", StandardScaler()),
        ])
        transformers.append(("poly_block", poly_pipe, poly_cols))
    if num_cols:
        transformers.append(("num_block", StandardScaler(), num_cols))
    if categorical_cols:
        transformers.append((
            "cat_block",
            OneHotEncoder(sparse_output=False, handle_unknown="ignore"),
            categorical_cols,
        ))
    preprocessor = ColumnTransformer(transformers, remainder="drop", verbose_feature_names_out=False)
    return preprocessor

# Helper: Compute uplift and build X, y

def build_X_y(df, numeric_cols, categorical_cols=None, poly_cols=None):
    # Validate columns
    required = set(numeric_cols)
    required.add('germination rate')
    required.add('base germination rate')
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    X = df[numeric_cols]
    if categorical_cols:
        for col in categorical_cols:
            if col not in df.columns:
                raise ValueError(f"Categorical column '{col}' missing from data.")
        X = pd.concat([X, df[categorical_cols]], axis=1)
    y = df['germination rate'] - df['base germination rate']
    return X, y

# Helper: Validate required columns (for external use)
def validate_required_columns(df, numeric_cols, poly_cols=None, categorical_cols=None):
    required = set(numeric_cols)
    if poly_cols:
        required |= set(poly_cols)
    if categorical_cols:
        required |= set(categorical_cols)
    required.add('germination rate')
    required.add('base germination rate')
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
