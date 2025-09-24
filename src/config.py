import os
from datetime import datetime
import argparse

# ==== DEFAULTS AND COLUMN CONFIGS ====

TRAIN_XLSX = os.getenv("TRAIN_XLSX", "./data/train.xlsx")
EXTERNAL_XLSX = os.getenv("EXTERNAL_XLSX", "./data/external.xlsx")
SHEET = os.getenv("SHEET", None)  # None = first sheet
RANDOM_STATE = int(os.getenv("RANDOM_STATE", 3))
TEST_SIZE = float(os.getenv("TEST_SIZE", 0.30))
CV_FOLDS = int(os.getenv("CV_FOLDS", 5))

# Try these in order for group/cultivar column detection
GROUP_COL_CANDIDATES = ["cultivar", "seed", "variety"]

SEED_COLS = [
    'base germination rate', 'base germination index', 'base germination potential', 'weight of each seed (gr)'
]
PLASMA_COLS = [
    'power (w)', 'plasma time', 'voltage (kV)', 'gas flow rate (L/min)', 'temperature'
]
GERM_COLS = [
    'germination days', 'growing temp'
]
CATEGORICAL_COLS = []  # pipeline must support adding later
POLY_COLS = ['power (w)', 'plasma time', 'voltage (kV)']

MODEL_PARAMS_GRID = {
    'model__n_estimators': [400],
    'model__max_depth': [None],
    'model__min_samples_split': [4],
    'model__min_samples_leaf': [2],
}

EXPERIMENT_NAME = "CP_Uplift_ET"
REGISTERED_MODEL_NAME = "cp_uplift_et"

# ==== METADATA HELPERS ====
def get_run_name(base="run"):
    return f"{base}_" + datetime.now().strftime("%Y%m%d_%H%M%S")

def get_tracking_uri(cli_uri=None):
    import re
    uri = cli_uri or os.getenv("MLFLOW_TRACKING_URI", None)
    if uri:
        # If on Windows and uri is a local path, convert to file:/// URI for MLflow registry compatibility
        if re.match(r"^[A-Za-z]:\\", uri):
            # Convert C:\path\to\dir to file:///C:/path/to/dir
            uri_fixed = "file:///" + uri.replace("\\", "/")
            print(f"[MLflow] Converted tracking URI to file URI for registry compatibility: {uri_fixed}")
            return uri_fixed
        return uri
    # Default: use local mlruns as file URI
    default_path = os.path.abspath("./mlruns")
    if os.name == "nt":
        uri_fixed = "file:///" + default_path.replace("\\", "/")
        print(f"[MLflow] Using default file URI for registry: {uri_fixed}")
        return uri_fixed
    return default_path

# ==== CULTIVAR COLUMN RESOLVER ====
def resolve_group_col(df, override=None):
    if override is not None:
        if override in df.columns:
            return override
        raise ValueError(f"Group column override '{override}' not found in data columns: {list(df.columns)}")
    for cand in GROUP_COL_CANDIDATES:
        if cand in df.columns:
            return cand
    raise ValueError(f"None of the group/cultivar columns found in data. Tried: {GROUP_COL_CANDIDATES}")

# ==== ARGPARSE FOR CLI ====
def get_arg_parser():
    parser = argparse.ArgumentParser(description="Train Extra Trees uplift regressor for cold plasma seed priming.")
    parser.add_argument('--train_xlsx', type=str, default=TRAIN_XLSX)
    parser.add_argument('--external_xlsx', type=str, default=EXTERNAL_XLSX)
    parser.add_argument('--experiment', type=str, default=EXPERIMENT_NAME)
    parser.add_argument('--run_name', type=str, default=None)
    parser.add_argument('--test_size', type=float, default=TEST_SIZE)
    parser.add_argument('--cv_folds', type=int, default=CV_FOLDS)
    parser.add_argument('--random_state', type=int, default=RANDOM_STATE)
    parser.add_argument('--register', action='store_true')
    parser.add_argument('--tracking_uri', type=str, default=None)
    parser.add_argument('--group_col_override', type=str, default=None)
    return parser
