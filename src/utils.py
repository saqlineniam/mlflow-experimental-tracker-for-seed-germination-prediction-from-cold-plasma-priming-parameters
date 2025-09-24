import os
import numpy as np
import pandas as pd
import random
import joblib
import logging
from datetime import datetime

def load_excel(path, sheet=None):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Excel file not found: {path}")
    df = pd.read_excel(path, sheet_name=sheet)
    if isinstance(df, dict):
        # Multiple sheets, default to first
        first = next(iter(df.values()))
        return first
    return df

def safe_make_dir(path):
    os.makedirs(path, exist_ok=True)

def timestamped_run_name(base="run"):
    return f"{base}_" + datetime.now().strftime("%Y%m%d_%H%M%S")

def set_reproducibility(seed=3):
    np.random.seed(seed)
    random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except ImportError:
        pass
    joblib.parallel.random_state = seed

def setup_logging(log_path=None):
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s',
        handlers=[logging.StreamHandler()] + ([logging.FileHandler(log_path)] if log_path else [])
    )
