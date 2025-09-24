import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def mae(y_true, y_pred):
    return mean_absolute_error(y_true, y_pred)

def mape(y_true, y_pred):
    y_true = np.clip(np.array(y_true), 1e-8, None)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def r2(y_true, y_pred):
    return r2_score(y_true, y_pred)

def metrics_dict(prefix, y_true, y_pred):
    return {
        f"{prefix}_RMSE": rmse(y_true, y_pred),
        f"{prefix}_MAE": mae(y_true, y_pred),
        f"{prefix}_MAPE": mape(y_true, y_pred),
        f"{prefix}_R2": r2(y_true, y_pred),
    }
