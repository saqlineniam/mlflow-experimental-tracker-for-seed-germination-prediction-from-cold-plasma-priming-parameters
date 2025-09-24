import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_predict, KFold, LeaveOneGroupOut
from .metrics import metrics_dict
from .plots import parity_plot, residuals_plot, per_group_bar, oof_calibration_plot

# ---- TRAIN/TEST EVAL ----
def train_test_eval(pipeline, X_train, y_train, X_test, y_test):
    pipeline.fit(X_train, y_train)
    y_pred_train = pipeline.predict(X_train)
    y_pred_test = pipeline.predict(X_test)
    train_metrics = metrics_dict("train", y_train, y_pred_train)
    test_metrics = metrics_dict("test", y_test, y_pred_test)
    figures = {
        "parity_test": parity_plot(y_test, y_pred_test, "Test Parity Plot"),
        "residuals_test": residuals_plot(y_test, y_pred_test, "Test Residuals Plot"),
    }
    return train_metrics, test_metrics, figures, y_pred_train, y_pred_test

# ---- OOF EVAL ----
def oof_eval(pipeline, X, y, cv):
    y_pred_oof = cross_val_predict(pipeline, X, y, cv=cv, method='predict', n_jobs=-1)
    oof_metrics = metrics_dict("oof", y, y_pred_oof)
    figures = {
        "parity_oof": parity_plot(y, y_pred_oof, "OOF Parity Plot"),
        "residuals_oof": residuals_plot(y, y_pred_oof, "OOF Residuals Plot"),
        "calibration_oof": oof_calibration_plot(y, y_pred_oof, "OOF Calibration Plot"),
    }
    return y_pred_oof, oof_metrics, figures

# ---- CULTIVAR-AWARE EVAL ----
def cultivar_eval(pipeline, X, y, cultivar_series, cv):
    groups = cultivar_series.values
    y_pred_oof = cross_val_predict(pipeline, X, y, cv=cv, groups=groups, method='predict', n_jobs=-1)
    df = pd.DataFrame({
        'cultivar': groups,
        'y_true': y,
        'y_pred': y_pred_oof
    })
    agg = df.groupby('cultivar').agg(
        count = ('y_true', 'count'),
        MAE = ('y_true', lambda x: np.mean(np.abs(x - df.loc[x.index, 'y_pred']))),
        RMSE = ('y_true', lambda x: np.sqrt(np.mean((x - df.loc[x.index, 'y_pred'])**2))),
        R2 = ('y_true', lambda x: 1 - np.sum((x - df.loc[x.index, 'y_pred'])**2) / np.sum((x - x.mean())**2) if len(x) > 1 else np.nan),
        y_true_mean = ('y_true', 'mean'),
        y_pred_mean = ('y_pred', 'mean'),
    ).reset_index()
    figures = {
        'cultivar_mae': per_group_bar(agg, 'MAE', 'Per-Cultivar MAE'),
        'cultivar_rmse': per_group_bar(agg, 'RMSE', 'Per-Cultivar RMSE'),
        'cultivar_r2': per_group_bar(agg, 'R2', 'Per-Cultivar R2'),
    }
    return agg, figures

# ---- LOGO EVAL ----
def logo_eval(pipeline, X, y, groups):
    logo = LeaveOneGroupOut()
    metrics_list = []
    fold_results = []
    for train_idx, test_idx in logo.split(X, y, groups):
        X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
        y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]
        pipeline.fit(X_tr, y_tr)
        y_pred = pipeline.predict(X_te)
        fold_metrics = metrics_dict("logo", y_te, y_pred)
        fold_metrics["group"] = groups[test_idx[0]]
        metrics_list.append(fold_metrics)
        fold_results.append({"group": groups[test_idx[0]], "y_true": y_te.values, "y_pred": y_pred})
    metrics_df = pd.DataFrame(metrics_list)
    agg_metrics = metrics_df.mean(numeric_only=True).to_dict()
    agg_metrics.update({f"{k}_std": metrics_df[k].std() for k in ['logo_RMSE', 'logo_MAE', 'logo_R2'] if k in metrics_df})
    return metrics_df, agg_metrics, fold_results

# ---- EXTERNAL EVAL ----
def external_eval(pipeline, X_ext, y_ext):
    y_pred_ext = pipeline.predict(X_ext)
    ext_metrics = metrics_dict("ext", y_ext, y_pred_ext)
    fig = residuals_plot(y_ext, y_pred_ext, "External Residuals Plot")
    return ext_metrics, fig, y_pred_ext
