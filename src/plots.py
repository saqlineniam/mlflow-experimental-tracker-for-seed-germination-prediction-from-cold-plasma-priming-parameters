import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def parity_plot(y_true, y_pred, title="Parity Plot"):
    fig, ax = plt.subplots()
    ax.scatter(y_true, y_pred, alpha=0.7)
    minv = min(np.min(y_true), np.min(y_pred))
    maxv = max(np.max(y_true), np.max(y_pred))
    ax.plot([minv, maxv], [minv, maxv], 'k--', lw=1)
    ax.set_xlabel("Observed Uplift")
    ax.set_ylabel("Predicted Uplift")
    ax.set_title(title)
    return fig

def residuals_plot(y_true, y_pred, title="Residuals Plot"):
    residuals = y_true - y_pred
    fig, ax = plt.subplots()
    ax.scatter(y_pred, residuals, alpha=0.7)
    ax.axhline(0, color='k', linestyle='--', lw=1)
    ax.set_xlabel("Predicted Uplift")
    ax.set_ylabel("Residual (Observed - Predicted)")
    ax.set_title(title)
    return fig

def per_group_bar(metric_by_group_df, metric_col, title="Per-Group Metric"):
    fig, ax = plt.subplots(figsize=(8, 4))
    metric_by_group_df = metric_by_group_df.sort_values(metric_col, ascending=False)
    ax.bar(metric_by_group_df.iloc[:,0].astype(str), metric_by_group_df[metric_col])
    ax.set_xlabel("Group")
    ax.set_ylabel(metric_col)
    ax.set_title(title)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    return fig

def feature_importance_bar(names, importances, title="Feature Importance"):
    order = np.argsort(importances)[::-1]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(np.array(names)[order], np.array(importances)[order])
    ax.set_xlabel("Feature")
    ax.set_ylabel("Importance")
    ax.set_title(title)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    return fig

def oof_calibration_plot(y_true, y_pred, title="OOF Calibration Plot"):
    # Regression calibration: bin predictions, plot mean observed vs mean predicted per bin
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n_bins = 10
    bins = np.percentile(y_pred, np.linspace(0, 100, n_bins + 1))
    bin_ids = np.digitize(y_pred, bins[1:-1], right=True)
    mean_pred = [y_pred[bin_ids == i].mean() if np.any(bin_ids == i) else np.nan for i in range(n_bins)]
    mean_true = [y_true[bin_ids == i].mean() if np.any(bin_ids == i) else np.nan for i in range(n_bins)]
    fig, ax = plt.subplots()
    ax.plot(mean_pred, mean_true, marker='o')
    minv = np.nanmin([*mean_pred, *mean_true])
    maxv = np.nanmax([*mean_pred, *mean_true])
    ax.plot([minv, maxv], [minv, maxv], 'k--', lw=1)
    ax.set_xlabel("Mean predicted uplift (bin)")
    ax.set_ylabel("Mean observed uplift (bin)")
    ax.set_title(title)
    return fig
