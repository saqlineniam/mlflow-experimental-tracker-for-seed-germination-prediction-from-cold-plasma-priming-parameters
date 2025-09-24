# Cold Plasma Uplift Prediction with Extra Trees and MLflow

This repository provides a **fully reproducible MLflow pipeline** for training and evaluating an Extra Trees regressor to predict *germination rate uplift* under cold plasma seed priming. It is designed for research and publication, with all metrics, figures, and artifacts logged for paper-ready results.

---

## Project Structure

```
.
├── README.md
├── mlruns/                      # MLflow local tracking
├── data/
│   ├── train.xlsx               # Main training data (user supplies)
│   └── external.xlsx            # External validation data (optional)
├── outputs/                     # Figures, tables, serialized artifacts
├── env/
│   ├── conda.yaml
│   └── requirements.txt
├── mlproject/
│   └── MLproject                # MLflow CLI entrypoint
└── src/
    ├── config.py                # All config, CLI, column lists
    ├── features.py              # Preprocessing, validation
    ├── metrics.py               # Metric helpers
    ├── plots.py                 # Matplotlib-only figures
    ├── evaluate.py              # Evaluation routines
    ├── train.py                 # Main training script
    ├── registry.py              # Model registry helpers
    ├── utils.py                 # IO, reproducibility, logging
    └── __init__.py
```

---

## How to Set Up

### 1. Install Environment

**With Conda (recommended):**

```sh
conda env create -f env/conda.yaml
conda activate cp-uplift-et
```

**Or with pip:**

```sh
pip install -r env/requirements.txt
```

---

## How to Run

### 1. Start the MLflow UI (optional)

```sh
mlflow ui --backend-store-uri ./mlruns
```

### 2. Train and Evaluate

```sh
python -m src.train --train_xlsx ./data/train.xlsx --external_xlsx ./data/external.xlsx --register false
```

**Custom data paths:**

```sh
python -m src.train --train_xlsx "C:/Users/.../datasheet_final.xlsx" --external_xlsx "C:/Users/.../tesyt.xlsx"
```

**Register model to MLflow Model Registry:**

```sh
python -m src.train --register true
```

---

## Artifacts & Outputs

All outputs are saved under `./outputs` and also logged to MLflow:

- **Leaderboard**: `leaderboard.csv`
- **Figures**: Parity, residuals, OOF, calibration, per-cultivar bar charts, feature importances
- **Per-cultivar metrics**: `cultivar_metrics.csv`
- **LOGO metrics**: `logo_metrics.csv`, `logo_summary.json`
- **Feature importance**: `feature_importance.csv`, `feature_importance.png`
- **Permutation importance**: `permutation_importance.csv`, `permutation_importance.png`
- **Final model**: `model.pkl` (also registered to MLflow if `--register true`)
- **External validation**: `external_preds.csv`, `external_residuals.png` (if external.xlsx provided)

Attach these artifacts to your paper or supplement as needed.

---

## Model Registry Usage

If you pass `--register true`, the trained model will be registered under the MLflow Model Registry as `cp_uplift_et`. You can manage model stages (e.g., Staging/Production) via the MLflow UI.

---

## Citation

If you use this pipeline in your research, please cite as:

> "This work used the CP_Uplift_ET MLflow pipeline (https://github.com/saqlineniam/mlflow-experimental-tracker-for-seed-germination-prediction-from-cold-plasma-priming-parameters), an open-source framework for reproducible uplift modeling under cold plasma seed priming."

---

## Contact

For issues or questions, please open an issue or contact the maintainer.
