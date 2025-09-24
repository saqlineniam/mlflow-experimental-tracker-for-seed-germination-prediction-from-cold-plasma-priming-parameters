import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.pipeline import Pipeline
from sklearn.inspection import permutation_importance
import mlflow
import mlflow.sklearn
from . import config, features, metrics, plots, evaluate, registry, utils

def main():
    parser = config.get_arg_parser()
    args = parser.parse_args()
    utils.set_reproducibility(args.random_state)
    utils.safe_make_dir("./outputs")
    tracking_uri = config.get_tracking_uri(args.tracking_uri)
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(args.experiment)
    run_name = args.run_name or utils.timestamped_run_name("uplift_et")
    with mlflow.start_run(run_name=run_name) as run:
        # Log code versions and config
        mlflow.log_param("python_version", sys.version)
        mlflow.log_param("sklearn_version", __import__('sklearn').__version__)
        mlflow.log_param("numpy_version", np.__version__)
        mlflow.log_param("random_state", args.random_state)
        mlflow.log_param("test_size", args.test_size)
        mlflow.log_param("cv_folds", args.cv_folds)
        mlflow.log_param("poly_cols", config.POLY_COLS)
        mlflow.log_param("categorical_cols", config.CATEGORICAL_COLS)
        mlflow.log_param("seed_cols", config.SEED_COLS)
        mlflow.log_param("plasma_cols", config.PLASMA_COLS)
        mlflow.log_param("germ_cols", config.GERM_COLS)
        # Load train data
        df = utils.load_excel(args.train_xlsx, sheet=config.SHEET)
        # Detect group/cultivar column
        try:
            group_col = config.resolve_group_col(df, args.group_col_override)
        except Exception as e:
            mlflow.set_tag("cultivar_col", "MISSING")
            group_col = None
        else:
            mlflow.set_tag("cultivar_col", group_col)
        # Build X, y
        numeric_cols = config.SEED_COLS + config.PLASMA_COLS + config.GERM_COLS
        categorical_cols = config.CATEGORICAL_COLS
        poly_cols = config.POLY_COLS if all(c in df.columns for c in config.POLY_COLS) else []
        if not poly_cols:
            mlflow.set_tag("poly_cols_warning", "POLY_COLS missing, using numerics only")
        features.validate_required_columns(df, numeric_cols, poly_cols, categorical_cols)
        X, y = features.build_X_y(df, numeric_cols, categorical_cols, poly_cols)
        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=args.test_size, random_state=args.random_state
        )
        # Preprocessor and pipeline
        preprocessor = features.make_preprocessor(numeric_cols, categorical_cols, poly_cols)
        model = ExtraTreesRegressor(random_state=args.random_state)
        pipeline = Pipeline([
            ("pre", preprocessor),
            ("model", model)
        ])
        # GridSearchCV
        grid = GridSearchCV(
            pipeline,
            param_grid=config.MODEL_PARAMS_GRID,
            cv=KFold(args.cv_folds, shuffle=True, random_state=args.random_state),
            n_jobs=-1,
            scoring='neg_mean_squared_error',
            verbose=1
        )
        grid.fit(X_train, y_train)
        mlflow.log_params(grid.best_params_)
        best_pipe = grid.best_estimator_
        # Train/test eval
        train_metrics, test_metrics, figs, y_pred_train, y_pred_test = evaluate.train_test_eval(
            best_pipe, X_train, y_train, X_test, y_test
        )
        mlflow.log_metrics(train_metrics)
        mlflow.log_metrics(test_metrics)
        for name, fig in figs.items():
            fig_path = f"./outputs/{name}.png"
            fig.savefig(fig_path)
            mlflow.log_figure(fig, f"figures/{name}.png")
        # Leaderboard
        leaderboard = pd.DataFrame([
            {**train_metrics, **test_metrics, **grid.best_params_}
        ])
        leaderboard_path = "./outputs/leaderboard.csv"
        leaderboard.to_csv(leaderboard_path, index=False)
        mlflow.log_artifact(leaderboard_path)
        # OOF eval
        cv = KFold(args.cv_folds, shuffle=True, random_state=args.random_state)
        y_pred_oof, oof_metrics, oof_figs = evaluate.oof_eval(best_pipe, X, y, cv)
        mlflow.log_metrics(oof_metrics)
        oof_preds_path = "./outputs/oof_preds.csv"
        pd.DataFrame({"y_true": y, "y_pred_oof": y_pred_oof}).to_csv(oof_preds_path, index=False)
        mlflow.log_artifact(oof_preds_path)
        for name, fig in oof_figs.items():
            fig_path = f"./outputs/{name}.png"
            fig.savefig(fig_path)
            mlflow.log_figure(fig, f"figures/{name}.png")
        # Cultivar-aware CV
        if group_col is not None:
            cultivar_metrics_df, cultivar_figs = evaluate.cultivar_eval(best_pipe, X, y, df[group_col], cv)
            cultivar_metrics_path = "./outputs/cultivar_metrics.csv"
            cultivar_metrics_df.to_csv(cultivar_metrics_path, index=False)
            mlflow.log_artifact(cultivar_metrics_path)
            for name, fig in cultivar_figs.items():
                fig_path = f"./outputs/{name}.png"
                fig.savefig(fig_path)
                mlflow.log_figure(fig, f"figures/{name}.png")
            # LOGO
            logo_metrics_df, logo_agg_metrics, logo_folds = evaluate.logo_eval(best_pipe, X, y, df[group_col])
            logo_metrics_path = "./outputs/logo_metrics.csv"
            logo_metrics_df.to_csv(logo_metrics_path, index=False)
            mlflow.log_artifact(logo_metrics_path)
            logo_summary_path = "./outputs/logo_summary.json"
            with open(logo_summary_path, "w") as f:
                json.dump(logo_agg_metrics, f, indent=2)
            mlflow.log_artifact(logo_summary_path)
            mlflow.log_metrics({k: v for k, v in logo_agg_metrics.items() if isinstance(v, (int, float, np.floating))})
        else:
            mlflow.set_tag("logo_eval", "SKIPPED")
        # External validation
        ext_metrics = None
        if os.path.exists(args.external_xlsx):
            df_ext = utils.load_excel(args.external_xlsx, sheet=config.SHEET)
            try:
                features.validate_required_columns(df_ext, numeric_cols, poly_cols, categorical_cols)
                X_ext, y_ext = features.build_X_y(df_ext, numeric_cols, categorical_cols, poly_cols)
                ext_metrics, ext_fig, y_pred_ext = evaluate.external_eval(best_pipe, X_ext, y_ext)
                mlflow.log_metrics(ext_metrics)
                ext_preds_path = "./outputs/external_preds.csv"
                pd.DataFrame({"y_true": y_ext, "y_pred": y_pred_ext}).to_csv(ext_preds_path, index=False)
                mlflow.log_artifact(ext_preds_path)
                ext_fig_path = "./outputs/external_residuals.png"
                ext_fig.savefig(ext_fig_path)
                mlflow.log_figure(ext_fig, "figures/external_residuals.png")
            except Exception as e:
                mlflow.set_tag("external_eval", f"FAILED: {e}")
        else:
            mlflow.set_tag("external_eval", "SKIPPED: file not found")
        # Feature importance
        feature_names = best_pipe.named_steps["pre"].get_feature_names_out()
        importances = best_pipe.named_steps["model"].feature_importances_
        featimp_df = pd.DataFrame({"feature": feature_names, "importance": importances})
        featimp_path = "./outputs/feature_importance.csv"
        featimp_df.to_csv(featimp_path, index=False)
        mlflow.log_artifact(featimp_path)
        fig_fi = plots.feature_importance_bar(feature_names, importances, "Extra Trees Feature Importance")
        fig_fi_path = "./outputs/feature_importance.png"
        fig_fi.savefig(fig_fi_path)
        mlflow.log_figure(fig_fi, "figures/feature_importance.png")
        # Permutation importance
        perm = permutation_importance(best_pipe, X_test, y_test, n_repeats=10, random_state=args.random_state, n_jobs=-1)
        import warnings
        fnames = feature_names
        importances = perm.importances_mean
        if len(fnames) != len(importances):
            minlen = min(len(fnames), len(importances))
            warnings.warn(f"Permutation importance: feature_names ({len(fnames)}) and importances ({len(importances)}) length mismatch. Trimming to {minlen}.")
            fnames = fnames[:minlen]
            importances = importances[:minlen]
        perm_df = pd.DataFrame({"feature": fnames, "importance": importances})
        perm_path = "./outputs/permutation_importance.csv"
        perm_df.to_csv(perm_path, index=False)
        mlflow.log_artifact(perm_path)
        fig_perm = plots.feature_importance_bar(fnames, importances, "Permutation Importance")
        fig_perm_path = "./outputs/permutation_importance.png"
        fig_perm.savefig(fig_perm_path)
        mlflow.log_figure(fig_perm, "figures/permutation_importance.png")
        # Model signature and registration
        input_example = X_test.head(5)
        signature = registry.infer_model_signature(best_pipe, input_example)
        model_info = registry.log_model_and_register(
            run, best_pipe, signature, input_example, registered_model_name=config.REGISTERED_MODEL_NAME, register=args.register
        )
        # Save model to outputs
        import joblib
        model_pkl_path = "./outputs/model.pkl"
        joblib.dump(best_pipe, model_pkl_path)
        mlflow.log_artifact(model_pkl_path)
        # Log summary
        mlflow.set_tags({
            "target": "uplift",
            "model": "ExtraTrees",
            "cv": f"KFold{args.cv_folds}",
            "poly_cols": ",".join(poly_cols),
            "cultivar_col": group_col or "MISSING",
            "n_params": best_pipe.named_steps["model"].get_params().__len__(),
            "n_features_after_transform": len(feature_names),
            "n_samples_train": len(X_train),
            "n_samples_test": len(X_test),
        })
        print("\n=== RUN SUMMARY ===")
        print(f"Run ID: {run.info.run_id}")
        print(f"Test RMSE: {test_metrics['test_RMSE']:.3f}")
        print(f"Test MAE: {test_metrics['test_MAE']:.3f}")
        print(f"Test R2: {test_metrics['test_R2']:.3f}")
        print(f"Artifacts saved in ./outputs and logged to MLflow under experiment '{args.experiment}'")

if __name__ == "__main__":
    main()
