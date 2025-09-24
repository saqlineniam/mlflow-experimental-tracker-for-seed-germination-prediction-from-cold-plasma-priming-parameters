import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature

# Infer model signature using a sample of preprocessed input
def infer_model_signature(pipeline, X_sample, y_sample=None):
    y_pred_sample = pipeline.predict(X_sample)
    return infer_signature(X_sample, y_pred_sample)

# Log model and register in MLflow Model Registry
def log_model_and_register(run, pipeline, signature, input_example_df, registered_model_name="cp_uplift_et", register=False, transition_stage=None):
    model_info = mlflow.sklearn.log_model(
        sk_model=pipeline,
        artifact_path="model",
        signature=signature,
        input_example=input_example_df,
        registered_model_name=registered_model_name if register else None
    )
    if register and transition_stage is not None:
        client = mlflow.tracking.MlflowClient()
        latest = client.get_latest_versions(registered_model_name, stages=["None"])
        if latest:
            model_version = latest[0].version
            client.transition_model_version_stage(
                name=registered_model_name,
                version=model_version,
                stage=transition_stage,
            )
    return model_info
