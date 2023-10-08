from mlflow.tracking import MlflowClient
import mlflow.pyfunc
client = MlflowClient()

registered_model_name = "gpt2_model"
model_version = 1
# model_name = mlflow.pyfunc.load_model(f"models:/{registered_model_name}/{model_version}")


client.transition_model_version_stage(
    name=registered_model_name,
    version=model_version,
    stage="Production"
)
