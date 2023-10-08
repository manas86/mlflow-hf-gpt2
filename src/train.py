import mlflow
from datetime import datetime
import mlflow.pyfunc
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load the GPT-2 model and tokenizer
model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# Define names for tensorboard logging and mlflow
experiment_name = "mlflow_llm"
run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
await_creation_for = 60 

# Define a function to generate text
def generate_text(prompt, max_length=50):
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    output = model.generate(input_ids, max_length=max_length, num_return_sequences=1, pad_token_id=tokenizer.eos_token_id)
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    return generated_text

# Create a custom PythonModel class
class GPT2Model(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        self.model = model  # Load the GPT-2 model during loading

    def predict(self, context, model_input):
        # Generate text based on the provided prompt
        return generate_text(model_input, max_length=100) 

# List of prompts
prompts = [
    "Once upon a time,",
    "In a galaxy far, far away,",
    "To be or not to be,",
    "It was a dark and stormy night,",
    "The sun was setting over the horizon,",
    "She looked into his eyes and said,",
]

def main():
    # Set the experiment name and create an MLflow run
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name = run_name) as mlflow_run:
        # Log parameters
        mlflow.log_param("max_length", 100)

        # Generate text and log for each prompt
        for i, prompt in enumerate(prompts):
            generated_text = generate_text(prompt, max_length=100)  # Adjust max_length as needed
            artifact_name = f"generated_text_prompt_{i}.txt"

            # Log the generated text as an artifact using mlflow.log_artifact
            with open(artifact_name, "w") as file:
                file.write(generated_text)

            # Log the generated text as an artifact
            mlflow.log_text(generated_text, artifact_name)

            mlflow.log_artifact(artifact_name)

            # Log the length of the generated text as a metric
            text_length = len(generated_text)
            mlflow.log_metric(f"prompt_{i}_text_length", text_length)

            # Log the generated text as a prediction
            mlflow.log_text(generate_text(prompt, max_length=100), f"prediction_prompt_{i}.txt")

            mlflow_run_id = mlflow_run.info.run_id
            print(f"MLFlow Run ID {i}: {mlflow_run_id}")

        # Save the model using MLflow
        mlflow.pyfunc.log_model(artifact_path="gpt2_model", python_model=GPT2Model())

        # Register the model in the MLflow model registry
        # Name for the registered model
        register_model_name = "gpt2_model"
        # Specify a version for the registered model
        model_version = "1"  
        registered_model_uri = mlflow.register_model(f"runs:/{mlflow.active_run().info.run_id}/gpt2_model", register_model_name, model_version)

        # End the MLflow run
        mlflow.end_run()

    print(f"Model '{register_model_name}' version {model_version} registered successfully with uri {registered_model_uri}.")

if __name__ == "__main__":
    main()
