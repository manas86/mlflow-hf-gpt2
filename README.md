# mlflow-hf-gpt2
HUggingface GPT-2 Text Generation with MLflow


## Introduction
This README provides detailed information about the 

- Python source code dir included in this project.
- Python unit tests included in this project.
  
## Project Structure for Source code:

## Overview train.py
This project demonstrates text generation using the GPT-2 model and logs the generated text along with other metadata using MLflow. The generated text is based on a set of predefined prompts.

## Prerequisites
Before running the code, make sure you have the following dependencies installed:

- `mlflow`: Machine learning lifecycle management library.
- `transformers`: Hugging Face Transformers library for pre-trained language models.

You can install these dependencies using pip:

```bash
pip install mlflow transformers
```

## Usage
1. Load the GPT-2 model and tokenizer.
2. Define an experiment name and run name for MLflow.
3. Define a function to generate text based on a given prompt.
4. Create a custom PythonModel class to integrate with MLflow.
5. Define a list of prompts for text generation.
6. In the `main` function:
   - Set the experiment name and create an MLflow run.
   - Log parameters and generate text for each prompt.
   - Log the generated text as an artifact.
   - Log the length of the generated text as a metric.
   - Save the model using MLflow.
   - Register the model in the MLflow model registry.

To run the code, execute the following command:

```bash
python train.py
```

## Project Structure
The project includes the following files:
- `train.py`: The main Python script for text generation and MLflow integration.
- `generated_text_prompt_*.txt`: Text files containing the generated text for each prompt (generated during execution). 
- `prediction_prompt_*.txt`: Text files containing the generated text for each prompt, logged as predictions (generated during execution).

## Custom PythonModel Class
A custom `PythonModel` class (`GPT2Model`) is defined to integrate with MLflow. This class loads the pre-trained GPT-2 model during loading and generates text based on the provided input.

## Running the Code
Ensure you have the necessary dependencies installed and execute the script as described in the "Usage" section. The generated text and model artifacts will be logged and tracked by MLflow.

## Overview transition_model.py

MLflow Model Deployment Script
This script demonstrates how to use the MLflow Python API to transition a specific version of a registered model to the "Production" stage. The script assumes that you have already registered a model with a given name and version in your MLflow tracking server.

## Prerequisites
Before running the script, make sure you have the following dependencies installed:

- `mlflow`: Machine learning lifecycle management library.

You can install this dependency using pip:

```bash
pip install mlflow
```

## Usage
1. Import the necessary modules and create a `MlflowClient` object.
2. Specify the name of the registered model (`registered_model_name`) and the desired model version (`model_version`) that you want to transition to the "Production" stage.
3. Uncomment the line `# model_name = mlflow.pyfunc.load_model(f"models:/{registered_model_name}/{model_version}")` if you need to load the model. This line is currently commented out.
4. Use the `client.transition_model_version_stage` method to transition the specified model version to the "Production" stage.

To run the script, execute the following command:

```bash
python transition_model.py
```

## Script Structure
- `transition_model.py`: The main Python script for transitioning a model version to the "Production" stage.


## Running the Script
Ensure you have the `mlflow` dependency installed and execute the script as described in the "Usage" section. The specified model version will be transitioned to the "Production" stage in your MLflow tracking server.

## Important Notes
- This script assumes that you have already registered the model with the specified name and version in MLflow.
- Make sure you have appropriate permissions to transition model versions to the "Production" stage in your MLflow environment.


## Project Structure for Test: 
The project consists of the following files and directories:

- `train.py`: Contains the main code for training a GPT-2 model.
- `README.txt`: This file, provides information about the unit tests.
- `test_train.py`: Unit test script for testing the `generate_text`, `GPT2Model`, and `main` functions in `train.py`.

## Running the Unit Tests
To run the unit tests, make sure you have the required dependencies installed. You can install them using pip:

```bash
pip install mlflow transformers
```

Once you have the dependencies installed, you can run the unit tests by executing the following command from the project root directory:

```bash
python -m unittest test_train.py
```

## Unit Test Descriptions

### `TestGenerateText` Class
- `test_generate_text`: Tests the `generate_text` function by generating text based on a prompt and checking if the result is a string.

### `TestGPT2Model` Class
- `test_load_context`: Tests the `load_context` method of the `GPT2Model` class by mocking the model creation and verifying that the model was created.
- `test_predict`: Tests the `predict` method of the `GPT2Model` class by mocking model predictions and checking if the result is a string.

### `TestMainFunction` Class
- `test_main`: Tests the `main` function by mocking MLflow functions (`set_experiment`, `start_run`, `register_model`, `log_model`, `end_run`) and checking if they were called during the execution of `main`.

## Usage
You can use these unit tests to ensure the correctness of the functions in `train.py` before deploying the GPT-2 model. Simply run the tests, and if all tests pass without errors, your code should be in good shape.

## Dependencies
- `mlflow`: Machine learning lifecycle management library.
- `transformers`: Hugging Face Transformers library for pre-trained language models.

## Use the MLflow UI to view and manage the model trained and stored

To use the MLflow UI to view and manage the model you trained and stored previously, you can follow these steps:

1. **Start MLflow Tracking Server**:

   Before you can use the MLflow UI, you need to start the MLflow Tracking Server. You can start it from the command line using the following command:

   ```bash
   mlflow server
   ```

   By default, the server should start on `localhost` at port `5000`.

2. **Access the MLflow UI**:

   Open a web browser and navigate to the MLflow UI by entering the following URL:

   ```
   http://localhost:5000
   ```

   If you started the server with different host and port configurations, adjust the URL accordingly.

3. **View Your MLflow Experiment**:

   Once you access the MLflow UI, you'll see the MLflow Tracking Dashboard. It will list all the experiments you've run. Locate the experiment where you saved your GPT-2 model, and click on it to view its details.

4. **Explore Run Information**:

   Inside the experiment, you'll see a list of runs. Each run corresponds to a specific execution of your code. You can click on a run to explore its details, including the parameters, metrics, and artifacts that you logged during that run.
   ![image](https://github.com/manas86/mlflow-hf-gpt2/assets/30902765/fd41d922-2d72-4151-bc02-f38c0e3b5d3f)

   ![image](https://github.com/manas86/mlflow-hf-gpt2/assets/30902765/42999b64-f8eb-4fbd-bdfa-c0d7507ab97a)


Using the MLflow UI, you can easily monitor and manage your experiments, view run details, and access the artifacts, including the saved GPT-2 model. This provides a convenient way to track and share your machine learning experiments with your team or collaborators.

