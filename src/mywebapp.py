from flask import Flask, request, jsonify
import mlflow.pyfunc

app = Flask(__name__)

# Load the registered model
registered_model_name = "gpt2_model"  
model = mlflow.pyfunc.load_model(f"models:/{registered_model_name}/1")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Input data is provided as JSON
        data = request.get_json()  
        input_data = data.get("input_data", "")

        # Make predictions using the loaded model
        predictions = model.predict(input_data)

        return jsonify({"predictions": predictions.tolist()})
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(host="localhost", port=7000)
