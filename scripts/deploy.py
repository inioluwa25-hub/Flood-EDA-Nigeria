# flood_prediction_api_fixed.py
from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
import joblib
from pathlib import Path
import pandas as pd
import traceback
from pyngrok import ngrok

app = Flask(__name__)

# Configuration
BASE_DIR = Path(__file__).parent
MODEL_DIR = BASE_DIR / "../models"
PORT = 5001

# Updated input dimensions
MODEL_INPUT_SHAPES = {
    "lstm": (1, 268),  # (timesteps, features)
    "cnn": (4, 268),
    "hybrid": (7, 268, 1),
}


class FloodPredictor:
    def __init__(self):
        self.models = {}
        self.preprocessor = None
        self.loaded = False

    def load_assets(self):
        """Load all required models and preprocessor"""
        try:
            # Load preprocessor
            preprocessor_path = MODEL_DIR / "preprocessor.joblib"
            if not preprocessor_path.exists():
                raise FileNotFoundError(
                    f"Preprocessor not found at {preprocessor_path}"
                )

            self.preprocessor = joblib.load(preprocessor_path)
            print("✅ Preprocessor loaded successfully")

            # Load models with proper test inputs
            model_files = {
                "lstm": "lstm_final.keras",
                "cnn": "cnn_final.keras",
                "hybrid": "hybrid_final.keras",
            }

            for name, filename in model_files.items():
                model_path = MODEL_DIR / filename
                if not model_path.exists():
                    print(f"⚠️ Model {name} not found at {model_path}")
                    continue

                try:
                    model = tf.keras.models.load_model(model_path)

                    # Create model-specific test input
                    if name == "hybrid":
                        test_input = np.random.random((1, *MODEL_INPUT_SHAPES[name]))
                    elif name == "cnn":
                        test_input = np.random.random((1, *MODEL_INPUT_SHAPES[name]))
                    else:  # LSTM
                        test_input = np.random.random(
                            (1, 1, MODEL_INPUT_SHAPES[name][1])
                        )

                    # Test prediction
                    _ = model.predict(test_input)

                    self.models[name] = model
                    print(f"✅ Model {name} loaded and validated")
                    print(f"Input shape: {model.input_shape}")

                except Exception as e:
                    print(f"❌ Failed to load {name} model: {str(e)}")
                    continue

            self.loaded = bool(self.models) and (self.preprocessor is not None)
            return self.loaded

        except Exception as e:
            print(f"❌ Initialization failed: {str(e)}")
            traceback.print_exc()
            return False

    def preprocess_input(self, data):
        """Convert input data to model-ready format"""
        try:
            input_df = pd.DataFrame([data])
            processed = self.preprocessor.transform(input_df)
            print(f"Preprocessed shape: {processed.shape}")
            print(f"Feature count: {processed.shape[1]}")
            return processed
        except Exception as e:
            raise ValueError(f"Input processing error: {str(e)}")

    def reshape_for_model(self, processed_input, model_name):
        """Reshape input for specific model requirements"""
        try:
            print(f"\nReshaping for {model_name}")
            print(f"Input shape: {processed_input.shape}")

            # Ensure we have exactly 268 features
            if processed_input.shape[1] > 268:
                processed_input = processed_input[:, :268]
            elif processed_input.shape[1] < 268:
                padding = np.zeros(
                    (processed_input.shape[0], 268 - processed_input.shape[1])
                )
                processed_input = np.hstack([processed_input, padding])

            if model_name == "hybrid":
                # Create 7 timesteps of the same data
                repeated = np.repeat(processed_input[:, np.newaxis, :], 7, axis=1)
                # Add channel dimension
                reshaped = repeated[:, :, :, np.newaxis]
            elif model_name == "cnn":
                # Create 4 timesteps
                reshaped = np.repeat(processed_input[:, np.newaxis, :], 4, axis=1)
            else:  # LSTM
                reshaped = processed_input.reshape(1, 1, 268)

            print(f"Output shape: {reshaped.shape}")
            return reshaped

        except Exception as e:
            print(f"Reshape error: {str(e)}")
            traceback.print_exc()
            raise ValueError(f"Reshaping failed: {str(e)}")

    def predict(self, data, model_name="hybrid"):
        """Make prediction with specified model"""
        if not self.loaded:
            raise RuntimeError("Models not loaded")

        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not available")

        try:
            processed_input = self.preprocess_input(data)
            print(f"Processed input shape before reshaping: {processed_input.shape}")
            model_input = self.reshape_for_model(processed_input, model_name)
            print(f"Model input shape for {model_name}: {model_input.shape}")

            model = self.models[model_name]
            prediction = model.predict(model_input, verbose=0)

            probability = float(
                prediction[0][0] if hasattr(prediction[0], "__len__") else prediction[0]
            )

            return {
                "probability": probability,
                "prediction": "Flood likely" if probability > 0.5 else "Flood unlikely",
            }
        except Exception as e:
            traceback.print_exc()  # Add this for debugging
            raise RuntimeError(f"Prediction failed: {str(e)}")


# Initialize predictor
predictor = FloodPredictor()


@app.route("/predict", methods=["POST"])
def predict_endpoint():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No input data", "status": "failed"}), 400

        model_choice = data.pop("model", "hybrid").lower()
        result = predictor.predict(data, model_choice)

        return jsonify({"status": "success", "model_used": model_choice, **result})

    except ValueError as e:
        return jsonify({"error": str(e), "status": "failed"}), 400
    except RuntimeError as e:
        return jsonify({"error": str(e), "status": "failed"}), 500
    except Exception as e:
        return (
            jsonify({"error": f"Unexpected error: {str(e)}", "status": "failed"}),
            500,
        )


@app.route("/health", methods=["GET"])
def health_check():
    return jsonify(
        {
            "status": "ready" if predictor.loaded else "loading",
            "models_loaded": list(predictor.models.keys()),
            "preprocessor_loaded": predictor.preprocessor is not None,
        }
    )


def start_ngrok():
    # Kill any existing tunnels
    ngrok.kill()

    # Start new HTTP tunnel
    tunnel = ngrok.connect(PORT, bind_tls=True)
    public_url = tunnel.public_url
    print(f"\nNgrok Tunnel URL: {public_url}")
    print(f"Forwarding to: http://localhost:{PORT}")
    return public_url


if __name__ == "__main__":
    print("Starting Flood Prediction API...")

    if not predictor.load_assets():
        print("❌ Failed to load models. Exiting.")
        exit(1)

    try:
        public_url = start_ngrok()
        print(f"\nAPI available at: {public_url}/predict")
    except Exception as e:
        print(f"⚠️ Ngrok not started: {str(e)}")
        print(f"Local URL: http://localhost:{PORT}/predict")

    app.run(host="0.0.0.0", port=PORT)
