# diagnostic.py - Run this to identify the issue
import tensorflow as tf
from pathlib import Path
import joblib
import os

BASE_DIR = Path(__file__).parent

print("=== DIAGNOSTIC SCRIPT ===")
print(f"Current directory: {BASE_DIR}")

# Check if model files exist
model_files = [
    "../models/lstm_final.keras",
    "../models/cnn_final.keras",
    "../models/hybrid_final.keras",
    "../models/preprocessor.joblib",
]

print("\n=== FILE EXISTENCE CHECK ===")
for file_path in model_files:
    full_path = BASE_DIR / file_path
    exists = full_path.exists()
    print(f"{file_path}: {'EXISTS' if exists else 'MISSING'}")
    if exists:
        print(f"  Size: {full_path.stat().st_size} bytes")

print("\n=== ATTEMPTING TO LOAD MODELS ===")

# Test loading each model individually
for model_name in ["lstm_final.keras", "cnn_final.keras", "hybrid_final.keras"]:
    try:
        model_path = BASE_DIR / f"../models/{model_name}"
        print(f"\nLoading {model_name}...")

        model = tf.keras.models.load_model(model_path)

        print(f"  SUCCESS: {type(model)}")
        print(f"  Has predict method: {hasattr(model, 'predict')}")
        print(f"  Is dict: {isinstance(model, dict)}")

        if hasattr(model, "input_shape"):
            print(f"  Input shape: {model.input_shape}")

    except Exception as e:
        print(f"  FAILED: {e}")

print("\n=== TESTING PREPROCESSOR ===")
try:
    preprocessor_path = BASE_DIR / "../models/preprocessor.joblib"
    preprocessor = joblib.load(preprocessor_path)

    print(f"Preprocessor type: {type(preprocessor)}")
    print(f"Preprocessor is dict: {isinstance(preprocessor, dict)}")
    print(f"Preprocessor has predict: {hasattr(preprocessor, 'predict')}")

except Exception as e:
    print(f"Preprocessor loading failed: {e}")

print("\n=== TENSORFLOW INFO ===")
print(f"TensorFlow version: {tf.__version__}")
print(f"Keras version: {tf.keras.__version__}")

print("\n=== RECOMMENDATIONS ===")
print("1. Check if all model files exist and are not corrupted")
print("2. Verify you're loading .keras files with tf.keras.models.load_model()")
print("3. Verify you're loading .joblib files with joblib.load()")
print("4. Make sure you're not accidentally swapping model and preprocessor variables")
