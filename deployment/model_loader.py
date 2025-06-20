import joblib
import tensorflow as tf
import numpy as np


def load_model():
    model = tf.keras.models.load_model("models/final_flood_model.keras")
    scaler = joblib.load("models/scaler.joblib")
    return model, scaler


def predict(model, scaler, features):
    scaled = scaler.transform(np.array(features).reshape(1, -1))
    return model.predict(scaled.reshape(1, 4, 1))[0][0]
