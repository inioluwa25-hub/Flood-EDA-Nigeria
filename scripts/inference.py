import tensorflow as tf
import numpy as np
import joblib

model = tf.keras.models.load_model("final_flood_model.keras")
scaler = joblib.load("scaler.joblib")


def predict_flood(input_data):
    scaled = scaler.transform(np.array(input_data).reshape(1, -1))
    return model.predict(scaled.reshape(1, 4, 1))[0][0]
