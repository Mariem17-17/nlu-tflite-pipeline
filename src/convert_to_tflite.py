import tensorflow as tf
import os

os.makedirs("models/tflite", exist_ok=True)

model = tf.keras.models.load_model("models/exports/nlu_model.h5")

converter = tf.lite.TFLiteConverter.from_keras_model(model)

converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,
    tf.lite.OpsSet.SELECT_TF_OPS
]
converter._experimental_lower_tensor_list_ops = False

tflite_model = converter.convert()

with open("models/tflite/nlu_model.tflite", "wb") as f:
    f.write(tflite_model)

print("TFLite model exported successfully.")
