import tensorflow as tf

model = tf.keras.models.load_model('models/exports/nlu_model.h5')

converter = tf.lite.TFLiteConverter.from_keras_model(model)

converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS, 
    tf.lite.OpsSet.SELECT_TF_OPS
]

converter._experimental_lower_tensor_list_ops = False
# ----------------------------------------

converter.optimizations = [tf.lite.Optimize.DEFAULT]

try:
    tflite_model = converter.convert()
    with open('models/tflite/nlu_model.tflite', 'wb') as f:
        f.write(tflite_model)
    print("Succès ! Le fichier .tflite est prêt pour Android.")
except Exception as e:
    print(f"Erreur lors de la conversion : {e}")