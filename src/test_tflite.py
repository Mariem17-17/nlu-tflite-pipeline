import numpy as np
import json
import tensorflow as tf
import os

TFLITE_MODEL_PATH = 'models/tflite/nlu_model.tflite'
VOCAB_PATH = 'models/exports/vocab.json'
MAP_PATH = 'models/exports/intent_map.json'

def test_nlu(sentence):
    print(f"\n--- Initialisation du test TFLite ---")
    
    try:
        interpreter = tf.lite.Interpreter(model_path=TFLITE_MODEL_PATH)
        
        interpreter.allocate_tensors()
        
    except RuntimeError as e:
        print(f"Tentative de secours : Activation manuelle du Flex Delegate...")
        from tensorflow.python.framework import ops
        ops.load_library(None)
        interpreter = tf.lite.Interpreter(model_path=TFLITE_MODEL_PATH)
        interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    with open(VOCAB_PATH, 'r', encoding='utf-8') as f:
        vocab = json.load(f)
    
    with open(MAP_PATH, 'r', encoding='utf-8') as f:
        intent_map = json.load(f)
    
    id_to_intent = {v: k for k, v in intent_map.items()}

    words = sentence.lower().split()
    sequence = [vocab.get(w, 0) for w in words] # 0 si le mot est inconnu
    
    padded_sequence = sequence[:20] + [0] * (20 - len(sequence))
    
    input_data = np.array([padded_sequence], dtype=np.float32)

    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    output_data = interpreter.get_tensor(output_details[0]['index'])
    predicted_id = np.argmax(output_data)
    confidence = output_data[0][predicted_id]

    print(f"\nPhrase : '{sentence}'")
    print(f"Résultat : {id_to_intent[predicted_id].upper()}")
    print(f"Confiance : {confidence * 100:.2f}%")
    print(f"-------------------------------------\n")

if __name__ == "__main__":
    test_nlu("Appelle Ahmed")
    
    test_nlu("Ouvre l'application YouTube s'il te plaît")