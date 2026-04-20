import numpy as np
import json
import tensorflow as tf

TFLITE_MODEL_PATH = "models/tflite/nlu_model.tflite"
VOCAB_PATH = "models/exports/vocab.json"
MAP_PATH = "models/exports/intent_map.json"
MAX_LEN = 20

def test_nlu(sentence):
    interpreter = tf.lite.Interpreter(model_path=TFLITE_MODEL_PATH)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    with open(VOCAB_PATH, "r", encoding="utf-8") as f:
        vocab = json.load(f)

    with open(MAP_PATH, "r", encoding="utf-8") as f:
        intent_map = json.load(f)

    id_to_intent = {v: k for k, v in intent_map.items()}

    words = sentence.lower().split()
    sequence = [vocab.get(w, vocab.get("<OOV>", 0)) for w in words]

    padded = sequence[:MAX_LEN] + [0] * (MAX_LEN - len(sequence))
    input_data = np.array([padded], dtype=np.int32)

    interpreter.set_tensor(input_details[0]["index"], input_data)
    interpreter.invoke()

    output_data = interpreter.get_tensor(output_details[0]["index"])
    pred_id = int(np.argmax(output_data))
    confidence = float(output_data[0][pred_id])

    print(f"\nSentence: {sentence}")
    print(f"Intent: {id_to_intent[pred_id]}")
    print(f"Confidence: {confidence * 100:.2f}%")

if __name__ == "__main__":
    test_nlu("call ahmed")
    test_nlu("play llama three explained on youtube")