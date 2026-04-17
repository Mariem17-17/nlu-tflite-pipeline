import numpy as np
import json
import tensorflow as tf

# On utilise le modèle .h5 qui n'a pas besoin de Flex Delegate sur PC
MODEL_PATH = 'models/exports/nlu_model.h5'
VOCAB_PATH = 'models/exports/vocab.json'
MAP_PATH = 'models/exports/intent_map.json'

def quick_test(sentence):
    # 1. Charger le modèle
    model = tf.keras.models.load_model(MODEL_PATH)
    
    # 2. Charger les assets
    with open(VOCAB_PATH, 'r') as f: vocab = json.load(f)
    with open(MAP_PATH, 'r') as f: intent_map = json.load(f)
    id_to_intent = {v: k for k, v in intent_map.items()}

    # 3. Prétraitement
    words = sentence.lower().split()
    sequence = [vocab.get(w, 0) for w in words]
    padded = sequence[:20] + [0] * (20 - len(sequence))
    
    # 4. Prédiction
    pred = model.predict(np.array([padded]))
    idx = np.argmax(pred)
    
    print(f"\nTest : {sentence}")
    print(f"Résultat : {id_to_intent[idx]} ({np.max(pred)*100:.2f}%)")

quick_test("Call Ahmed")
quick_test("play the song dark horse on youtube")