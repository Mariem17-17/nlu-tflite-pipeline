import json
import numpy as np
import os
from preprocess import NLUPreprocessor
from architecture import build_nlu_model

DATA_PATH = 'data/processed/nlu_dataset.json'
MODEL_EXPORT_PATH = 'models/exports/nlu_model.h5'
VOCAB_PATH = 'models/exports/vocab.json'
MAP_PATH = 'models/exports/intent_map.json'

os.makedirs('models/exports/', exist_ok=True)

print("Chargement des données...")
with open(DATA_PATH, 'r', encoding='utf-8') as f:
    data = json.load(f)

texts = [item['input'] for item in data]
intent_labels = [item['output'].split(':')[0].replace('[', '').replace('ACTION_', '') for item in data]

print("Prétraitement des textes...")
preprocessor = NLUPreprocessor(max_len=20)
intent_map = preprocessor.fit(texts, intent_labels)
X_train = preprocessor.transform(texts)
y_train = np.array([intent_map[label] for label in intent_labels])

print(f"Initialisation du modèle pour {len(intent_map)} intentions...")
vocab_size = len(preprocessor.tokenizer.word_index) + 1
model = build_nlu_model(vocab_size, len(intent_map), max_len=20)

print("Début de l'entraînement...")
model.fit(X_train, y_train, epochs=30, batch_size=16)

print("Sauvegarde des fichiers...")
model.save(MODEL_EXPORT_PATH)
preprocessor.save_assets(VOCAB_PATH, MAP_PATH)
