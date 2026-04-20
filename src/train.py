import json
import numpy as np
import os
from preprocess import NLUPreprocessor
from architecture import build_nlu_model

DATA_PATH = "data/nlu_en_dataset.json"
MODEL_EXPORT_PATH = "models/exports/nlu_model.h5"
VOCAB_PATH = "models/exports/vocab.json"
MAP_PATH = "models/exports/intent_map.json"

os.makedirs("models/exports", exist_ok=True)

print("Loading dataset...")
with open(DATA_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

texts = [item["input"] for item in data]
intent_labels = [item["intent"] for item in data]

print("Preprocessing texts...")
preprocessor = NLUPreprocessor(max_len=20)
intent_map = preprocessor.fit(texts, intent_labels)

X_train = preprocessor.transform(texts)
y_train = np.array([intent_map[label] for label in intent_labels])

print(f"Initializing model for {len(intent_map)} intents...")
vocab_size = len(preprocessor.tokenizer.word_index) + 1

model = build_nlu_model(
    vocab_size=vocab_size,
    num_intents=len(intent_map),
    max_len=20
)

print("Training...")
model.fit(X_train, y_train, epochs=30, batch_size=16)

print("Saving model and assets...")
model.save(MODEL_EXPORT_PATH)
preprocessor.save_assets(VOCAB_PATH, MAP_PATH)

print("Training complete.")
