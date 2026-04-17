import json
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from architecture import build_nlu_model

with open('data/processed/nlu_dataset.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

texts = [item['input'] for item in data]
intent_labels = [item['output'].split(':')[0].replace('[', '') for item in data]
unique_intents = list(set(intent_labels))
intent_map = {name: i for i, name in enumerate(unique_intents)}
y_train = np.array([intent_map[label] for label in intent_labels])

tokenizer = Tokenizer(lower=True, char_level=False)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
max_len = 20
X_train = pad_sequences(sequences, maxlen=max_len, padding='post')

model = build_nlu_model(len(tokenizer.word_index) + 1, len(unique_intents), max_len)
model.fit(X_train, y_train, epochs=30, batch_size=16)

model.save('models/exports/nlu_model.h5')
with open('models/exports/vocab.json', 'w') as f:
    json.dump(tokenizer.word_index, f)