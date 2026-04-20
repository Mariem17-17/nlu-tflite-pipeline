import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.models import load_model
from preprocess import NLUPreprocessor

MODEL_PATH = "models/exports/nlu_model.h5"
DATA_PATH = "data/nlu_en_dataset.json"
VOCAB_PATH = "models/exports/vocab.json"
MAP_PATH = "models/exports/intent_map.json"

model = load_model(MODEL_PATH)

preprocessor = NLUPreprocessor(max_len=20)
preprocessor.load_assets(VOCAB_PATH, MAP_PATH)

with open(DATA_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

samples = data[-200:]

texts = [item["input"] for item in samples]
y_true = [preprocessor.intent_map[item["intent"]] for item in samples]

X_test = preprocessor.transform(texts)
y_pred_probs = model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)

cm = confusion_matrix(y_true, y_pred)
intent_names = list(preprocessor.intent_map.keys())

plt.figure(figsize=(10, 8))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    xticklabels=intent_names,
    yticklabels=intent_names,
    cmap="Blues"
)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("NLU Confusion Matrix")
plt.tight_layout()
plt.savefig("models/exports/confusion_matrix.png")
plt.show()

print(classification_report(y_true, y_pred, target_names=intent_names))