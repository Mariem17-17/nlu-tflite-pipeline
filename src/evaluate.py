import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.models import load_model
from preprocess import NLUPreprocessor

model = load_model('models/exports/nlu_model.h5')
preprocessor = NLUPreprocessor(max_len=20)
preprocessor.load_assets('models/exports/vocab.json', 'models/exports/intent_map.json')

with open('data/processed/nlu_dataset.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

texts = [item['input'] for item in data[-200:]]
y_true = [
    preprocessor.intent_map[
        item['output'].split(':')[0].replace('[', '').replace('ACTION_', '')
    ] 
    for item in data[-200:]
]
X_test = preprocessor.transform(texts)
y_pred_probs = model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)

cm = confusion_matrix(y_true, y_pred)
intent_names = list(preprocessor.intent_map.keys())

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=intent_names, yticklabels=intent_names, cmap='Blues')
plt.xlabel('Prédictions de l\'IA')
plt.ylabel('Vérité (Réalité)')
plt.title('Matrice de Confusion - NLU Engine')
plt.savefig('models/exports/confusion_matrix.png')
plt.show()

print(classification_report(y_true, y_pred, target_names=intent_names))