import json

def load_dataset(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    texts = []
    intents = []
    slots = []

    for sample in data:
        texts.append(sample["input"])
        intents.append(sample["intent"])
        slots.append(sample.get("slots", {}))

    return texts, intents, slots