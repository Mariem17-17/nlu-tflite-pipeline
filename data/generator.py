import json
import random

N_SAMPLES = 10000

NAMES = ["Ahmed", "Samira", "Walid", "Amine", "Sarah", "Leila", "Mom", "Dad", "Souhail"]
NUMBERS = ["55123456", "22888999", "98765432", "20111222"]
MESSAGES = [
    "i will be late by ten minutes",
    "call me back soon",
    "happy birthday",
    "meeting confirmed"
]
QUERIES = ["llama three explained", "coding in kotlin", "best couscous recipe"]

DIGITS = {
    "0": "zero", "1": "one", "2": "two", "3": "three",
    "4": "four", "5": "five", "6": "six",
    "7": "seven", "8": "eight", "9": "nine"
}

def speak_number(num):
    return " ".join(DIGITS[d] for d in num)


def speak_time(h, m):
    parts = []

    if h >= 10:
        parts.append(DIGITS[str(h // 10)])
    parts.append(DIGITS[str(h % 10)])

    parts.append(DIGITS[str(m // 10)])
    parts.append(DIGITS[str(m % 10)])

    return " ".join(parts)


DATASET = []

TEMPLATES = {
    "CALL": [
        lambda n, num: (f"call {n}", {"contact": n}),
        lambda n, num: (f"dial {speak_number(num)}", {"number": num})
    ],
    "SMS": [
        lambda n, msg: (f"send a message to {n}", {"contact": n}),
        lambda n, msg: (f"text {n}", {"contact": n})
    ],
    "ALARM": [
        lambda h, m: (f"set alarm for {h}:{m:02d}", {"time": f"{h:02d}:{m:02d}"}),
        lambda h, m: (f"wake me up at {speak_time(h, m)}", {"time": f"{h:02d}:{m:02d}"})
    ],
    "CONTACT": [
        lambda n, num: (
            f"add contact {n} with number {speak_number(num)}",
            {"contact": n, "number": num}
        )
    ],
    "YOUTUBE": [
        lambda q: (f"search {q} on youtube", {"query": q}),
        lambda q: (f"play {q} on youtube", {"query": q})
    ]
}

for _ in range(N_SAMPLES):
    intent = random.choice(list(TEMPLATES.keys()))

    name = random.choice(NAMES)
    number = random.choice(NUMBERS)
    msg = random.choice(MESSAGES)
    query = random.choice(QUERIES)
    hour = random.randint(0, 23)
    minute = random.choice([0, 15, 30, 45])

    if intent == "CALL":
        text, slots = random.choice(TEMPLATES["CALL"])(name, number)
    elif intent == "SMS":
        text, slots = random.choice(TEMPLATES["SMS"])(name, msg)
    elif intent == "ALARM":
        text, slots = random.choice(TEMPLATES["ALARM"])(hour, minute)
    elif intent == "CONTACT":
        text, slots = random.choice(TEMPLATES["CONTACT"])(name, number)
    elif intent == "YOUTUBE":
        text, slots = random.choice(TEMPLATES["YOUTUBE"])(query)

    DATASET.append({
        "lang": "en",
        "intent": intent,
        "input": text.lower(),
        "slots": slots
    })

with open("nlu_en_dataset.json", "w", encoding="utf-8") as f:
    json.dump(DATASET, f, indent=2)

print(f"Generated {len(DATASET)} English NLU samples")