import json
import random

names = ["Ahmed", "Samira", "Walid", "Amine", "Sarah", "Mom", "Dad", "Leila", "Boss"]
numbers = ["55123456", "22888999", "98765432", "20111222", "50444333"]
messages = ["I am outside", "Call me back soon", "Happy Birthday!", "I will be late by 10 minutes", "Meeting confirmed"]
times = ["08:00", "12:30", "20:00", "07:15", "22:45"]
queries = ["coding in kotlin", "ISI Kef", "best couscous recipe", "Llama 3 explained"]

templates = [
    {"intent": "CALL", "utterances": ["Call {name}", "Dial {number}", "Phone {name}", "Appelle {name}"]},
    {"intent": "SMS", "utterances": ["Send a text to {name} saying {msg}", "Tell {name} that {msg}", "SMS {number} {msg}", "Envoie un message à {name} pour dire {msg}"]},
    {"intent": "ALARM", "utterances": ["Set alarm for {time}", "Wake me up at {time}", "Mets une alarme à {time}", "Réveille-moi à {time}"]},
    {"intent": "YOUTUBE", "utterances": ["Search {query} on YouTube", "Play {query} on YT", "Cherche {query} sur Youtube"]},
    {"intent": "CONTACT", "utterances": ["Add {name} with number {number}", "Save {name} as {number}", "Crée le contact {name} numéro {number}"]},
    {"intent": "NOTE", "utterances": ["Take a note: {msg}", "Remember that {msg}", "Note que {msg}"]}
]

dataset = []

for _ in range(3000):
    tpl = random.choice(templates)
    utterance = random.choice(tpl['utterances'])
    
    val_name = random.choice(names)
    val_num = random.choice(numbers)
    val_msg = random.choice(messages)
    val_time = random.choice(times)
    val_query = random.choice(queries)
    
    formatted_text = utterance.format(name=val_name, number=val_num, msg=val_msg, time=val_time, query=val_query)
    
    if tpl['intent'] == "CALL": target = f"[ACTION_CALL:{val_name if 'name' in utterance else val_num}]"
    elif tpl['intent'] == "SMS": target = f"[ACTION_SMS:{val_name if 'name' in utterance else val_num}:{val_msg}]"
    elif tpl['intent'] == "ALARM": target = f"[ACTION_ALARM:{val_time.split(':')[0]}:{val_time.split(':')[1]}]"
    elif tpl['intent'] == "YOUTUBE": target = f"[ACTION_YOUTUBE:{val_query}]"
    elif tpl['intent'] == "CONTACT": target = f"[ACTION_ADD_CONTACT:{val_name}:{val_num}]"
    elif tpl['intent'] == "NOTE": target = f"[ACTION_NOTE:{val_msg}]"
    
    dataset.append({"input": formatted_text, "output": target})

with open('nlu_dataset.json', 'w', encoding='utf-8') as f:
    json.dump(dataset, f, indent=4, ensure_ascii=False)

print(f"Dataset de {len(dataset)} exemples généré avec succès !")