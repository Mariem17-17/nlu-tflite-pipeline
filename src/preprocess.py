import json
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

class NLUPreprocessor:
    def __init__(self, max_len=20):
        self.tokenizer = Tokenizer(
            lower=True,
            char_level=False,
            oov_token="<OOV>"
        )
        self.max_len = max_len
        self.intent_map = {}

    def fit(self, texts, intent_labels):
        self.tokenizer.fit_on_texts(texts)
        unique_intents = sorted(set(intent_labels))
        self.intent_map = {label: i for i, label in enumerate(unique_intents)}
        return self.intent_map

    def transform(self, texts):
        sequences = self.tokenizer.texts_to_sequences(texts)
        return pad_sequences(
            sequences,
            maxlen=self.max_len,
            padding="post",
            truncating="post"
        )

    def save_assets(self, vocab_path, map_path):
        with open(vocab_path, "w", encoding="utf-8") as f:
            json.dump(self.tokenizer.word_index, f, ensure_ascii=False)

        with open(map_path, "w", encoding="utf-8") as f:
            json.dump(self.intent_map, f, ensure_ascii=False)

    def load_assets(self, vocab_path, map_path):
        with open(vocab_path, "r", encoding="utf-8") as f:
            word_index = json.load(f)

        with open(map_path, "r", encoding="utf-8") as f:
            self.intent_map = json.load(f)

        self.tokenizer.word_index = word_index
        self.tokenizer.index_word = {v: k for k, v in word_index.items()}
