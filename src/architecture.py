import tensorflow as tf
from tensorflow.keras import layers, models

def build_nlu_model(vocab_size, num_intents, max_len):
    input_text = layers.Input(shape=(max_len,), name="input_text")

    embedding = layers.Embedding(
        input_dim=vocab_size,
        output_dim=64
    )(input_text)

    lstm = layers.Bidirectional(
        layers.LSTM(64, return_sequences=False)
    )(embedding)

    intent_dropout = layers.Dropout(0.3)(lstm)
    intent_output = layers.Dense(
        num_intents,
        activation="softmax",
        name="intent_output"
    )(intent_dropout)

    model = models.Model(inputs=input_text, outputs=intent_output)

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model