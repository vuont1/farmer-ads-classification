import torch
import nltk
import tensorflow as tf
import gradio as gr
from transformers import TFBertModel, BertTokenizer
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
nltk.download("punkt")

# Die gleiche Funktion wie beim erstellen des Models wird hier für die rekoinstruktion und das laden verwendet
def build_bert_model(max_length):
    bert_model = TFBertModel.from_pretrained('bert-base-uncased')
    input_ids = Input(shape=(max_length,), dtype=tf.int32, name="input_ids")
    attention_mask = Input(shape=(max_length,), dtype=tf.int32, name="attention_mask")
    bert_outputs = bert_model({'input_ids': input_ids, 'attention_mask': attention_mask})
    pooled_output = bert_outputs.pooler_output
    x = Dropout(0.3)(pooled_output)
    x = Dense(256, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    x = Dense(128, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    x = Dense(64, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    output = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=[input_ids, attention_mask], outputs=output)
    bert_model.trainable = False
    model.compile(
        loss='binary_crossentropy',
        optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=5e-5),
        metrics=['accuracy']
    )
    return model

# Hyperparameter
MAX_LEN = 100

# Word2Vec
w2v_model = Word2Vec.load("models/word2vec.model")
EMBEDDING_DIM = w2v_model.vector_size

# Tokenizer rekonstruieren aus Word2Vec
tokenizer = Tokenizer()
tokenizer.word_index = {word: idx + 1 for idx, word in enumerate(w2v_model.wv.index_to_key)}

# BERT-Modell rekonstruieren und Gewichte laden
bert_model = build_bert_model(max_length=MAX_LEN)
bert_model.load_weights("models/bert_transfer_weights.h5")
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


# Modelle laden
models = {
    "Modell 1 (BERT)": {
        "model": bert_model,
    },
    "Modell 2 (CNN)": {
        "model": load_model("models/cnn_word2vec.keras")
    },
    "Modell 3 (LSTM)": {
        "model": load_model("models/lstm_word2vec.keras")
    }
}

# Preprocessing für CNN/LSTM
def preprocess_for_dl(text):
    tokens = word_tokenize(text.lower())
    sequence = tokenizer.texts_to_sequences([' '.join(tokens)])
    padded = pad_sequences(sequence, maxlen=MAX_LEN)
    return padded

# Preprocessing für BERT
def preprocess_for_bert(text):
    inputs = bert_tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=MAX_LEN,
        padding='max_length',
        truncation=True,
        return_tensors='tf'
    )
    return inputs['input_ids'], inputs['attention_mask']

def predict(text):
    results = {}

    for name, data in models.items():
        model = data["model"]

        if "BERT" in name:
            input_ids, attention_mask = preprocess_for_bert(text)
            pred_prob = model.predict({'input_ids': input_ids, 'attention_mask': attention_mask}, verbose=0)[0][0]
        else:
            processed_input = preprocess_for_dl(text)
            pred_prob = model.predict(processed_input, verbose=0)[0][0]

        pred = 1 if pred_prob > 0.5 else 0
        label = "✅ Akzeptiert" if pred == 1 else "❌ Abgelehnt"
        results[name] = f"{label} (Confidence: {pred_prob:.2f})"

    return results

# Gradio-Interface
iface = gr.Interface(
    fn=predict,
    inputs=gr.Textbox(lines=5, placeholder="Gib hier die Anzeige ein..."),
    outputs=gr.JSON(label="Ergebnisse"),
    title="Vergleich von Klassifikationsmodellen für Werbe-Anzeigen",
)

iface.launch(share=True)