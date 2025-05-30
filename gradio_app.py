import gradio as gr
import torch
from transformers import BertForSequenceClassification, BertTokenizer
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
import nltk

# Setup
nltk.download("punkt")

# Hyperparameter
MAX_LEN = 100

# Word2Vec laden
w2v_model = Word2Vec.load("models/word2vec.model")
EMBEDDING_DIM = w2v_model.vector_size

# Tokenizer rekonstruieren aus Word2Vec
tokenizer = Tokenizer()
tokenizer.word_index = {word: idx + 1 for idx, word in enumerate(w2v_model.wv.index_to_key)}

# Modelle laden
models = {
    "Modell 1 (Fine-tuned BERT)": {
        "model": BertForSequenceClassification.from_pretrained("my-finetuned-bert"),
        "tokenizer": BertTokenizer.from_pretrained("my-finetuned-bert")
    },
    "Modell 2 (CNN)": {
        "model": load_model("models/cnn_word2vec.keras")
    },
    "Modell 3 (LSTM)": {
        "model": load_model("models/lstm_word2vec.keras")
    }
}

# Preprocessing-Funktion für CNN/LSTM
def preprocess_for_dl(text):
    tokens = word_tokenize(text.lower())
    sequence = tokenizer.texts_to_sequences([' '.join(tokens)])
    padded = pad_sequences(sequence, maxlen=MAX_LEN)
    return padded

# Hauptvorhersagefunktion
def predict(text):
    results = {}

    for name, data in models.items():
        if "tokenizer" in data and "model" in data:
            # BERT
            inputs = data["tokenizer"](text, return_tensors="pt", truncation=True, padding=True)
            with torch.no_grad():
                output = data["model"](**inputs)
            probs = torch.softmax(output.logits, dim=1)
            pred = torch.argmax(probs).item()
            conf = probs[0][pred].item()
            label = "✅ Akzeptiert" if pred == 1 else "❌ Abgelehnt"
            results[name] = f"{label} (Confidence: {conf:.2f})"

        elif "model" in data:
            # CNN oder LSTM
            processed_input = preprocess_for_dl(text)
            pred_prob = data["model"].predict(processed_input, verbose=0)[0][0]
            pred = 1 if pred_prob > 0.5 else 0
            label = "✅ Akzeptiert" if pred == 1 else "❌ Abgelehnt"
            results[name] = f"{label} (Confidence: {pred_prob:.2f})"

        else:
            results[name] = "Error"

    return results

# Gradio-Interface
iface = gr.Interface(
    fn=predict,
    inputs=gr.Textbox(lines=5, placeholder="Gib hier die Anzeige ein..."),
    outputs=gr.JSON(label="Ergebnisse"),
    title="Vergleich von Klassifikationsmodellen für Werbe-Anzeigen",
)

iface.launch(share=True)
