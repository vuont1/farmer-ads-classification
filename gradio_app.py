import gradio as gr
import torch
from transformers import BertForSequenceClassification, BertTokenizer
from tensorflow.keras.models import load_model
from gensim.models import Word2Vec

# Modelle laden
models = {
    "Modell 3 (Fine-tuned BERT)": {
        "model": BertForSequenceClassification.from_pretrained("my-finetuned-bert"),
        "tokenizer": BertTokenizer.from_pretrained("my-finetuned-bert")
    },
    "Modell 2 (Word2Vec)": {
        "w2v_model": Word2Vec.load("models/word2vec.model")
    },
    "Modell 3 (CNN)": {
        "cnn_model": load_model("models/cnn_word2vec.h5")
    },
    "Modell 4 (LSTM)": {
        "lstm_model": load_model("models/lstm_word2vec.h5")
    }
}

def predict(text):
    results = {}
    for name, data in models.items():
        inputs = data["tokenizer"](text, return_tensors="pt", truncation=True)
        with torch.no_grad():
            output = data["model"](**inputs)
        pred = torch.argmax(output.logits).item()
        results[name] = "✅ Akzeptiert (1)" if pred == 1 else "❌ Abgelehnt (0)"
    return results

iface = gr.Interface(
    fn=predict,
    inputs=gr.Textbox(lines=5, placeholder="Enter AD to check"),
    outputs=gr.JSON(label="Results"),
    title="Modell-Vergleich für Textklassifikation"
)

iface.launch()