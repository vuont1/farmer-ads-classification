import gradio as gr
import torch
from transformers import BertForSequenceClassification, BertTokenizer

# Modelle laden
models = {
    "Modell 3 (Fine-tuned BERT)": {
        "model": BertForSequenceClassification.from_pretrained("my-finetuned-bert"),
        "tokenizer": BertTokenizer.from_pretrained("my-finetuned-bert")
    },
    "Modell 2 (BERT cased)": {
        "model": BertForSequenceClassification.from_pretrained("bert-base-cased", num_labels=2),
        "tokenizer": BertTokenizer.from_pretrained("bert-base-cased")
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