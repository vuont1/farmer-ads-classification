# Farm-Ads Textklassifikation mit Deep Learning

## 1. Überblick
In diesem Projekt wird eine Textklassifikation für landwirtschaftliche Inserate („Farm-Ads“) 
mithilfe verschiedener Deep-Learning-Modelle umgesetzt. 
Ziel ist es, den Textinhalt automatisch einer bestimmten Klasse zuzuordnen 
(z. B. akzeptierte oder abgelehnte Werbung), um die automatische Kategorisierung zu ermöglichen.

### Datenbeschreibung

Die Daten stammen aus Textanzeigen von zwölf Webseiten mit Fokus auf 
landwirtschaftliche Themen und Tierhaltung. Pro Anzeige sind jeweils zwei Textquellen vorhanden:

- **Ad Creative** (Werbetext), mit dem Präfix `ad-` versehen
- **Landing Page** (Zielseite), inkl. `title`- und `header`-Markups aus dem HTML

Die Daten wurden bereits **gestemmt** und von **Stoppwörtern** bereinigt. Jede Anzeige steht in einer Zeile, wobei:

- das **erste Element das Label** ist:
  - `1` = akzeptierte Anzeige
  - `-1` = abgelehnte Anzeige

- der Rest die Textmerkmale enthält

Zusätzlich ist eine **Bag-of-Words-Repräsentation** im **SVMlight-Format** enthalten:
```
<label> index1:value1 index2:value2 ...
```
Diese Darstellung wurde in der zugehörigen Publikation verwendet.

### Verzeichnisstruktur
```
project/
├── farm-ads                                    # Textdaten
├── farm-ads-vect                               # Vektordaten
├── models/                                     # Gespeicherte Modelle
├── visualization/                              # Plots und Grafiken
└── gradio_app.py                               # GUI für Klassifizierung der Modelle
└── cnn_lstm_bert_model_analysis_training.ipynb # Training Modelle und Erstellung der Visualisierungen
```

## 2. Anforderungen (requirements.txt)
Wir haben mit folgenden Versionen gearbeitet:

- Python: 3.11  
- TensorFlow: 2.15.0  
- Keras: 3.1.0  
- NumPy: 1.26.4  
- Pandas: 2.2.2 

Diese Versionen wurden verwendet, um sicherzustellen, 
dass die Notebooks reproduzierbar ausgeführt werden können. 
Es kann bei anderen Versionen zu Inkompatibilitäten kommen.

## 3. Erklärung der Pakete und Methoden

### Datenverarbeitung
- **NumPy:** Numerische Operationen und Array-Handling
- **Pandas:** Strukturierte Datenmanipulation
- **Matplotlib/Seaborn:** Datenvisualisierung und Modellbewertung

### Machine Learning
- **Scikit-learn:** Datenaufteilung und Evaluierungsmetriken
- **TensorFlow/Keras:** Deep Learning Framework

### NLP (Natural Language Processing)
- **NLTK:** Tokenization und Textvorverarbeitung
- **Gensim:** Word2Vec Embeddings
- **Transformers:** BERT-Modell für Transfer Learning

## 4. Modellarchitekturen und deren Begründung
### 1. CNN Model (14 Layers)
```
Embedding → SpatialDropout → Conv1D Blocks → GlobalMaxPooling → Dense
```
- **Zweck:** Erkennt lokale Textmuster (n-grams)
- **Stärken:** Schnell, effizient für kurze Texte

### 2. LSTM Model (11 Layers)
```
Embedding → SpatialDropout → LSTM Blocks → Dense
```
- **Zweck:** Erfasst sequenzielle Abhängigkeiten
- **Stärken:** Versteht längere Kontexte und Wortfolgen

### 3. BERT Transfer Learning
```
Frozen BERT → Custom Dense Layers → Classification
```
- **Zweck:** Nutzt vortrainiertes Sprachwissen
- **Stärken:** State-of-the-Art Textverständnis

## 5. Verwendung
### Daten laden und vorverarbeiten
```python
texts, labels, vector_df = load_farm_ads_data("farm-ads", "farm-ads-vect")
cleaned_texts, tokenized_texts = preprocess_text(texts)
```

### Modelle trainieren
```python
# CNN Training
cnn_model = build_cnn_model(vocab_size, embedding_dim, embedding_matrix)
cnn_results = train_evaluate_model(cnn_model, X_train_pad, y_train, X_test_pad, y_test)

# LSTM Training  
lstm_model = build_lstm_model(vocab_size, embedding_dim, embedding_matrix)
lstm_results = train_evaluate_model(lstm_model, X_train_pad, y_train, X_test_pad, y_test)

# BERT Training
bert_model = build_bert_model(max_length=128)
bert_results = train_evaluate_model(bert_model, bert_inputs, y_train, bert_test_inputs, y_test)
```

## 6. Evaluierung

### Metriken
- **Accuracy:** Klassifikationsgenauigkeit
- **ROC-AUC:** Threshold-unabhängige Bewertung
- **Confusion Matrix:** Detaillierte Fehleranalyse
- **Classification Report:** Precision, Recall, F1-Score

### Visualisierungen
- Datenverteilung und Worthäufigkeiten
- Trainingsverläufe (Accuracy/Loss)
- ROC-Kurven für alle Modelle
- Model Performance Comparison

## 7. Features

### Datenvorverarbeitung
- Automatische Tokenization mit NLTK
- Sequence Padding für einheitliche Längen
- Label Encoding (-1/1 → 0/1)
- Stratified Train/Test Split (80% training und 20% test + Klassenverteilung gleich)

### Model Training
- Early Stopping gegen Overfitting
- Batch Normalization für stabile Trainings
- Dropout Regularization
- Adam Optimizer mit adaptiver Lernrate

### Transfer Learning
- Vortrainierte Word2Vec Embeddings
- Frozen BERT Base Model
- Custom Classification Heads

## 8. Output

### Gespeicherte Modelle
```
models/
├── word2vec.model           # Word2Vec Embeddings
├── embedding_matrix.npy     # Embedding Matrix
├── cnn_word2vec.keras      # CNN Model
├── lstm_word2vec.keras     # LSTM Model
└── bert_transfer.keras     # BERT Model
```

### Visualisierungen
```
visualization/
├── distributions.png          # Klassenverteilung
├── word_frequency.png        # Worthäufigkeiten
├── model_comparison.png      # Performance Vergleich
├── *_training_history.png    # Trainingsverläufe
├── *_confusion_matrix.png    # Confusion Matrices
└── *_roc_curve.png          # ROC Kurven
```

## 9. Konfiguration

### Hyperparameter
- **CNN:** 15 Epochen, Batch Size 32
- **LSTM:** 15 Epochen, Batch Size 32  
- **BERT:** 5 Epochen, Batch Size 16
- **Word2Vec:** 100 Dimensionen, Window 5
- **Max Sequence Length:** 100 (CNN/LSTM), 128 (BERT)

## 10. Projektziele

1. **Vergleich verschiedener NLP-Architekturen**
2. **Optimale Klassifikation von Farm-Anzeigen**
3. **Reproduzierbare ML-Pipeline**
4. **Umfassende Modellbewertung**

## 11. Ergebnisse

Das Projekt liefert einen detaillierten Vergleich der drei Modellarchitekturen mit:
- Accuracy- und AUC-Scores
- Detaillierte Classification Reports
- Visuelle Performance-Analyse
- Gespeicherte Modelle für Produktionseinsatz

## 12. Warum diese spezifische Kombination?

Wir haben uns für CNN, LSTM und BERT entschieden, 
um einen Ensemble-Ansatz zu verfolgen und unterschiedliche 
Aspekte der "farm-ads"-Textdaten zu erfassen. 
Jedes Modell wird individuell auf seine Performance getestet, 
obwohl uns klar ist, dass nicht alle gleichermassen geeignet sind.

**Ensemble-Ansatz:**
- **CNN (Convolutional Neural Network): Sehr passend**: da es lokale Muster und wiederkehrende Schlagwörter (wie "ad-jerry", "title-understand") in den Daten hervorragend erkennt.
- **LSTM (Long Short-Term Memory Network): Passend**: da es sequenzielle Abhängigkeiten in den längeren Wortabfolgen und Tag-Strukturen ("ad-rheumatoid ad-arthritis ad-expert") des Datensatzes modellieren kann.
- **BERT (Bidirectional Encoder Representations from Transformers): Weniger passend**: BERT ist für natürliche Sprache optimiert; die stark tokenisierten "farm-ads"-Daten ohne Satzstruktur könnten seine Fähigkeit zum kontextuellen Verständnis einschränken und zu ungenutzter Komplexität oder Overfitting führen.

**Weitere Vorteile**
**Vergleichbarkeit:**
- Gleiche Datenbasis & Evaluierung
- Objektiver Modellvergleich

**Produktionsreife:**
- Visualisierung & Fehlerbehandlung
- Speichermöglichkeit & Re-Usability

