from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer

# Carica il modello e il tokenizer italiano
model_name = "dbmdz/bert-base-italian-xxl-cased"
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Crea il pipeline per l'analisi sentimentale
sentiment_analyzer = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

def analyze_sentiment(text):
    # Usa il modello pre-addestrato per l'italiano
    sentiment = sentiment_analyzer(text)[0]
    sentiment_label = sentiment['label']  # 'POSITIVE' o 'NEGATIVE'
    sentiment_score = sentiment['score']  # Punteggio di fiducia nel risultato

    return sentiment_label, sentiment_score