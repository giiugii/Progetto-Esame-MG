from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer

# Carica il modello e il tokenizer italiano
#model_name = "dbmdz/bert-base-italian-xxl-cased"
model_name="nlptown/bert-base-multilingual-uncased-sentiment"
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Crea il pipeline per l'analisi sentimentale
sentiment_analyzer = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

def analyze_sentiment(text):
    # Usa il modello pre-addestrato per l'italiano
    sentiment = sentiment_analyzer(text)[0]
    sentiment_label = sentiment['label']  # 'POSITIVE' o 'NEGATIVE'
    sentiment_score = sentiment['score']  # Punteggio di fiducia nel risultato

    # Tradurre il numero di stelle in etichetta (positivo, neutro, negativo)
    if sentiment_label in ['1 star', '2 stars']:
        sentiment_label = "Negativo"
    elif sentiment_label in ['3 stars']:
        sentiment_label = "Neutro"
    elif sentiment_label in ['4 stars', '5 stars']:
        sentiment_label = "Positivo"

    return sentiment_label, sentiment_score