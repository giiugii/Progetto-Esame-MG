#importazione
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer

#caricamento del modello e del tokenizer
model_name="nlptown/bert-base-multilingual-uncased-sentiment"
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

sentiment_analyzer = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

def analyze_sentiment(text):
    sentiment = sentiment_analyzer(text)[0]
    sentiment_label = sentiment['label'] 
    sentiment_score = sentiment['score']  

    #traduzione dei risultati
    if sentiment_label in ['1 star', '2 stars']:
        sentiment_label = "Negativo"
    elif sentiment_label in ['3 stars']:
        sentiment_label = "Neutro"
    elif sentiment_label in ['4 stars', '5 stars']:
        sentiment_label = "Positivo"

    return sentiment_label, sentiment_score