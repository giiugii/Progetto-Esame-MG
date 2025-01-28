from transformers import pipeline

sentiment_analyzer = pipeline("sentiment-analysis")

def analyze_sentiment(text):
    # Usa il modello pre-addestrato di Hugging Face
    sentiment = sentiment_analyzer(text)[0]  # restituisce un dizionario con etichetta e punteggio
    sentiment_label = sentiment['label']  # 'POSITIVE' o 'NEGATIVE'
    sentiment_score = sentiment['score']  # Punteggio di fiducia nel risultato

    return sentiment_label, sentiment_score