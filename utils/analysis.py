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

import stanza
import pandas as pd

# Carica il dizionario EmoLex
def load_emolex(file_path):
    emolex_df = pd.read_csv(file_path, sep="\t", header=None)
    emolex_df.columns = ['Word', 'Parlante', 'Anticipation', 'Anger', 'Fear', 'Joy', 'Sadness', 'Surprise', 'Disgust', 'Trust', 'Negative', 'Positive']
    emolex = {}
    for index, row in emolex_df.iterrows():
        word = row['Word']  # Togliamo maiuscole per fare il match
        emotions = row[1:].values  # Prendiamo le emozioni associate
        emolex[word] = emotions  # Mappiamo parola -> emozioni
    return emolex

# Funzione di lemmatizzazione con Stanza (se non lo hai già integrato)
def lemmatize_text(text):
    nlp = stanza.Pipeline('it', processors='tokenize,lemma')
    doc = nlp(text)
    lemmi = [word.lemma for sentence in doc.sentences for word in sentence.words]
    return lemmi

# Funzione per analizzare le emozioni in un testo
def analizza_emozioni(text, emolex):
    lemmi = lemmatize_text(text)
    emotion_scores = {
        'Anticipation': 0,
        'Anger': 0,
        'Fear': 0,
        'Joy': 0,
        'Sadness': 0,
        'Surprise': 0,
        'Disgust': 0,
        'Trust': 0,
        'Negative': 0,
        'Positive': 0
    }
    
    for lemma in lemmi:
        lemma = lemma.lower()  # Convertiamo la parola in minuscolo per fare il match
        if lemma in emolex:
            emotions = emolex[lemma]  # Prendi il punteggio delle emozioni
            emotion_scores['Anticipation'] += float (emotions[0])
            emotion_scores['Anger'] += float (emotions[1])
            emotion_scores['Fear'] += float (emotions[2])
            emotion_scores['Joy'] += float (emotions[3])
            emotion_scores['Sadness'] += float (emotions[4])
            emotion_scores['Surprise'] += float (emotions[5])
            emotion_scores['Disgust'] += float (emotions[6])
            emotion_scores['Trust'] += float (emotions[7])
            emotion_scores['Negative'] += float (emotions[8])
            emotion_scores['Positive'] += float (emotions[9])
    
    return emotion_scores