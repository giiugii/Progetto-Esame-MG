#importazione
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
import stanza

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

#funzione per avere i lemmi delle parole del testo
def lemmatize_text(text):
    nlp = stanza.Pipeline('it', processors='tokenize,lemma')
    doc = nlp(text)
    lemmi = [word.lemma for sentence in doc.sentences for word in sentence.words]
    return lemmi

#funzione per fare l'analisi delle emozioni con EmoLex
def load_emolex (file_path):
    with open(file_path) as file:
        lines= file.readlines()
    header = lines[0].strip().split("\t")[1:-1] 
    emolex = {}
    for line in lines[1:]: 
        values= line.strip().split("\t")
        word= values[-1]
        emotion_scores= values[1:-1]
        emotion_scores = [float(score) for score in emotion_scores]
        emolex[word] = dict(zip(header, emotion_scores))
    return emolex

def analyze_emotions_average(text, emolex):
    lemmi = lemmatize_text(text)
    emotion_scores = {emotion: 0 for emotion in emolex[list(emolex.keys())[0]].keys()} 
    for lemma in lemmi:
        lemma=lemma.lower() 
        if lemma in emolex:
            emotions = emolex[lemma]
            for emotion, score in emotions.items():
                emotion_scores[emotion] += score
    num_words = len(lemmi) 
    if num_words > 0:
        for emotion in emotion_scores:
            emotion_scores[emotion] /= num_words 
    return emotion_scores