#from pyannote.audio import Pipeline
from transformers import pipeline

sentiment_analyzer = pipeline("sentiment-analysis")

def analyze_sentiment(text):
    # Usa il modello pre-addestrato di Hugging Face
    sentiment = sentiment_analyzer(text)[0]  # restituisce un dizionario con etichetta e punteggio
    
    sentiment_label = sentiment['label']  # 'POSITIVE' o 'NEGATIVE'
    sentiment_score = sentiment['score']  # Punteggio di fiducia nel risultato

    # Mappa le etichette
    #if sentiment_label == 'LABEL_0':
        #sentiment_label = 'Negativo'
    #elif sentiment_label == 'LABEL_1':
        #sentiment_label = 'Positivo'
    
    return sentiment_label, sentiment_score

# Sostituisci 'YOUR_AUTH_TOKEN' con il token ottenuto da Hugging Face
#diarization_pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization", use_auth_token="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")

#def diarize(audio_path):
    #try:
        #diarization = diarization_pipeline(audio_path)
        #diarize_data = []
        #for turn, _, speaker in diarization.itertracks(yield_label=True):
            #diarize_data.append(f"Speaker {speaker}: {turn.start:.2f} - {turn.end:.2f}")
        #return "\n".join(diarize_data)
    #except Exception as e:
        #print(f"Errore durante la diarizzazione: {e}")
        #return "Errore durante la diarizzazione"