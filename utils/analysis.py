# utils/analysis.py
from pyannote.audio import Pipeline
from transformers import pipeline
import os

# Sostituisci 'YOUR_AUTH_TOKEN' con il token ottenuto da Hugging Face
diarization_pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization", use_auth_token="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")

sentiment_pipeline = pipeline("sentiment-analysis")

def sentiment_analysis(text):
    result = sentiment_pipeline(text[:512])  # Limita la lunghezza del testo per evitare errori
    return result[0]['label'], result[0]['score']

def diarize(audio_path):
    try:
        diarization = diarization_pipeline(audio_path)
        diarize_data = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            diarize_data.append(f"Speaker {speaker}: {turn.start:.2f} - {turn.end:.2f}")
        return "\n".join(diarize_data)
    except Exception as e:
        print(f"Errore durante la diarizzazione: {e}")
        return "Errore durante la diarizzazione"