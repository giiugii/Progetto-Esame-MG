import gradio as gr
from utils.analysis import analyze_sentiment
from utils.video_processor import diarize_and_transcribe_audio

def process_video(audio_file):
    try:
        subtitles= diarize_and_transcribe_audio(audio_file)
    except Exception as e:
        print(f"Errore nella diarizzazione o trascrizione: {e}")
        return "Errore nella diarizzazione o trascrizione", None

    # Esegui l'analisi sentimentale per ogni segmento
    sentiment_results = []
    for segment in subtitles:
        speaker, text = segment[1], segment[2]
        try:
            # Analizza il sentiment del testo per ogni speaker
            sentiment_label, sentiment_score = analyze_sentiment(text)
            sentiment_results.append((speaker, text, sentiment_label, sentiment_score))
        except Exception as e:
            print(f"Errore nell'analisi sentimentale per il parlante {speaker}: {e}")
            sentiment_results.append((speaker, text, "Errore", 0))

    # Organizza i risultati di trascrizione
    transcript_text = ""
    for speaker, text, _, _ in sentiment_results:
        transcript_text += f"{speaker}: {text}\n\n"

    # Organizza i risultati di analisi sentimentale
    sentiment_text = ""
    for speaker, text, sentiment_label, sentiment_score in sentiment_results:
        sentiment_text += f"{speaker}: Sentiment: {sentiment_label} (Punteggio: {sentiment_score:.2f})\n\n"

    return transcript_text, sentiment_text

def create_gradio_interface():
    iface = gr.Interface(
        fn=process_video,
        inputs=gr.Video(label="Carica Video"), 
        outputs=[gr.Textbox(label="Trascrizione"),gr.Textbox(label="Analisi Sentimentale")],
        title="Analisi audio di un video",
        description="Carica un video per generare la trascrizione dell'audio e per avere l'analisi sentimentale.",
        theme="default",
        live=False
    )
    return iface

