import gradio as gr
from utils.analysis import analyze_sentiment
from utils.video_processor import diarize_and_transcribe_audio
import pandas as pd
import plotly.express as px
import plotly.io as pio
import tempfile

def process_video(audio_file):
    try:
        subtitles = diarize_and_transcribe_audio(audio_file)
    except Exception as e:
        print(f"Errore nella diarizzazione o trascrizione: {e}")
        return "Errore nella diarizzazione o trascrizione", None, None, None

    sentiment_results = []
    for segment in subtitles:
        speaker, text = segment[1], segment[2]
        try:
            sentiment_label, sentiment_score = analyze_sentiment(text)
            sentiment_results.append({
                'Speaker': speaker, 
                'Text': text, 
                'Sentiment': sentiment_label, 
                'Score': sentiment_score
            })
        except Exception as e:
            print(f"Errore nell'analisi sentimentale per il parlante {speaker}: {e}")
            sentiment_results.append({
                'Speaker': speaker, 
                'Text': text, 
                'Sentiment': "Errore", 
                'Score': 0
            })
    
    # Creazione di un DataFrame per facilitare la visualizzazione
    df = pd.DataFrame(sentiment_results)
    
    # Creazione del grafico
    fig = px.bar(df, x='Speaker', y='Score', color='Sentiment', barmode='group', title="Analisi Sentimentale per Speaker")
    
    # Salva il grafico come immagine temporanea
    with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmpfile:
        pio.write_image(fig, tmpfile.name)
        plot_path = tmpfile.name

    # Organizza i risultati di trascrizione
    transcript_text = "\n\n".join([f"{row['Speaker']}: {row['Text']}" for index, row in df.iterrows()])

    # Organizza i risultati di analisi sentimentale
    sentiment_text = "\n\n".join([f"{row['Speaker']}: Sentiment: {row['Sentiment']} (Punteggio: {row['Score']:.2f})" for index, row in df.iterrows()])

    # Nota: La creazione di sentiment_counts e avg_scores non è utilizzata nel return, ma potrebbe essere utile per altre analisi
    # sentiment_counts = df['Sentiment'].value_counts().to_dict()
    # avg_scores = df.groupby('Speaker')['Score'].mean().to_dict()

    return transcript_text, sentiment_text, df, plot_path

def create_gradio_interface():
    iface = gr.Interface(
        fn=process_video,
        inputs=gr.Video(label="Carica Video"), 
        outputs=[
            gr.Textbox(label="Trascrizione"),
            gr.Textbox(label="Analisi Sentimentale"),
            gr.Dataframe(label="Tabella Sentimentale"),
            gr.Image(label="Grafico Sentimentale")
        ],
        title="Analisi audio di un video",
        description="Carica un video per generare la trascrizione dell'audio, l'analisi sentimentale, una tabella riassuntiva e un grafico.",
        theme="default",
        live=False
    )
    return iface