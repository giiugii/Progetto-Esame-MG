import gradio as gr
from utils.analysis import analyze_sentiment
from utils.video_processor import diarize_and_transcribe_audio
import pandas as pd
import plotly.express as px
import plotly.io as pio
import tempfile

def process_video(audio_file):
    """
    Processa un file video per estrarre e analizzare il parlato.
    
    Args:
    audio_file (str): Percorso del file video da analizzare.
    
    Returns:
    tuple: Una tupla contenente:
        - transcript_text (str): La trascrizione completa del video.
        - df (pd.DataFrame): DataFrame con l'analisi sentimentale per ogni segmento.
        - plot_path (str): Percorso del file immagine del grafico.
        - sentiment_totals (dict): Dizionario con punteggi totali di sentiment per speaker.
    """
    try:
        subtitles = diarize_and_transcribe_audio(audio_file)
    except Exception as e:
        print(f"Errore nella diarizzazione o trascrizione: {e}")
        return "Errore nella diarizzazione o trascrizione", None, None, {}

    sentiment_results = []

    for segment in subtitles:
        speaker, text = segment[1], segment[2]
        try:
            sentiment_label, sentiment_score = analyze_sentiment(text)
            sentiment_results.append({
                'Turno': f"Turno {len(sentiment_results) + 1}",
                'Speaker': speaker, 
                'Frase': text, 
                'Sentiment': sentiment_label, 
                'Score': sentiment_score
            })
        except Exception as e:
            print(f"Errore nell'analisi sentimentale per il parlante {speaker}: {e}")
            sentiment_results.append({
                'Turno': f"Turno {len(sentiment_results) + 1}",
                'Speaker': speaker, 
                'Frase': text, 
                'Sentiment': "Errore", 
                'Score': 0
            })

    # Converti i risultati in un DataFrame
    df = pd.DataFrame(sentiment_results)
    
    # Crea una versione testuale della trascrizione
    transcript_text = "\n\n".join([f"{row['Speaker']}: {row['Frase']}" for index, row in df.iterrows()])

    # Creazione del grafico dettagliato
    fig = px.scatter(df, 
                     x='Turno', 
                     y='Score', 
                     color='Speaker', 
                     size='Score', 
                     hover_data=['Frase'], 
                     title='Analisi Sentimentale per Turno e Speaker',
                     labels={'Score': 'Punteggio Sentimentale', 'Turno': 'Turno di Parlato'})

    # Salva il grafico come immagine temporanea
    with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmpfile:
        pio.write_image(fig, tmpfile.name)
        plot_path = tmpfile.name

    # Calcolo del sentiment totale per ciascun speaker
    sentiment_totals = []
    for speaker in df['Speaker'].unique():
        speaker_data = df[df['Speaker'] == speaker]
        total_score = speaker_data['Score'].sum()  # Somma dei punteggi
        avg_score = speaker_data['Score'].mean()   # Media dei punteggi
        num_phrases = len(speaker_data)            # Numero di frasi per speaker
        most_common_sentiment = speaker_data['Sentiment'].mode()[0]  # Sentiment più comune
        sentiment_totals.append({
            'Speaker': speaker,
            'Sentiment più comune': most_common_sentiment,
            'Numero di frasi': num_phrases,
            'Somma dei punteggi': f"{total_score:.2f}",
            'Media dei punteggi': f"{avg_score:.2f}"
        })

    return transcript_text, df, plot_path, pd.DataFrame(sentiment_totals)

def create_gradio_interface():
    """
    Crea e restituisce l'interfaccia Gradio per la nostra applicazione.

    Returns:
    gr.Blocks: L'interfaccia Gradio configurata.
    """
    iface = gr.Interface(
        fn=process_video,
        inputs=gr.Video(label="Carica Video"), 
        outputs=[
            gr.Textbox(label="Trascrizione"),
            gr.Dataframe(label="Tabella Sentimentale", datatype=['str', 'str', 'str', 'number']),
            gr.Image(label="Grafico Sentimentale"),
            gr.Dataframe(label="Sentimental Analysis Totale")
        ],
        title="Analisi audio di un video",
        description="Carica un video per generare la trascrizione dell'audio, una tabella con l'analisi sentimentale, un grafico dettagliato e il riassunto totale del sentiment per ciascun parlante.",
        theme="default",
        live=False
    )

    with gr.Blocks() as iface:
        gr.Markdown("# Analisi audio di un video")
        gr.Markdown("Carica un video per generare la trascrizione dell'audio, una tabella con l'analisi sentimentale, un grafico dettagliato e il riassunto totale del sentiment per ciascun parlante.")

        video_input = gr.Video(label="Carica Video")

        # Trascrizione
        transcript_output = gr.Textbox(label="Trascrizione", lines=5)

        # Tabella Sentimentale che occupa entrambe le colonne
        with gr.Row():
            table_output = gr.Dataframe(label="Tabella Sentimentale", datatype=['str', 'str', 'str', 'number'])

        # Grafico e Analisi Totale
        with gr.Row():
            with gr.Column(scale=3):
                plot_output = gr.Image(label="Grafico Sentimentale")
            with gr.Column(scale=1):
                total_sentiment_output = gr.Text(label="Sentimental Analysis Totale", lines=10)

        # Collegamento degli input agli output
        video_input.upload(process_video, inputs=[video_input], outputs=[transcript_output, table_output, plot_output, total_sentiment_output])

    return iface