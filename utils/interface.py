import gradio as gr
from utils.analysis import analyze_sentiment
from utils.video_processor import diarize_and_transcribe_audio
import pandas as pd
import plotly.express as px
import plotly.io as pio
import tempfile

def process_video(audio_file): #ritorna tuple
    #trascrizione e diarizzazione
    subtitles = diarize_and_transcribe_audio(audio_file)

    #tabella analisi sentimentale turno*turno
    sentiment_results = []
    for segment in subtitles:
        speaker, text = segment[1], segment[2]
        sentiment_label, sentiment_score = analyze_sentiment(text)
        sentiment_results.append({
            'Turno': f"Turno {len(sentiment_results) + 1}",
            'Parlante': speaker, 
            'Frase': text, 
            'Sentimento': sentiment_label, 
            'Punteggio': sentiment_score
        })

    # Converti i risultati in un DataFrame
    df = pd.DataFrame(sentiment_results)
    
    # Crea una versione testuale della trascrizione
    transcript_text = "\n\n".join([f"{row['Parlante']}: {row['Frase']}" for index, row in df.iterrows()])

    # Creazione del grafico dettagliato
    fig = px.scatter(df, 
                     x='Turno', 
                     y='Punteggio', 
                     color='Parlante', 
                     size='Punteggio', 
                     hover_data=['Frase'], 
                     title='Analisi Sentimentale per Turno e Parlante',
                     labels={'Punteggio': 'Punteggio Sentimentale', 'Turno': 'Turno di Parlato'})

    # Salva il grafico come immagine temporanea
    with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmpfile:
        pio.write_image(fig, tmpfile.name)
        plot_path = tmpfile.name

    # Calcolo dell'analisi sentimentale totale per ciascun parlante
    sentiment_totals = []
    for speaker in df['Parlante'].unique():
        speaker_data = df[df['Parlante'] == speaker]
        avg_score = speaker_data['Punteggio'].mean()   # Media dei punteggi
        num_phrases = len(speaker_data)            # Numero di frasi per speaker
        most_common_sentiment = speaker_data['Sentimento'].mode()[0]  # Sentiment più comune
        sentiment_totals.append({
            'Parlante': speaker,
            'Sentimento più comune': most_common_sentiment,
            'Numero di frasi': num_phrases,
            'Media dei punteggi': f"{avg_score:.2f}"
        })

    return transcript_text, df, pd.DataFrame(sentiment_totals), plot_path

def create_gradio_interface():
    iface = gr.Interface(
        fn=process_video,
        inputs=gr.Video(label="Carica Video"), 
        outputs=[
            gr.Textbox(label="Trascrizione"),
            gr.Dataframe(label="Tabella analisi sentimentale per turni", datatype=['str', 'str', 'str', 'number']),
            gr.Dataframe(label="Tabella analisi sentimentale per parlante"),
            gr.Image(label="Grafico analisi sentimentale")
        ],
        title="Analisi audio di un video",
        description="Carica un video per generare la trascrizione dell'audio e l'analisi sentimentale per turni e per parlante.",
        theme="default",
        live=False
    )
    with gr.Blocks() as iface:
        gr.Markdown("<h2 style='text-align: center;'><strong>Analisi audio di un video</strong></h2>")
        gr.Markdown("Carica un video per generare la trascrizione dell'audio e l'analisi sentimentale per turni e per parlante.")
        with gr.Row():
            with gr.Column(scale=6):
                video_input = gr.Video(label="Carica Video", elem_id="video-box")
                #submit_button = gr.Button("Submit")
            with gr.Column(scale=6):
                transcript_output = gr.Textbox(label="Trascrizione", lines=10, elem_id="transcript-box")
        with gr.Row():
            with gr.Column(scale=3):
                clear_button = gr.Button("Cancella", elem_id="clear-btn")
            with gr.Column(scale=3):
                submit_button= gr.Button("Invia", elem_id="submit-btn")
            with gr.Column(scale=3):
                pass
            with gr.Column(scale=3):
                pass
        table_output = gr.Dataframe(label="Tabella analisi sentimentale per turni", datatype=['str', 'str', 'str', 'number'])
        total_sentiment_output = gr.DataFrame(label="Tabella analisi sentimentale per parlante")
        plot_output = gr.Image(label="Grafico analisi sentimentale")
        # Collegamento degli input agli output
        submit_button.click(process_video, inputs=[video_input], outputs=[transcript_output, table_output, total_sentiment_output, plot_output])
        # Funzione per ripulire i risultati quando si preme Clear
        def clear_outputs():
            return None, "", None, None, None
        # Collegamento del pulsante Clear per resettare gli output
        clear_button.click(clear_outputs, inputs=None, outputs=[video_input,transcript_output, table_output, total_sentiment_output, plot_output])

    return iface