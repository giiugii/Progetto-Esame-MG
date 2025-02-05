#importazioni
import gradio as gr
from utils.analysis import analyze_sentiment, load_emolex, analyze_emotions_average
from utils.video_processor import diarize_and_transcribe_audio
import pandas as pd
import plotly.express as px
import plotly.io as pio
import tempfile
import requests
import matplotlib.pyplot as plt

emolex = load_emolex('utils/Italian-NRC-EmoLex.txt')

def process_video(audio_file): 
    with open(audio_file, 'rb') as f:
        response = requests.post(
            'http://127.0.0.1:5000/upload', 
            files={'file': f}
        )
    if response.status_code != 200:
        return "Errore nel caricamento dell'audio"
    file_data = response.json()
    if 'filepath' in file_data:
        audio_file = file_data['filepath']  
    else:
        return "Errore: percorso file non trovato nella risposta"

    subtitles = diarize_and_transcribe_audio(audio_file)

    sentiment_results = []
    emotion_results = []
    
    for segment in subtitles:
        speaker, text = segment[1], segment[2]

        sentiment_label, sentiment_score = analyze_sentiment(text)
        emotion_scores = analyze_emotions_average(text, emolex)
        
        sentiment_results.append({
            'Turno': f"Turno {len(sentiment_results) + 1}",
            'Parlante': speaker, 
            'Frase': text, 
            'Sentimento': sentiment_label, 
            'Punteggio di fiducia': sentiment_score
        })
        
        emotion_results.append({
            'Turno': f"Turno {len(emotion_results) + 1}",
            'Parlante': speaker,
            'Frase': text,
            **emotion_scores 
        })

    df = pd.DataFrame(sentiment_results)
    emotion_df = pd.DataFrame(emotion_results)

    transcript_text = "\n\n".join([f"{row['Parlante']}: {row['Frase']}" for index, row in df.iterrows()])

    sentiment_mapping = {
        'Positivo': 1,
        'Neutro': 0,
        'Negativo': -1
    }
    df['sentimento_numerico'] = df['Sentimento'].map(sentiment_mapping)

    fig = px.scatter(df, 
                     x='Turno', 
                     y='sentimento_numerico', 
                     color='Parlante', 
                     size='Punteggio di fiducia', 
                     title='Grafico analisi sentimentale',
                     labels={'sentimento_numerico': 'Sentimento', 'Turno': 'Turno di Parlato'},
                     category_orders={'sentimento_numerico': [-1, 0, 1]}) 
    
    df.drop(columns=['sentimento_numerico'], inplace=True)
    fig.update_layout(
        yaxis=dict(
            tickmode='array',
            tickvals=[-1, 0, 1],
            ticktext=['Negativo', 'Neutro', 'Positivo'])
    )
    with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmpfile:
        pio.write_image(fig, tmpfile.name)
        plot_path = tmpfile.name
    
    emotion_columns = [col for col in emotion_df.columns if col not in ['Turno', 'Parlante', 'Frase']]
    emotion_df[emotion_columns].plot(kind='bar')
    plt.title('Media delle emozioni per ciascun turno')
    plt.xlabel('Turni')
    plt.ylabel('Intensità delle emozioni')
    plt.xticks(range(len(emotion_df)), emotion_df['Turno'], rotation=45)
    with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmpfile:
        plt.tight_layout()
        plt.savefig(tmpfile, format='png') 
        plot_path_2 = tmpfile.name 

    sentiment_totals = []
    for speaker in df['Parlante'].unique():
        speaker_data = df[df['Parlante'] == speaker]
        avg_score = speaker_data['Punteggio di fiducia'].mean()   
        num_phrases = len(speaker_data)            
        most_common_sentiment = speaker_data['Sentimento'].mode()[0]  
        sentiment_totals.append({
            'Parlante': speaker,
            'Sentimento più comune': most_common_sentiment,
            'Numero di frasi': num_phrases,
            'Media dei punteggi di fiducia': f"{avg_score:.2f}"
        })
    
    response = requests.post("http://127.0.0.1:5000/save_results", json={
        'sentiment_results': df.to_dict(orient='records')
    })
    
    return transcript_text, df, emotion_df, pd.DataFrame(sentiment_totals), plot_path, plot_path_2

#funzione gradio
def create_gradio_interface():
    with gr.Blocks() as iface:
        gr.Markdown("<h2 style='text-align: center;'><strong>Analisi audio di un video</strong></h2>")
        gr.Markdown("Carica un video per ricevere la trascrizione dell'audio e l'analisi sentimentale per turni e per parlante.")
        with gr.Row():
            with gr.Column(scale=6):
                video_input = gr.Video(label="Carica Video", elem_id="video-box")
            with gr.Column(scale=6):
                transcript_output = gr.Textbox(label="Trascrizione", lines=10, elem_id="transcript-box")
        with gr.Row():
            with gr.Column(scale=3):
                clear_button = gr.Button("Cancella", elem_id="clear-btn")
            with gr.Column(scale=3):
                submit_button = gr.Button("Invia", elem_id="submit-btn")
            with gr.Column(scale=3):
                pass
            with gr.Column(scale=3):
                pass
        
        table_output = gr.Dataframe(label="Tabella analisi sentimentale per turni", datatype=['str', 'str', 'str', 'number'])
        table_emotion = gr.Dataframe(label="Tabella analisi emozionale per turni", datatype=['str', 'str', 'str', 'number', 'number', 'number', 'number', 'number', 'number', 'number'])
        total_sentiment_output = gr.Dataframe(label="Tabella analisi sentimentale per parlante")
        plot_output = gr.Image(label="Grafico analisi sentimentale")
        plot_output_2 = gr.Image(label="Grafico analisi emozionale")

        submit_button.click(process_video, inputs=[video_input], outputs=[transcript_output, table_output, table_emotion, total_sentiment_output, plot_output, plot_output_2])        
        
        def clear_outputs():
            return None, "", None, None, None, None, None
        clear_button.click(clear_outputs, inputs=None, outputs=[video_input, transcript_output, table_output, table_emotion, total_sentiment_output, plot_output, plot_output_2])

    return iface