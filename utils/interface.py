#importazioni
import gradio as gr
from utils.analysis import analyze_sentiment, load_emolex, analizza_emozioni
from utils.video_processor import diarize_and_transcribe_audio
import pandas as pd
import plotly.express as px
import plotly.io as pio
import tempfile
import requests
import plotly.graph_objects as go

emolex = load_emolex(r"/Users/giuliatonielli/Desktop/Progetto-Esame-MG/Italian-NRC-EmoLex.txt")

def create_emotion_chart(emotion_df):
    emotions = ['Anticipazione', 'Rabbia', 'Paura', 'Gioia', 'Tristezza', 'Sorpresa', 'Disgusto', 'Fiducia', 'Negativo', 'Positivo']
    
    fig = go.Figure()
    for speaker in emotion_df['Parlante'].unique():
        speaker_data = emotion_df[emotion_df['Parlante'] == speaker]
        fig.add_trace(go.Bar(
            x=emotions,
            y=[speaker_data[emotion].mean() for emotion in emotions],
            name=speaker
        ))
    
    fig.update_layout(
        barmode='group',
        title='Analisi Emozionale per Parlante',
        xaxis_title='Emozioni',
        yaxis_title='Media dei punteggi',
        legend_title='Parlante'
    )
    
    with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmpfile:
        pio.write_image(fig, tmpfile.name)
        return tmpfile.name

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

        emotion_scores = analizza_emozioni(text, emolex)
        sentiment_label, sentiment_score = analyze_sentiment(text)
        
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
            'Anticipazione': emotion_scores['Anticipation'],
            'Rabbia': emotion_scores['Anger'],
            'Paura': emotion_scores['Fear'],
            'Gioia': emotion_scores['Joy'],
            'Tristezza': emotion_scores['Sadness'],
            'Sorpresa': emotion_scores['Surprise'],
            'Disgusto': emotion_scores['Disgust'],
            'Fiducia': emotion_scores['Trust'],
            'Negativo': emotion_scores['Negative'],
            'Positivo': emotion_scores['Positive']
        })

    df = pd.DataFrame(sentiment_results)
    emotion_df = pd.DataFrame(emotion_results)

    if not emotion_df.empty:
        emotion_chart_path = create_emotion_chart(emotion_df)
    else:
        emotion_chart_path = None

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
    
    return transcript_text, df, emotion_df, pd.DataFrame(sentiment_totals), plot_path, emotion_chart_path



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
        
        table_output = gr.Dataframe(label="Tabella analisi sentimentale per turni", datatype=['str', 'str', 'str', 'number'])
        table_emotion = gr.Dataframe(label="Tabella analisi emozionale per turni", datatype=['str', 'str', 'str', 'number', 'number', 'number', 'number', 'number', 'number', 'number'])
        total_sentiment_output = gr.Dataframe(label="Tabella analisi sentimentale per parlante")
        plot_output = gr.Image(label="Grafico analisi sentimentale")
        emotion_plot_output = gr.Image(label="Grafico analisi emozionale")

        submit_button.click(process_video, inputs=[video_input], outputs=[transcript_output, table_output, table_emotion, total_sentiment_output, plot_output, emotion_plot_output])        
        
        def clear_outputs():
            return None, "", None, None, None, None, None
        clear_button.click(clear_outputs, inputs=None, outputs=[video_input, transcript_output, table_output, table_emotion, total_sentiment_output, plot_output, emotion_plot_output])

    return iface