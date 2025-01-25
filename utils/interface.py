import gradio as gr
from utils.analysis import analyze_sentiment
from utils.video_processor import extract_subtitles

def process_video(video_file):
    subtitles = extract_subtitles(video_file)
    sentiment_label, sentiment_score = analyze_sentiment(subtitles)
    return subtitles, sentiment_label, sentiment_score

def create_gradio_interface():
    iface = gr.Interface(
        fn=process_video,
        inputs=gr.Video(label="Carica Video"), 
        outputs=[gr.Textbox(label="Sottotitoli"), gr.Textbox(label="Analisi Sentimentale"), gr.Textbox(label="Punteggio Sentimentale")],
        title="Generatore di Sottotitoli",
        description="Carica un video per generare sottotitoli e per avere l'analisi sentimentale.",
        theme="default",
        live=False
    )
    return iface

