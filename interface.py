#interfaccia Gradio per il caricamento e output

import gradio as gr
import requests

def process_video(video):
    files = {'file': video}
    response = requests.post('http://127.0.0.1:5000/upload', files=files)
    if response.status_code == 200:
        result = response.json()
        return result['srt_path'], result['diarization']
    else:
        return "Errore: " + response.json().get('error', 'Unknown error')

interface = gr.Interface(
    fn=process_video,
    inputs=gr.Video(label="Carica un video"),
    outputs=[
        gr.File(label="Sottotitoli (.srt)"),
        gr.JSON(label="Diarizzazione (turni di parola)")
    ],
    title="Generatore di Sottotitoli e Diarizzazione"
)

interface.launch()
