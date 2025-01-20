# utils/interface.py

import gradio as gr
import requests

def process_video(video):
    files = {'file': video}
    response = requests.post('http://127.0.0.1:5000/upload', files=files)
    if response.status_code == 200:
        result = response.json()
        # Leggiamo il file SRT
        with open(result['srt_path'], 'r') as file:
            srt_content = file.read()
        # Pulizia del file temporaneo SRT
        os.unlink(result['srt_path'])
        return srt_content, result['diarization']
    else:
        return "Errore: " + response.json().get('error', 'Unknown error'), None

def create_gradio_interface():
    iface = gr.Interface(
        fn=process_video,
        inputs=gr.Video(label="Carica un video"),
        outputs=[
            gr.Textbox(label="Sottotitoli (.srt)"),
            gr.Textbox(label="Diarizzazione (turni di parola)")
        ],
        title="Generatore di Sottotitoli e Diarizzazione",
        description="Carica un video per generare sottotitoli (.srt) e ottenere la diarizzazione dei turni di parola.",
        theme="default",
        live=False  # Impostato a False per evitare esecuzioni in tempo reale non necessarie
    )
    return iface