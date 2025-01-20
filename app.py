# app.py

from flask import Flask, render_template, request, jsonify, send_file
import os
import tempfile
from utils.video_processor import process_video
from utils.interface import create_gradio_interface
import threading

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    srt_content = ''
    sentiment = ''
    diarize_data = ''
    if request.method == 'POST':
        video_file = request.files['video']
        if video_file.filename.lower().endswith(('.mp4', '.avi', '.mov')):
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            video_file.save(temp_file.name)
            try:
                srt_content, sentiment, diarize_data = process_video(temp_file.name)
            except Exception as e:
                # Gestisci l'errore se il processamento fallisce
                srt_content = 'Errore durante il processamento del video'
                sentiment = 'Errore'
                diarize_data = f'Errore durante il processamento: {str(e)}'
            finally:
                os.unlink(temp_file.name)  # Rimuovi il file temporaneo
            
    return render_template('index.html', srt=srt_content, sentiment=sentiment, diarize=diarize_data)

@app.route('/upload', methods=['POST'])
def upload_video():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    if file:
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        file.save(temp_file.name)
        try:
            srt_content, sentiment, diarize_data = process_video(temp_file.name)
            # Creazione di un file SRT temporaneo
            srt_path = tempfile.NamedTemporaryFile(delete=False, suffix='.srt').name
            with open(srt_path, 'w') as f:
                f.write(srt_content)
            os.unlink(temp_file.name)  # Rimuovi il file video temporaneo
            return jsonify({'srt_path': srt_path, 'diarization': diarize_data}), 200
        except Exception as e:
            os.unlink(temp_file.name) if os.path.exists(temp_file.name) else None
            return jsonify({'error': str(e)}), 500

def run_gradio():
    iface = create_gradio_interface()
    iface.launch(share=True)

if __name__ == '__main__':
    # Avvia l'interfaccia di Gradio in un thread separato
    gradio_thread = threading.Thread(target=run_gradio)
    gradio_thread.start()
    
    # Flask app
    app.run(debug=True, port=5000)