#importazioni
from flask import Flask, jsonify, request, render_template, send_file
from utils.interface import create_gradio_interface
import os
import threading
import pandas as pd

#creazione dell'app flask
app = Flask(__name__)

UPLOAD_FOLDER= 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['ALLOWED_EXTENSIONS'] = {'mp3', 'wav', 'mp4'}
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

UPLOAD_FOLDER_2 = './saved_results'
os.makedirs(UPLOAD_FOLDER_2, exist_ok=True)
def save_results_to_csv(df, filename="results.csv"):
    filepath = os.path.join(UPLOAD_FOLDER_2, filename)
    df.to_csv(filepath, index=False)
    return filepath

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_audio():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    if file:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        if not os.path.exists(os.path.dirname(file_path)):
            os.makedirs(os.path.dirname(file_path))
        file.save(file_path)
        return jsonify({'filepath': file_path})
    return jsonify({'error': 'File not allowed'})

@app.route('/save_results', methods=['POST'])
def save_results():
    data = request.json 
    sentiment_data = data.get('sentiment_results')
    sentiment_data_df = pd.DataFrame(sentiment_data)
    filepath = save_results_to_csv(sentiment_data_df)
    return jsonify({
        "message": "Dati processati e salvati", "file": filepath
    })

@app.route('/get_results', methods=['GET'])
def get_results():
    filename = "results.csv"
    filepath = os.path.join(UPLOAD_FOLDER_2, filename)
    if os.path.exists(filepath):
        return send_file(filepath, as_attachment=True)
    else:
        return jsonify({"error": "File non trovato"}), 404

def run_gradio():
    interface = create_gradio_interface()
    interface.launch(server_name="127.0.0.1", server_port=None, share=False, inline=False, debug=True)

def run_flask():
    app.run(debug=True, use_reloader= False)

#avvio dell'app principale 
if __name__ == '__main__':
    flask_thread = threading.Thread(target=run_flask)
    flask_thread.start()
    run_gradio()
    flask_thread.join()