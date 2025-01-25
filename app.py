from flask import Flask, render_template, request
from utils.interface import create_gradio_interface
from multiprocessing import Process
import gradio

app = Flask(__name__)

# Funzione per avviare Gradio in un thread separato
def run_gradio():
    interface = create_gradio_interface()
    interface.launch(server_name="127.0.0.1", server_port=None, share=False, inline=False, debug=True)

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    # Avvia Gradio in un processo separato
    p = Process(target=run_gradio)
    p.start()
    # Esegui il tuo Flask normalmente
    app.run(debug=True, port=5000)