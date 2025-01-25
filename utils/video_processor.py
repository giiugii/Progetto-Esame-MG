import moviepy.editor as mp
import whisper 
    
# Funzione per estrarre i sottotitoli dal video
def extract_subtitles(video_file):
    # Usa MoviePy per estrarre l'audio dal video
    video = mp.VideoFileClip(video_file)
    video.audio.write_audiofile("audio.wav")
    
    # Usa Whisper per trascrivere l'audio dal video
    model = whisper.load_model("base")  # Puoi scegliere un modello più grande se necessario (e.g., "small", "medium", "large")
    
    try:
        result = model.transcribe("audio.wav", language='it')  # Imposta la lingua su italiano
        # Restituisci la trascrizione
        return result['text']
    
    except Exception as e:  # Gestisce qualsiasi errore in modo generico
        return f"Errore nella trascrizione: {str(e)}"



