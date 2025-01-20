# utils/video_processor.py

import whisper
from moviepy.editor import VideoFileClip
from utils.analysis import sentiment_analysis, diarize
import os

def extract_audio(video_path, audio_path="temp_audio.wav"):
    video = VideoFileClip(video_path)
    audio = video.audio
    audio.write_audiofile(audio_path, codec='pcm_s16le')
    video.close()
    return audio_path

def process_video(video_path):
    # Carica il modello di whisper
    model = whisper.load_model("base")
    
    # Estrai l'audio dal video
    temp_audio = extract_audio(video_path)
    
    # Transcrivi il video
    result = model.transcribe(temp_audio)
    srt_content = result['text']
    
    # Analisi sentimentale
    sentiment = sentiment_analysis(srt_content)
    
    # Diarizzazione
    diarize_data = diarize(temp_audio)
    
    # Rimuovi il file audio temporaneo
    os.unlink(temp_audio)
    
    return srt_content, sentiment, diarize_data