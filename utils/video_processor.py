import whisper 
from pyannote.audio import Pipeline
import os
import moviepy.editor as mp
from pyannote.core import Segment

# Funzione per ottenere i segmenti di testo con i timestamp
def get_text_with_timestamp(transcribe_res):
    timestamp_texts = []
    for item in transcribe_res['segments']:
        start = item['start']
        end = item['end']
        text = item['text']
        timestamp_texts.append((Segment(start, end), text))
    return timestamp_texts

# Funzione per associare gli speaker ai segmenti di testo
def add_speaker_info_to_text(timestamp_texts, ann):
    spk_text = []
    for seg, text in timestamp_texts:
        spk = ann.crop(seg).argmax()  # Identifica lo speaker in base al segmento
        spk_text.append((seg, spk, text))
    return spk_text

# Funzione per unire i segmenti di testo dello stesso speaker
def merge_cache(text_cache):
    sentence = ''.join([item[-1] for item in text_cache])
    spk = text_cache[0][1]
    start = text_cache[0][0].start
    end = text_cache[-1][0].end
    return Segment(start, end), spk, sentence

# Funzione per unire le frasi
def merge_sentence(spk_text):
    merged_spk_text = []
    pre_spk = None
    text_cache = []
    for seg, spk, text in spk_text:
        if spk != pre_spk and pre_spk is not None and len(text_cache) > 0:
            merged_spk_text.append(merge_cache(text_cache))
            text_cache = [(seg, spk, text)]
            pre_spk = spk

        elif text and len(text) > 0 and text[-1] in ['.', '!', '?']:  # Punteggiatura di fine frase
            text_cache.append((seg, spk, text))
            merged_spk_text.append(merge_cache(text_cache))
            text_cache = []
            pre_spk = spk
        else:
            text_cache.append((seg, spk, text))
            pre_spk = spk
    if len(text_cache) > 0:
        merged_spk_text.append(merge_cache(text_cache))
    return merged_spk_text

# Funzione principale per la diarizzazione e trascrizione
def diarize_and_transcribe_audio(audio_file):
    # Carica il pipeline di diarizzazione
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token="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")

    # Usa MoviePy per estrarre l'audio dal video
    try:
        video = mp.VideoFileClip(audio_file)
        audio_path = "audio.wav"  # Salviamo l'audio come file audio.wav
        video.audio.write_audiofile(audio_path)
    except Exception as e:
        print(f"Errore durante l'estrazione dell'audio: {e}")
        return f"Errore durante l'estrazione dell'audio: {e}"

    # Carica il modello Whisper per la trascrizione
    model = whisper.load_model("base")

    try:
        # Esegui la trascrizione con Whisper
        result = model.transcribe(audio_path, language="it")
        
        # Ottieni i segmenti di testo con timestamp
        timestamp_texts = get_text_with_timestamp(result)
        
        # Esegui la diarizzazione sull'audio estratto
        diarization = pipeline(audio_path)
        
        # Associa gli speaker ai segmenti di testo
        spk_text = add_speaker_info_to_text(timestamp_texts, diarization)
        
        # Unisci i segmenti di testo per ciascun speaker
        res_processed = merge_sentence(spk_text)
        
        # Elimina il file temporaneo dopo averlo utilizzato
        os.remove(audio_path)
        
        # Restituisci i risultati processati
        return res_processed
    
    except Exception as e:
        print(f"Errore nella diarizzazione o trascrizione: {e}")
        return "Errore nella diarizzazione o trascrizione"

#use_auth_token="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"
