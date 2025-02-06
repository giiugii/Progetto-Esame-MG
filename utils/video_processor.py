#importazioni
import whisper 
from pyannote.audio import Pipeline
import os
import moviepy.editor as mp
from pyannote.core import Segment

def get_text_with_timestamp(transcribe_res):
    timestamp_texts = []
    for item in transcribe_res['segments']:
        start = item['start']
        end = item['end']
        text = item['text']
        timestamp_texts.append((Segment(start, end), text))
    return timestamp_texts

def add_speaker_info_to_text(timestamp_texts, ann):
    spk_text = []
    for seg, text in timestamp_texts:
        spk = ann.crop(seg).argmax()  
        spk_text.append((seg, spk, text))
    return spk_text

def merge_cache(text_cache):
    sentence = ''.join([item[-1] for item in text_cache])
    spk = text_cache[0][1]
    start = text_cache[0][0].start
    end = text_cache[-1][0].end
    return Segment(start, end), spk, sentence

def merge_sentence(spk_text):
    merged_spk_text = []
    pre_spk = None
    text_cache = []
    for seg, spk, text in spk_text:
        if spk != pre_spk and pre_spk is not None and len(text_cache) > 0:
            merged_spk_text.append(merge_cache(text_cache))
            text_cache = [(seg, spk, text)]
            pre_spk = spk

        elif text and len(text) > 0 and text[-1] in ['.', '!', '?']:  
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

def diarize_and_transcribe_audio(audio_file):
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token="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")
    try:
        video = mp.VideoFileClip(audio_file)
        audio_path = "audio.wav"  
        video.audio.write_audiofile(audio_path)
    except Exception as e:
        print(f"Errore durante l'estrazione dell'audio: {e}")
        return f"Errore durante l'estrazione dell'audio: {e}"

    model = whisper.load_model("base")

    try:
        result = model.transcribe(audio_path, language="it")
        timestamp_texts = get_text_with_timestamp(result)
        diarization = pipeline(audio_path)
        spk_text = add_speaker_info_to_text(timestamp_texts, diarization)
        res_processed = merge_sentence(spk_text)
        os.remove(audio_path)
        return res_processed
    
    except Exception as e:
        print(f"Errore nella diarizzazione o trascrizione: {e}")
        return "Errore nella diarizzazione o trascrizione"