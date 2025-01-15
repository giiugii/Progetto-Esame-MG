#Script principale

from flask import Flask, request, jsonify
import whisper
from moviepy.video.io.VideoFileClip import VideoFileClip
import os
import srt
from pyannote.audio import Pipeline

app = Flask(__name__)

# Caricamento del modello Whisper
model = whisper.load_model("base")

# Caricamento del modello di diarizzazione
diarization_pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization")

@app.route('/upload', methods=['POST'])
def upload_video():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    video_file = request.files['file']
    file_path = os.path.join('uploads', video_file.filename)
    video_file.save(file_path)

    # Trascrizione con Whisper
    result = model.transcribe(file_path)
    transcription = result['text']

    # Creazione del file .srt
    subtitles = []
    for segment in result['segments']:
        start = segment['start']
        end = segment['end']
        text = segment['text']
        subtitles.append(srt.Subtitle(index=len(subtitles)+1, start=srt.srt_timestamp_to_timedelta(start), end=srt.srt_timestamp_to_timedelta(end), content=text))

    srt_path = file_path.replace('.mp4', '.srt')
    with open(srt_path, 'w', encoding='utf-8') as f:
        f.write(srt.compose(subtitles))

    # Diarizzazione
    diarization_result = diarization_pipeline({'uri': 'audio', 'audio': file_path})
    diarization_segments = [
        {
            "speaker": turn.speaker,
            "start": turn.start,
            "end": turn.end
        } for turn in diarization_result.itertracks(yield_label=True)
    ]

    return jsonify({
        'message': 'File processed successfully',
        'srt_path': srt_path,
        'diarization': diarization_segments
    })

if __name__ == '__main__':
    os.makedirs('uploads', exist_ok=True)
    app.run(debug=True)
