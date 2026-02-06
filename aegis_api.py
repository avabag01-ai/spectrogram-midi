import os
import tempfile
import base64
import io
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from aegis_engine import AegisEngine
import librosa
import numpy as np

app = Flask(__name__)
CORS(app)

engine = AegisEngine()
data_cache = {
    "last_key": None,
    "raw_data": None,
    "analysis_file": None,
    "start_time": 0,
    "end_time": 0
}

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    temp_dir = tempfile.gettempdir()
    file_path = os.path.join(temp_dir, file.filename)
    file.save(file_path)
    
    duration = float(librosa.get_duration(path=file_path))
    
    return jsonify({
        "file_path": file_path,
        "filename": file.filename,
        "duration": duration
    })

@app.route('/analyze', methods=['POST'])
def analyze():
    params = request.json
    file_path = params.get('file_path')
    start_time = params.get('start_time', 0)
    end_time = params.get('end_time', None)
    turbo_mode = params.get('turbo_mode', False)
    rake_sens = params.get('rake_sens', 0.6)
    full_band_mode = params.get('full_band_mode', False)
    
    if not file_path or not os.path.exists(file_path):
        return jsonify({"error": "File not found"}), 404
    
    current_key = f"{file_path}_{start_time}_{end_time}_{full_band_mode}_{rake_sens}_{turbo_mode}"
    
    if data_cache["last_key"] != current_key:
        analysis_file = file_path
        if full_band_mode:
            temp_dir = tempfile.gettempdir()
            analysis_file = engine.separate_stems(file_path, temp_dir)
            
        raw_data = engine.audio_to_midi(
            analysis_file, 
            output_mid=None,
            start_time=start_time,
            end_time=end_time,
            turbo_mode=turbo_mode,
            rake_sensitivity=rake_sens
        )
        
        data_cache["last_key"] = current_key
        data_cache["raw_data"] = raw_data
        data_cache["analysis_file"] = analysis_file
        data_cache["start_time"] = start_time
        data_cache["end_time"] = end_time

    return jsonify({"status": "success", "message": "Analysis cached"})

@app.route('/filter', methods=['POST'])
def filter_midi():
    if not data_cache["raw_data"]:
        return jsonify({"error": "No analysis data cached"}), 400
        
    params = request.json
    c_thresh = params.get('confidence', 0.70)
    s_ms = params.get('sustain', 70)
    m_ms = params.get('min_duration', 100)
    p_num = params.get('patch', 27)
    
    midi_buffer = io.BytesIO()
    events = engine.extract_events(
        data_cache["raw_data"],
        midi_buffer,
        min_note_duration_ms=m_ms,
        confidence_threshold=c_thresh,
        midi_program=p_num,
        sustain_ms=s_ms
    )
    
    midi_buffer.seek(0)
    midi_data = midi_buffer.read()
    midi_b64 = base64.b64encode(midi_data).decode('utf-8')
    
    return jsonify({
        "midi_b64": midi_b64,
        "event_count": len(events),
        "events": events[:100] # Return first 100 for preview
    })

if __name__ == '__main__':
    app.run(port=5005, debug=False)
