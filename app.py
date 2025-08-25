from flask import Flask, request, jsonify, send_file
from urllib.parse import unquote
import subprocess
import whisper
import os
import time
from transformers import pipeline
from huggingface_hub import snapshot_download
from threading import Thread

app = Flask(__name__, static_folder='./save')

# --- Model Configuration ---
AVAILABLE_MODELS = ["tiny", "base", "small", "medium", "large"]
# New structure for translation models
SUPPORTED_TRANSLATION_PAIRS = [
    {"source": "English", "target": "Chinese", "model": "Helsinki-NLP/opus-mt-en-zh"},
    {"source": "English", "target": "French", "model": "Helsinki-NLP/opus-mt-en-fr"},
    {"source": "English", "target": "Spanish", "model": "Helsinki-NLP/opus-mt-en-es"},
    {"source": "English", "target": "German", "model": "Helsinki-NLP/opus-mt-en-de"},
    {"source": "Chinese", "target": "English", "model": "Helsinki-NLP/opus-mt-zh-en"},
]

# In-memory cache for loaded translation pipelines
translation_pipelines = {}

# --- Model Status & Management ---
def get_whisper_model_status(model_name):
    """Checks if a Whisper model is cached locally."""
    try:
        cache_path = os.path.expanduser(f"~/.cache/whisper/{model_name}.pt")
        return "Ready" if os.path.exists(cache_path) else "Not Downloaded"
    except Exception as e:
        print(f"Could not determine status for Whisper model {model_name}: {e}")
        return "Not Downloaded"

def get_hf_model_status(model_name):
    """Checks if a Hugging Face model is cached locally."""
    try:
        # This will check for the model in the cache without downloading it.
        # It raises an OSError if the model is not found locally.
        snapshot_download(repo_id=model_name, local_files_only=True)
        return "Ready"
    except Exception as e:
        # Catch any exception to prevent the endpoint from hanging
        print(f"Could not determine status for HF model {model_name}: {e}")
        return "Not Downloaded"

def get_translation_pipeline(model_name):
    """Loads a translation pipeline, caching it in memory by model name."""
    if model_name not in translation_pipelines:
        print(f"Loading translation model: {model_name}")
        translation_pipelines[model_name] = pipeline("translation", model=model_name)
    return translation_pipelines[model_name]

def download_model_in_background(model_type, model_key):
    """Target function for background download thread."""
    print(f"Starting download for {model_type} model: {model_key}")
    if model_type == 'whisper':
        whisper.load_model(model_key)
    elif model_type == 'translation':
        # model_key for translation is the full model name
        get_translation_pipeline(model_key)
    print(f"Finished download for {model_type} model: {model_key}")


# Route to serve the new frontend
@app.route('/')
def new_index():
    return send_file(os.path.join(os.getcwd(), 'app.html'))

# Route to get the list of available models
@app.route('/models')
def get_models():
    return jsonify(AVAILABLE_MODELS)

@app.route('/translation_pairs')
def get_translation_pairs():
    return jsonify(SUPPORTED_TRANSLATION_PAIRS)

@app.route('/models/status')
def get_all_model_statuses():
    statuses = {
        'whisper': {model: get_whisper_model_status(model) for model in AVAILABLE_MODELS},
        'translation': {pair['model']: get_hf_model_status(pair['model']) for pair in SUPPORTED_TRANSLATION_PAIRS}
    }
    return jsonify(statuses)

@app.route('/models/download', methods=['POST'])
def download_model():
    data = request.get_json()
    model_type = data.get('model_type')
    model_key = data.get('model_key') # For translation, this will be the full model name

    if not model_type or not model_key:
        return jsonify({'error': 'model_type and model_key are required'}), 400

    # Basic validation
    if model_type == 'whisper' and model_key not in AVAILABLE_MODELS:
        return jsonify({'error': 'Invalid whisper model key'}), 400
    if model_type == 'translation' and not any(p['model'] == model_key for p in SUPPORTED_TRANSLATION_PAIRS):
        return jsonify({'error': 'Invalid translation model key'}), 400

    thread = Thread(target=download_model_in_background, args=(model_type, model_key))
    thread.start()

    return jsonify({'message': f'Download started for {model_type} model: {model_key}'}), 202


# File upload endpoint
@app.route('/upload', methods=['POST'])
def upload_video():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    uploaded_file = request.files['file']
    if uploaded_file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if uploaded_file:
        if not os.path.exists('./save'):
            os.makedirs('./save')
        video_path = os.path.join('./save', uploaded_file.filename)
        uploaded_file.save(video_path)
        return jsonify({'video_path': video_path})

# Subtitle extraction
@app.route('/extract_subtitles')
def extract_subtitles():
    video_path = request.args.get('video_path')
    model_name = request.args.get('model', 'tiny')
    if not video_path: return jsonify({'error': 'Video file path is required'}), 400
    if model_name not in AVAILABLE_MODELS: return jsonify({'error': f'Invalid model name.'}), 400
    video_path = unquote(video_path)
    if not os.path.exists(video_path): return jsonify({'error': 'Video file not found'}), 400
    try:
        audio_output_path = os.path.splitext(video_path)[0] + '.mp3'
        ffmpeg_command = f"ffmpeg -i '{video_path}' -y -vn -ar 16000 -ac 1 -c:a libmp3lame -q:a 4 '{audio_output_path}'"
        subprocess.run(ffmpeg_command, shell=True, check=True)
        model = whisper.load_model(model_name)
        result = model.transcribe(audio_output_path, language='en')
        vtt_content = "WEBVTT\n\n"
        for i, seg in enumerate(result['segments'], start=1):
            start = format_time(seg['start'])
            end = format_time(seg['end'])
            text = seg['text'].strip()
            vtt_content += f"{i}\n{start} --> {end}\n{text}\n\n"
        subtitles_filename = os.path.splitext(os.path.basename(video_path))[0] + '.vtt'
        subtitles_path = os.path.join('./save', subtitles_filename)
        with open(subtitles_path, "w", encoding='utf-8') as f: f.write(vtt_content)
        return jsonify({'subtitles_path': subtitles_path, 'vtt_content': vtt_content})
    except Exception as e:
        return jsonify({'error': f'An error occurred: {e}'}), 500

# Subtitle translation
@app.route('/translate', methods=['POST'])
def translate_subtitles():
    data = request.get_json()
    vtt_content = data.get('vtt_content')
    source_lang = data.get('source_lang')
    target_lang = data.get('target_lang')
    video_path = data.get('video_path')

    if not all([vtt_content, source_lang, target_lang, video_path]):
        return jsonify({'error': 'Missing required data'}), 400

    # Find the correct model for the requested language pair
    model_name = None
    for pair in SUPPORTED_TRANSLATION_PAIRS:
        if pair['source'] == source_lang and pair['target'] == target_lang:
            model_name = pair['model']
            break

    if not model_name:
        return jsonify({'error': f'Translation from {source_lang} to {target_lang} is not supported.'}), 400

    translator = get_translation_pipeline(model_name)
    if not translator:
        return jsonify({'error': 'Could not load translation model.'}), 500

    try:
        lines = vtt_content.strip().split('\n')
        translated_vtt_content = "WEBVTT\n\n"
        i = 1
        while i < len(lines):
            if '-->' in lines[i]:
                text_line = lines[i+1]
                # Some models are fine with single line, some need a list. List is safer.
                translated_text = translator([text_line])[0]['translation_text']
                translated_vtt_content += f"{lines[i-1]}\n{lines[i]}\n{translated_text}\n\n"
                i += 2
            else:
                i += 1

        base_filename = os.path.splitext(os.path.basename(video_path))[0]
        translated_filename = f"{base_filename}_{source_lang}_to_{target_lang}.vtt"
        translated_subtitles_path = os.path.join('./save', translated_filename)
        with open(translated_subtitles_path, "w", encoding='utf-8') as f:
            f.write(translated_vtt_content)
        return jsonify({'translated_vtt_path': translated_subtitles_path, 'translated_vtt_content': translated_vtt_content})
    except Exception as e:
        return jsonify({'error': f'An error occurred during translation: {e}'}), 500

# Helper to format time for VTT
def format_time(seconds):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds - int(seconds)) * 1000)
    return f"{h:02}:{m:02}:{s:02}.{ms:03}"

if __name__ == '__main__':
    app.run(debug=True, port=5001)
