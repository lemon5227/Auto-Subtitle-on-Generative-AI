from flask import Flask, request, jsonify, send_file
from urllib.parse import unquote
import subprocess
import whisper
import os
import time
from transformers import pipeline

app = Flask(__name__, static_folder='./save')

# A list of available Whisper models
AVAILABLE_MODELS = ["tiny", "base", "small", "medium", "large"]
# A dictionary for supported translation languages and models
SUPPORTED_LANGUAGES = {
    "Chinese": "Helsinki-NLP/opus-mt-en-zh",
    "French": "Helsinki-NLP/opus-mt-en-fr",
    "Spanish": "Helsinki-NLP/opus-mt-en-es",
    "German": "Helsinki-NLP/opus-mt-en-de",
}
# Cache for translation models
translation_pipelines = {}

def get_translation_pipeline(lang):
    if lang not in SUPPORTED_LANGUAGES:
        return None
    if lang not in translation_pipelines:
        model_name = SUPPORTED_LANGUAGES[lang]
        print(f"Loading translation model for {lang}: {model_name}")
        translation_pipelines[lang] = pipeline("translation", model=model_name)
    return translation_pipelines[lang]


# Route to serve the new frontend
@app.route('/')
def new_index():
    return send_file(os.path.join(os.getcwd(), 'app.html'))

# Route to get the list of available models
@app.route('/models')
def get_models():
    return jsonify(AVAILABLE_MODELS)

@app.route('/languages')
def get_languages():
    return jsonify(list(SUPPORTED_LANGUAGES.keys()))

# File upload endpoint (remains the same for now)
@app.route('/upload', methods=['POST'])
def upload_video():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    uploaded_file = request.files['file']

    if uploaded_file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if uploaded_file:
        # Create save directory if it doesn't exist
        if not os.path.exists('./save'):
            os.makedirs('./save')

        video_path = os.path.join('./save', uploaded_file.filename)
        uploaded_file.save(video_path)
        return jsonify({'video_path': video_path})

# Subtitle extraction with model selection
@app.route('/extract_subtitles')
def extract_subtitles():
    video_path = request.args.get('video_path')
    model_name = request.args.get('model', 'tiny') # Default to 'tiny'

    if not video_path:
        return jsonify({'error': 'Video file path is required'}), 400

    if model_name not in AVAILABLE_MODELS:
        return jsonify({'error': f'Invalid model name. Please choose from {AVAILABLE_MODELS}'}), 400

    video_path = unquote(video_path)

    if not os.path.exists(video_path):
        return jsonify({'error': 'Video file not found'}), 400

    try:
        # Extract audio using FFmpeg
        audio_output_path = os.path.splitext(video_path)[0] + '.mp3'
        # Use '-y' to overwrite existing files
        ffmpeg_command = f"ffmpeg -i '{video_path}' -y -vn -ar 16000 -ac 1 -c:a libmp3lame -q:a 4 '{audio_output_path}'"
        subprocess.run(ffmpeg_command, shell=True, check=True)

        start_time = time.time()

        # Load the selected Whisper model
        model = whisper.load_model(model_name)
        result = model.transcribe(audio_output_path, language='en')
        segments = result['segments']

        elapsed_time = time.time() - start_time
        print(f"Whisper transcription with '{model_name}' model took: {elapsed_time:.2f}s")

        # Generate WebVTT content
        vtt_content = "WEBVTT\n\n"
        for i, seg in enumerate(segments, start=1):
            start = format_time(seg['start'])
            end = format_time(seg['end'])
            text = seg['text'].strip()
            vtt_content += f"{i}\n{start} --> {end}\n{text}\n\n"

        # Save subtitles to a file
        subtitles_filename = os.path.splitext(os.path.basename(video_path))[0] + '.vtt'
        subtitles_path = os.path.join('./save', subtitles_filename)
        with open(subtitles_path, "w", encoding='utf-8') as f:
            f.write(vtt_content)

        return jsonify({
            'subtitles_path': subtitles_path,
            'vtt_content': vtt_content,
        })

    except subprocess.CalledProcessError as e:
        return jsonify({'error': f'FFmpeg failed: {e}'}), 500
    except Exception as e:
        return jsonify({'error': f'An error occurred: {e}'}), 500


@app.route('/translate', methods=['POST'])
def translate_subtitles():
    data = request.get_json()
    vtt_content = data.get('vtt_content')
    target_lang = data.get('target_lang')
    video_path = data.get('video_path')

    if not all([vtt_content, target_lang, video_path]):
        return jsonify({'error': 'Missing required data: vtt_content, target_lang, or video_path'}), 400

    translator = get_translation_pipeline(target_lang)
    if not translator:
        return jsonify({'error': 'Unsupported language'}), 400

    try:
        lines = vtt_content.strip().split('\n')
        translated_vtt_content = "WEBVTT\n\n"

        # We process the VTT file in chunks (header, timestamp, text)
        i = 1 # Skip WEBVTT header
        while i < len(lines):
            if '-->' in lines[i]:
                # This is a time line, the next line is the text
                start_time_line = lines[i]
                text_line = lines[i+1]

                # Translate the text
                translated_text = translator(text_line)[0]['translation_text']

                # Reconstruct the VTT block
                translated_vtt_content += f"{lines[i-1]}\n" # Sequence number
                translated_vtt_content += f"{start_time_line}\n"
                translated_vtt_content += f"{translated_text}\n\n"
                i += 2
            else:
                i += 1

        # Save the translated VTT to a file
        base_filename = os.path.splitext(os.path.basename(video_path))[0]
        translated_filename = f"{base_filename}_{target_lang}.vtt"
        translated_subtitles_path = os.path.join('./save', translated_filename)

        with open(translated_subtitles_path, "w", encoding='utf-8') as f:
            f.write(translated_vtt_content)

        return jsonify({
            'translated_vtt_path': translated_subtitles_path,
            'translated_vtt_content': translated_vtt_content
        })

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
    app.run(debug=True, port=5001) # Running on a different port
