
import os
import uuid
from threading import Thread, Lock
from flask import Flask, request, jsonify, render_template, send_file, send_from_directory
from werkzeug.utils import secure_filename
from ..downloader import VideoDownloader
from ..transcriber import SubtitleGenerator
from ..translator import SubtitleTranslator
from ..utils import format_time

app = Flask(__name__, template_folder='templates')

# State
download_lock = Lock()
fetch_status = {}
extract_lock = Lock()
extract_jobs = {}

# Services
# Initialize lazy to avoid startup overhead until needed? 
# Or just initialize on module load.
downloader = VideoDownloader('./save')
generator = SubtitleGenerator() # Will auto-detect GPU
translator = SubtitleTranslator()

AVAILABLE_MODELS = ["tiny", "base", "small", "medium", "large"]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/save/<path:filename>')
def serve_video(filename):
    save_dir = os.path.join(os.getcwd(), 'save')
    return send_from_directory(save_dir, filename)


# --- Fetch API ---

@app.route('/fetch', methods=['POST'])
def fetch_video():
    data = request.get_json()
    url = data.get('url')
    if not url:
        return jsonify({'error': 'url is required'}), 400
    
    video_id = str(uuid.uuid4())
    # Try to get stable ID if possible, but keep simple for now
    try:
         video_id = downloader.get_video_id(url)
    except:
         pass

    with download_lock:
        fetch_status[video_id] = {'status': 'Downloading...', 'path': None, 'error': None}
    
    def background_fetch(u, vid):
        try:
            path = downloader.download_video(u, vid)
            with download_lock:
                 fetch_status[vid] = {'status': 'Ready', 'path': path, 'error': None}
        except Exception as e:
             with download_lock:
                 fetch_status[vid] = {'status': 'Error', 'path': None, 'error': str(e)}

    Thread(target=background_fetch, args=(url, video_id)).start()
    return jsonify({'video_id': video_id, 'message': 'Download started'}), 202

@app.route('/fetch/status')
def get_fetch_status():
    video_id = request.args.get('video_id')
    if not video_id:
        return jsonify({'error': 'video_id is required'}), 400
    with download_lock:
        info = fetch_status.get(video_id)
    if not info:
        return jsonify({'error': 'video_id not found'}), 404
    return jsonify(info)

# --- Upload API ---

@app.route('/upload', methods=['POST'])
def upload_video():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    uploaded_file = request.files['file']
    if uploaded_file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if not os.path.exists('./save'):
        os.makedirs('./save')
        
    filename = secure_filename(uploaded_file.filename)
    path = os.path.join('./save', filename)
    uploaded_file.save(path)
    return jsonify({'video_path': path})

# --- Extract API ---

@app.route('/extract_async', methods=['POST'])
def extract_subtitles_async():
    data = request.get_json()
    video_path = data.get('video_path')
    model_name = data.get('model', 'tiny')
    language = data.get('language')
    use_faster = data.get('use_faster', False)
    
    if not video_path or not os.path.exists(video_path):
        return jsonify({'error': 'Video file not found'}), 400

    job_id = str(uuid.uuid4())
    with extract_lock:
        extract_jobs[job_id] = {'state': 'queued', 'percent': 0, 'message': 'Queued'}

    def background_extract(jid, vpath, mname, lang, faster):
        try:
            with extract_lock:
                extract_jobs[jid]['state'] = 'running'
                
            def progress(msg, percent):
                with extract_lock:
                    extract_jobs[jid]['message'] = msg
                    extract_jobs[jid]['percent'] = percent
            
            vtt_path, vtt_content = generator.generate_subtitles(
                vpath, model_name=mname, language=lang, use_faster=faster, progress_callback=progress
            )
            
            with extract_lock:
                extract_jobs[jid].update({
                    'state': 'done',
                    'percent': 100,
                    'message': 'Completed',
                    'subtitles_path': vtt_path,
                    'vtt_content': vtt_content
                })
        except Exception as e:
            with extract_lock:
                 extract_jobs[jid].update({'state': 'error', 'message': str(e)})

    Thread(target=background_extract, args=(job_id, video_path, model_name, language, use_faster)).start()
    return jsonify({'job_id': job_id}), 202

@app.route('/extract/status')
def extract_status():
    job_id = request.args.get('job_id')
    with extract_lock:
        info = extract_jobs.get(job_id)
    if not info:
        return jsonify({'error': 'Not found'}), 404
    return jsonify(info)

# --- Models API ---

@app.route('/models')
def get_models():
    return jsonify(AVAILABLE_MODELS) # Using the constant list for now

@app.route('/models/status')
def get_model_status():
    # Mock status for now, or check cache
    return jsonify({'whisper': {}, 'translation': {}}) # Simplified

@app.route('/translation_pairs')
def get_pairs():
    return jsonify(translator.get_supported_pairs())

# --- Translate API ---

@app.route('/translate', methods=['POST'])
def translate_subtitles():
    data = request.get_json()
    vtt_content = data.get('vtt_content')
    source_lang = data.get('source_lang')
    target_lang = data.get('target_lang')
    video_path = data.get('video_path', 'unknown_video')
    
    if not vtt_content:
        return jsonify({'error': 'No content'}), 400

    try:
        # Simple line-by-line translation
        lines = vtt_content.strip().split('\n')
        translated_vtt = "WEBVTT\n\n"
        
        i = 1
        # Skipping simple implementation details for brevity, usually we need robust parsing
        # Re-using the logic:
        # We need to parse VTT.
        # Let's do a naive parse as in original app.py
        
        # Find model
        model_name = None
        for pair in translator.get_supported_pairs():
            if pair['source'] == source_lang and pair['target'] == target_lang:
                model_name = pair['model']
                break
        
        if not model_name:
             return jsonify({'error': 'Unsupported pair'}), 400
             
        # Batch translation would be better but simple loop for now
        i = 1
        while i < len(lines):
            if '-->' in lines[i]:
                 # timestamp line
                 ts_line = lines[i]
                 text_line = lines[i+1] if i+1 < len(lines) else ""
                 
                 trans_text = translator.translate_text(text_line, source_lang, target_lang, model_name)
                 
                 translated_vtt += f"{lines[i-1]}\n{ts_line}\n{trans_text}\n\n"
                 i += 2 # Skip timestamp and text
                 # Check if next is newline
                 while i < len(lines) and lines[i].strip() == '':
                     i += 1
            else:
                 i += 1
                 
        base = os.path.splitext(os.path.basename(video_path))[0]
        out_path = os.path.join('./save', f"{base}_{source_lang}_{target_lang}.vtt")
        with open(out_path, 'w', encoding='utf-8') as f:
            f.write(translated_vtt)
            
        return jsonify({'translated_vtt_path': out_path, 'translated_vtt_content': translated_vtt})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/refine_subtitle', methods=['POST'])
def refine_api():
    data = request.get_json()
    text = data.get('text')
    # ... simple wrapper
    return jsonify({'refined_text': text}) # Placeholder unless we fully port the Qwen logic which is in translator

@app.route('/api/translate', methods=['POST'])
def api_translate():
    data = request.get_json()
    text = data.get('text')
    src = data.get('source_lang')
    tgt = data.get('target_lang')
    use_qwen = data.get('use_qwen', False)
    
    try:
        if use_qwen:
            res = translator.translate_with_qwen(text, src, tgt)
            return jsonify({'translated_text': res})
        else:
            res = translator.translate_text(text, src, tgt)
            return jsonify({'translated_text': res})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def start_server(port=7860):
    app.run(host='0.0.0.0', port=port, debug=True)

if __name__ == '__main__':
    start_server()
