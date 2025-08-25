from flask import Flask, request, jsonify, send_file
from urllib.parse import unquote
import subprocess
import shutil as _shutil
import sys
import whisper
import os
import time
import shutil
import torch
from transformers import pipeline
from huggingface_hub import snapshot_download
from threading import Thread, Lock
import shlex
import glob
import uuid
# Try to import python API of yt-dlp; if not available we'll fall back to system binary at runtime
try:
    import yt_dlp as ytdlp_api
except Exception:
    ytdlp_api = None

# Optional faster-whisper support
try:
    from faster_whisper import WhisperModel as FasterWhisperModel
    FASTER_AVAILABLE = True
except Exception:
    FasterWhisperModel = None
    FASTER_AVAILABLE = False

app = Flask(__name__, static_folder='./save')

# Serve additional static assets from ./static under /static/<path:filename>
@app.route('/static/<path:filename>')
def extra_static(filename):
    safe_path = os.path.join(os.getcwd(), 'static', filename)
    if os.path.exists(safe_path) and os.path.isfile(safe_path):
        return send_file(safe_path)
    return jsonify({'error': 'not found'}), 404

# Startup checks: ensure ffmpeg is available and warn/exit if not
if _shutil.which('ffmpeg') is None:
    print("ERROR: 'ffmpeg' not found in PATH. Please install ffmpeg (apt/brew/conda) before running the server.")
    sys.exit(1)

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

# Device selection for model inference
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device for inference: {DEVICE}")

# Track download status in-memory to avoid reporting Ready before download completes
download_status = {}  # keys: ('whisper', model_key) or ('translation', model_key) -> status string
download_lock = Lock()
# Track background fetch jobs for video downloading
fetch_status = {}

# Extraction job tracking
extract_jobs = {}  # job_id -> {state, processed_chunks, total_chunks, percent, message, subtitles_path, vtt_content}
extract_lock = Lock()

# --- Model Status & Management ---
def get_whisper_model_status(model_name):
    """Checks if a Whisper model is cached locally."""
    try:
        # If there is an in-memory download status, prefer that
        key = ('whisper', model_name)
        with download_lock:
            if key in download_status:
                return download_status[key]
        cache_path = os.path.expanduser(f"~/.cache/whisper/{model_name}.pt")
        return "Ready" if os.path.exists(cache_path) else "Not Downloaded"
    except Exception as e:
        print(f"Could not determine status for Whisper model {model_name}: {e}")
        return "Not Downloaded"

def get_hf_model_status(model_name):
    """Checks if a Hugging Face model is cached locally."""
    # If there is an in-memory download status, prefer that
    key = ('translation', model_name)
    with download_lock:
        if key in download_status:
            return download_status[key]

    # Try to detect cached files under the Hugging Face cache directory without network calls
    def find_hf_cache_path(name):
        hf_home = os.getenv('HF_HOME') or os.path.expanduser('~/.cache/huggingface/hub')
        if not os.path.isdir(hf_home):
            return None
        target = name.replace('/', '-')
        # Walk shallowly: check top-level dirs to avoid heavy scans
        try:
            for entry in os.listdir(hf_home):
                entry_path = os.path.join(hf_home, entry)
                if target in entry:
                    return entry_path
                # also check one level deeper
                if os.path.isdir(entry_path):
                    for sub in os.listdir(entry_path):
                        if target in sub:
                            return os.path.join(entry_path, sub)
        except Exception:
            return None
        return None

    cache_path = find_hf_cache_path(model_name)
    return "Ready" if cache_path else "Not Downloaded"

def get_translation_pipeline(model_name):
    """Loads a translation pipeline, caching it in memory by model name."""
    if model_name not in translation_pipelines:
        print(f"Loading translation model: {model_name}")
        translation_pipelines[model_name] = pipeline("translation", model=model_name)
    return translation_pipelines[model_name]

def download_model_in_background(model_type, model_key):
    """Target function for background download thread."""
    key = (model_type, model_key)
    print(f"Starting download for {model_type} model: {model_key}")
    try:
        if model_type == 'whisper':
            try:
                whisper.load_model(model_key, device=DEVICE)
            except TypeError:
                # Older whisper versions may not accept device param; load then move
                m = whisper.load_model(model_key)
                try:
                    m.to(DEVICE)
                except Exception:
                    pass
        elif model_type == 'translation':
            # Ensure all files are downloaded to the HF cache first
            print(f"Snapshot downloading translation model {model_key}...")
            model_path = snapshot_download(repo_id=model_key, local_files_only=False)
            print(f"Snapshot download completed: {model_path}")
            # Then load pipeline (may reuse cached files)
            get_translation_pipeline(model_key)
        # mark ready
        with download_lock:
            download_status[key] = 'Ready'
        print(f"Finished download for {model_type} model: {model_key}")
    except Exception as e:
        with download_lock:
            download_status[key] = f'Error: {e}'
        print(f"Download failed for {model_type} model: {model_key}, error: {e}")


# Route to serve the new frontend
@app.route('/')
def new_index():
    return send_file(os.path.join(os.getcwd(), 'app.html'))

# Route to get the list of available models
@app.route('/models')
def get_models():
    return jsonify(AVAILABLE_MODELS)

# 删除模型接口（必须在 app 定义后）
@app.route('/models/delete', methods=['POST'])
def delete_model():
    data = request.get_json()
    model_type = data.get('model_type')
    model_key = data.get('model_key')
    if not model_type or not model_key:
        return jsonify({'error': 'model_type and model_key are required'}), 400
    if model_type == 'whisper':
        model_path = os.path.expanduser(f"~/.cache/whisper/{model_key}.pt")
        if os.path.exists(model_path):
            os.remove(model_path)
            # clear any in-memory status
            with download_lock:
                download_status.pop(('whisper', model_key), None)
            return jsonify({'message': f'Model {model_key} deleted.'}), 200
        else:
            return jsonify({'error': 'Model file not found.'}), 404
    elif model_type == 'translation':
        try:
            # Try to locate local cache path for the HF model; allow network=False so it fails if absent
            model_path = snapshot_download(repo_id=model_key, local_files_only=True)
        except Exception:
            # If not found in cache, try common cache locations
            model_path = None
            possible_cache = os.path.expanduser('~/.cache/huggingface/hub')
            if os.path.isdir(possible_cache):
                # Try to find folders that match model_key name
                for root, dirs, files in os.walk(possible_cache):
                    if model_key.replace('/', '-') in root:
                        model_path = root
                        break
        if model_path and os.path.exists(model_path):
            try:
                if os.path.isdir(model_path):
                    shutil.rmtree(model_path)
                else:
                    os.remove(model_path)
                with download_lock:
                    download_status.pop(('translation', model_key), None)
                return jsonify({'message': f'Translation model {model_key} deleted.'}), 200
            except Exception as e:
                return jsonify({'error': f'Could not delete translation model files: {e}'}), 500
        else:
            return jsonify({'error': 'Translation model files not found in cache.'}), 404
    else:
        return jsonify({'error': 'Invalid model type.'}), 400

@app.route('/translation_pairs')
def get_translation_pairs():
    return jsonify(SUPPORTED_TRANSLATION_PAIRS)

@app.route('/models/status')
def get_all_model_statuses():
    statuses = {
        'whisper': {model: get_whisper_model_status(model) for model in AVAILABLE_MODELS},
        'translation': {pair['model']: get_hf_model_status(pair['model']) for pair in SUPPORTED_TRANSLATION_PAIRS}
    }
    # overlay in-memory download statuses
    with download_lock:
        for (mtype, mkey), st in download_status.items():
            if mtype == 'whisper' and mkey in statuses['whisper']:
                statuses['whisper'][mkey] = st
            if mtype == 'translation' and mkey in statuses['translation']:
                statuses['translation'][mkey] = st
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

    # Set in-memory status and start background download
    key = (model_type, model_key)
    with download_lock:
        download_status[key] = 'Downloading...'
    thread = Thread(target=download_model_in_background, args=(model_type, model_key))
    thread.start()

    return jsonify({'message': f'Download started for {model_type} model: {model_key}'}), 202


@app.route('/fetch', methods=['POST'])
def fetch_video():
    """Start background fetch of a video URL using yt-dlp. Returns a video_id for polling."""
    data = request.get_json()
    url = data.get('url')
    if not url:
        return jsonify({'error': 'url is required'}), 400
    try:
        # ensure save dir
        if not os.path.exists('./save'):
            os.makedirs('./save')

        # Try to get a stable id for the URL; prefer Python API if available
        vid = None
        if ytdlp_api is not None:
            try:
                with ytdlp_api.YoutubeDL({'quiet': True}) as ydl:
                    info = ydl.extract_info(url, download=False)
                    vid = info.get('id') or str(uuid.uuid4())
            except Exception:
                vid = str(uuid.uuid4())
        else:
            # Only call system yt-dlp if it exists on PATH to avoid FileNotFoundError
            if _shutil.which('yt-dlp') is not None:
                try:
                    res = subprocess.run(['yt-dlp', '--get-id', url], capture_output=True, text=True, check=True)
                    vid = res.stdout.strip() or str(uuid.uuid4())
                except Exception:
                    vid = str(uuid.uuid4())
            else:
                # no python API and no system binary -> use uuid and let background worker report missing binary
                vid = str(uuid.uuid4())

        # register status and start background worker
        with download_lock:
            fetch_status[vid] = {'status': 'Downloading...', 'path': None, 'error': None}

        thread = Thread(target=fetch_video_in_background, args=(url, vid))
        thread.start()

        return jsonify({'video_id': vid, 'message': 'Download started'}), 202
    except Exception as e:
        return jsonify({'error': f'An error occurred: {e}'}), 500


def fetch_video_in_background(url, vid):
    """Worker to download video with yt-dlp and update fetch_status when done."""
    try:
        output_template = os.path.join('./save', f"{vid}.%(ext)s")
        if ytdlp_api is not None:
            # Prefer a reasonable quality for transcription (<=720p) to save time/bandwidth
            ydl_opts = {
                'outtmpl': output_template,
                'format': 'bestvideo[height<=720]+bestaudio/best[height<=720]',
                'merge_output_format': 'mp4'
            }
            with ytdlp_api.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])
        else:
            # fallback to system yt-dlp binary
            if _shutil.which('yt-dlp') is None:
                with download_lock:
                    fetch_status[vid] = {'status': 'Error', 'path': None, 'error': "'yt-dlp' not found. Install the 'yt-dlp' package or ensure yt-dlp is on PATH."}
                return
            # Choose a moderate resolution (<=720p) to speed up downloads for transcription
            format_spec = 'bestvideo[height<=720]+bestaudio/best[height<=720]'
            cmd = ['yt-dlp', '-f', format_spec, '--merge-output-format', 'mp4', '-o', output_template, url]
            subprocess.run(cmd, check=True)

        # locate file
        matches = glob.glob(os.path.join('./save', f"{vid}.*"))
        if matches:
            # Prefer common video/audio extensions and avoid picking subtitle files (.vtt/.srt/.ass)
            preferred_exts = ['.mp4', '.mkv', '.webm', '.mov', '.mp3', '.m4a', '.wav', '.aac', '.flac', '.ogg', '.opus']
            chosen = None
            # try preferred extensions first
            for ext in preferred_exts:
                for m in matches:
                    if m.lower().endswith(ext):
                        chosen = m
                        break
                if chosen:
                    break

            # if no preferred extension found, pick first non-subtitle file
            if not chosen:
                for m in matches:
                    if not m.lower().endswith(('.vtt', '.srt', '.ass', '.sub')):
                        chosen = m
                        break

            # fallback to first match (shouldn't normally happen)
            if not chosen:
                chosen = matches[0]

            path = chosen
            with download_lock:
                fetch_status[vid] = {'status': 'Ready', 'path': path, 'error': None}
        else:
            with download_lock:
                fetch_status[vid] = {'status': 'Error', 'path': None, 'error': 'File not found after download.'}
    except Exception as e:
        with download_lock:
            fetch_status[vid] = {'status': 'Error', 'path': None, 'error': str(e)}


@app.route('/fetch/status')
def fetch_status_endpoint():
    video_id = request.args.get('video_id')
    if not video_id:
        return jsonify({'error': 'video_id is required'}), 400
    with download_lock:
        info = fetch_status.get(video_id)
    if not info:
        return jsonify({'error': 'video_id not found'}), 404
    return jsonify(info)


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
@app.route('/extract_async', methods=['POST'])
def extract_subtitles_async():
    data = request.get_json()
    video_path = data.get('video_path')
    model_name = data.get('model', 'tiny')
    segment_time = int(data.get('segment_time', 300))  # seconds per chunk
    use_faster = bool(data.get('use_faster', False))
    language = data.get('language')  # optional language code like 'en', 'zh'

    if not video_path:
        return jsonify({'error': 'Video file path is required'}), 400
    if model_name not in AVAILABLE_MODELS:
        return jsonify({'error': f'Invalid model name.'}), 400
    video_path = unquote(video_path)
    if not os.path.exists(video_path):
        return jsonify({'error': 'Video file not found'}), 400
    # If a VTT for this video already exists in ./save, skip re-generating and return a completed job
    base = os.path.splitext(os.path.basename(video_path))[0]
    existing_vtt = os.path.join('./save', f'{base}.vtt')
    if os.path.exists(existing_vtt):
        try:
            with open(existing_vtt, 'r', encoding='utf-8') as f:
                content = f.read()
        except Exception:
            content = None
        job_id = str(uuid.uuid4())
        with extract_lock:
            extract_jobs[job_id] = {
                'state': 'done',
                'processed_chunks': 0,
                'total_chunks': 0,
                'percent': 100,
                'message': 'Existing subtitles found; skipped generation',
                'subtitles_path': existing_vtt,
                'vtt_content': content
            }
        return jsonify({'job_id': job_id}), 202

    job_id = str(uuid.uuid4())
    with extract_lock:
        extract_jobs[job_id] = {'state': 'queued', 'processed_chunks': 0, 'total_chunks': None, 'percent': 0, 'message': 'Queued', 'subtitles_path': None, 'vtt_content': None}

    thread = Thread(target=extract_job_worker, args=(job_id, video_path, model_name, segment_time, use_faster, language))
    thread.start()
    return jsonify({'job_id': job_id}), 202


@app.route('/extract/status')
def extract_status():
    job_id = request.args.get('job_id')
    if not job_id:
        return jsonify({'error': 'job_id is required'}), 400
    with extract_lock:
        info = extract_jobs.get(job_id)
    if not info:
        return jsonify({'error': 'job_id not found'}), 404
    return jsonify(info)


def extract_job_worker(job_id, video_path, model_name, segment_time, use_faster=False, language=None):
    """Worker: slice audio, transcribe chunks sequentially, merge segments into VTT, update extract_jobs."""
    try:
        with extract_lock:
            extract_jobs[job_id]['state'] = 'running'
            extract_jobs[job_id]['message'] = 'Extracting audio and creating chunks'

        base = os.path.splitext(os.path.basename(video_path))[0]
        chunks_dir = os.path.join('./save', f'{job_id}_chunks')
        os.makedirs(chunks_dir, exist_ok=True)
        chunk_template = os.path.join(chunks_dir, 'chunk%03d.wav')
        # Estimate total chunks from duration using ffprobe so we can show progress during segmentation
        try:
            ffprobe_cmd = ['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'default=noprint_wrappers=1:nokey=1', video_path]
            res = subprocess.run(ffprobe_cmd, capture_output=True, text=True, check=True)
            duration = float(res.stdout.strip())
            estimated_total = max(1, int((duration + segment_time - 1) // segment_time))
        except Exception:
            estimated_total = None

        with extract_lock:
            extract_jobs[job_id]['total_chunks'] = estimated_total
            extract_jobs[job_id]['message'] = f'Creating chunks (expected {estimated_total})' if estimated_total else 'Creating chunks'

        # Use ffmpeg to create chunks of fixed duration
        cmd = ['ffmpeg', '-i', video_path, '-vn', '-ar', '44100', '-ac', '2', '-f', 'segment', '-segment_time', str(segment_time), '-reset_timestamps', '1', chunk_template]
        subprocess.run(cmd, check=True)

        # collect chunks
        chunks = sorted(glob.glob(os.path.join(chunks_dir, 'chunk*.wav')))
        total = len(chunks)
        if total == 0:
            raise RuntimeError('No audio chunks produced')

        with extract_lock:
            extract_jobs[job_id]['total_chunks'] = total
            extract_jobs[job_id]['message'] = f'{total} chunks created'

        all_segments = []
        # Prefer faster-whisper when requested and available
        if use_faster and FASTER_AVAILABLE:
            # faster-whisper: create model instance and transcribe chunks
            fw_model = FasterWhisperModel(model_name, device=DEVICE, compute_type="float16" if DEVICE.startswith('cuda') else "int8")
            for idx, chunk in enumerate(chunks):
                with extract_lock:
                    extract_jobs[job_id]['message'] = f'Transcribing chunk {idx+1}/{total} (faster-whisper)'

                # faster-whisper expects language code or None for auto-detect
                fw_lang = language if language else None
                segments, info = fw_model.transcribe(chunk, beam_size=5, language=fw_lang, word_timestamps=False)
                offset = idx * segment_time

                # build VTT fragment for this chunk and append to partial content
                chunk_vtt = ''
                for seg in segments:
                    # faster-whisper may return Segment objects, dicts, or tuples; handle all
                    if isinstance(seg, (list, tuple)):
                        try:
                            start, end, text = seg
                        except Exception:
                            # fallback to string repr
                            start = getattr(seg, 0, 0)
                            end = getattr(seg, 1, 0)
                            text = str(seg)
                    elif isinstance(seg, dict):
                        start = seg.get('start') or seg.get('start_time') or 0
                        end = seg.get('end') or seg.get('end_time') or 0
                        text = seg.get('text') or seg.get('content') or ''
                    else:
                        # object with attributes
                        start = getattr(seg, 'start', getattr(seg, 'start_time', 0))
                        end = getattr(seg, 'end', getattr(seg, 'end_time', 0))
                        text = getattr(seg, 'text', getattr(seg, 'content', ''))

                    # ensure numeric
                    try:
                        seg_start = float(start) + offset
                    except Exception:
                        seg_start = offset
                    try:
                        seg_end = float(end) + offset
                    except Exception:
                        seg_end = seg_start + 0.01

                    all_segments.append({'start': seg_start, 'end': seg_end, 'text': text})
                    # number will be filled when merging; use placeholder
                    chunk_vtt += f"{seg_start}\n{format_time(seg_start)} --> {format_time(seg_end)}\n{text}\n\n"
                with extract_lock:
                    prev = extract_jobs[job_id].get('vtt_partial', '')
                    extract_jobs[job_id]['vtt_partial'] = prev + chunk_vtt

                # mark this chunk as processed
                with extract_lock:
                    extract_jobs[job_id]['processed_chunks'] = idx + 1
                    extract_jobs[job_id]['percent'] = int(((idx + 1) / total) * 100)
        else:
            # fallback to openai-whisper
            try:
                model = whisper.load_model(model_name, device=DEVICE)
            except TypeError:
                model = whisper.load_model(model_name)
                try:
                    model.to(DEVICE)
                except Exception:
                    pass
            for idx, chunk in enumerate(chunks):
                with extract_lock:
                    extract_jobs[job_id]['message'] = f'Transcribing chunk {idx+1}/{total}'

                res = model.transcribe(chunk, language=language) if language else model.transcribe(chunk)
                # adjust timestamps by offset
                offset = idx * segment_time
                chunk_vtt = ''
                for seg in res.get('segments', []):
                    seg_start = seg['start'] + offset
                    seg_end = seg['end'] + offset
                    all_segments.append({'start': seg_start, 'end': seg_end, 'text': seg['text']})
                    chunk_vtt += f"{seg_start}\n{format_time(seg_start)} --> {format_time(seg_end)}\n{seg['text']}\n\n"

                with extract_lock:
                    prev = extract_jobs[job_id].get('vtt_partial', '')
                    extract_jobs[job_id]['vtt_partial'] = prev + chunk_vtt

            # mark this chunk as processed
            with extract_lock:
                extract_jobs[job_id]['processed_chunks'] = idx + 1
                extract_jobs[job_id]['percent'] = int(((idx + 1) / total) * 100)

        # build VTT
        vtt_content = 'WEBVTT\n\n'
        for i, seg in enumerate(all_segments, start=1):
            start = format_time(seg['start'])
            end = format_time(seg['end'])
            text = seg['text'].strip()
            vtt_content += f"{i}\n{start} --> {end}\n{text}\n\n"

        subtitles_filename = f'{base}.vtt'
        subtitles_path = os.path.join('./save', subtitles_filename)
        with open(subtitles_path, 'w', encoding='utf-8') as f:
            f.write(vtt_content)

        # cleanup chunks
        try:
            for c in chunks:
                os.remove(c)
            os.rmdir(chunks_dir)
        except Exception:
            pass

        with extract_lock:
            extract_jobs[job_id]['state'] = 'done'
            extract_jobs[job_id]['processed_chunks'] = total
            extract_jobs[job_id]['percent'] = 100
            extract_jobs[job_id]['message'] = 'Completed'
            extract_jobs[job_id]['subtitles_path'] = subtitles_path
            extract_jobs[job_id]['vtt_content'] = vtt_content
    except Exception as e:
        with extract_lock:
            extract_jobs[job_id]['state'] = 'error'
            extract_jobs[job_id]['message'] = str(e)
            extract_jobs[job_id]['percent'] = extract_jobs[job_id].get('percent', 0)

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
