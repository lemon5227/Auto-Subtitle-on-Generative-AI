from flask import Flask, request, jsonify, send_file
from flask_socketio import SocketIO, emit
from urllib.parse import unquote
import subprocess
import shutil as _shutil
import sys
import whisper
import os
import time
import shutil
import torch
import numpy as np
import threading
import queue
from transformers import pipeline
from huggingface_hub import snapshot_download
from threading import Thread, Lock
import shlex
import glob
import uuid
import base64
# Try to import python API of yt-dlp; if not available we'll fall back to system binary at runtime
try:
    import yt_dlp as ytdlp_api
except Exception:
    ytdlp_api = None

# Traditional to Simplified Chinese conversion
try:
    import opencc
    converter = opencc.OpenCC('t2s')  # Traditional to Simplified
    OPENCC_AVAILABLE = True
except Exception:
    converter = None
    OPENCC_AVAILABLE = False

# Optional faster-whisper support
try:
    from faster_whisper import WhisperModel as FasterWhisperModel
    FASTER_AVAILABLE = True
except Exception:
    FasterWhisperModel = None
    FASTER_AVAILABLE = False

# Optional distil-whisper support
try:
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
    DISTIL_AVAILABLE = True
except Exception:
    AutoModelForSpeechSeq2Seq = None
    AutoProcessor = None
    DISTIL_AVAILABLE = False

# Optional SenseVoice support
try:
    from transformers import AutoModel, AutoTokenizer
    # Also try FunASR if available for better SenseVoice support
    try:
        from funasr import AutoModel as FunASRAutoModel
        FUNASR_AVAILABLE = True
    except Exception:
        FUNASR_AVAILABLE = False
    SENSEVOICE_AVAILABLE = True
except Exception:
    SENSEVOICE_AVAILABLE = False
    FUNASR_AVAILABLE = False

app = Flask(__name__, static_folder='./save')
socketio = SocketIO(app, cors_allowed_origins="*")

# Global variables for real-time transcription
realtime_models = {}
realtime_audio_queues = {}
realtime_threads = {}
realtime_locks = {}

# Startup checks: ensure ffmpeg is available and warn/exit if not
def check_ffmpeg():
    """Check if ffmpeg is available and provide platform-specific installation instructions"""
    if _shutil.which('ffmpeg') is None:
        print("ERROR: 'ffmpeg' not found in PATH.")
        print("\n请根据您的系统安装 ffmpeg:")
        print("  Ubuntu/Debian: sudo apt install ffmpeg")
        print("  macOS (Homebrew): brew install ffmpeg")
        print("  macOS (MacPorts): sudo port install ffmpeg")
        print("  Conda: conda install ffmpeg")
        print("  Windows: 下载从 https://ffmpeg.org/ 并添加到 PATH")
        sys.exit(1)

check_ffmpeg()

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

# 智能GPU检测和设备选择系统
try:
    from gpu_detector import get_optimal_device, create_device_environment, GPUDetector
    
    # 应用设备优化环境变量
    device_env = create_device_environment()
    for key, value in device_env.items():
        os.environ[key] = value
    
    # 获取最佳设备并应用安全检查
    DEVICE, device_info = get_optimal_device()
    
    # GPU可用性验证
    gpu_validated = False
    if DEVICE == 'cuda':
        try:
            # 验证CUDA可用性
            test_tensor = torch.randn(10, 10).cuda()
            del test_tensor
            torch.cuda.empty_cache()
            gpu_validated = True
        except Exception as e:
            print(f"⚠️ CUDA验证失败: {e}")
            DEVICE = 'cpu'
    elif DEVICE == 'mps':
        try:
            # 验证MPS可用性
            test_tensor = torch.randn(10, 10).to('mps')
            del test_tensor
            gpu_validated = True
        except Exception as e:
            print(f"⚠️ MPS验证失败: {e}")
            DEVICE = 'cpu'
    
    # 打印设备信息
    print("🚀 智能GPU检测结果:")
    print("=" * 50)
    detector = GPUDetector()
    print(f"🎯 选择设备: {DEVICE}")
    print(f"📊 设备信息: {detector.get_device_summary()}")
    print(f"⚡ 性能等级: {device_info['performance_level']}")
    
    if gpu_validated and DEVICE != 'cpu':
        print("✅ GPU验证通过，将使用硬件加速")
    elif DEVICE == 'cpu':
        print("🔵 使用CPU模式，推荐选择较小的模型")
    
    if device_info['optimization_tips']:
        print("💡 优化建议:")
        for tip in device_info['optimization_tips'][:2]:  # 只显示前2个建议
            print(f"   • {tip}")
    print("=" * 50)
    
except ImportError as e:
    # 后备方案：使用原有的简单检测
    print("⚠️ GPU检测模块导入失败，使用基础检测")
    def get_device():
        """Get the best available device for inference"""
        if torch.cuda.is_available():
            return 'cuda'
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return 'mps'
        else:
            return 'cpu'
    
    DEVICE = get_device()
    print(f"使用设备: {DEVICE}")

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

# --- Real-time Transcription Functions ---

def get_realtime_model(model_name, language='zh'):
    """Get or load a real-time transcription model"""
    key = f"realtime_{model_name}_{language}"
    if key not in realtime_models:
        try:
            print(f"Loading real-time model: {model_name} for language: {language}")
            if model_name == 'sensevoice':
                # Use SenseVoice for Chinese
                if SENSEVOICE_AVAILABLE and language == 'zh':
                    device = get_device()
                    model_id = "FunAudioLLM/SenseVoiceSmall"
                    model_loaded = False
                    
                    # Debug FunASR availability
                    print(f"FUNASR_AVAILABLE: {FUNASR_AVAILABLE}")
                    
                    # Try different SenseVoice model identifiers for FunASR
                    if FUNASR_AVAILABLE:
                        sensevoice_models = [
                            "iic/SenseVoiceSmall",  # Try original
                            "damo/speech_sensevoice_asr_nat-zh_en-16k-common-vocab8404",  # Full ModelScope name
                            "damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch",  # Paraformer as backup
                        ]
                        
                        for sv_model in sensevoice_models:
                            try:
                                print(f"Loading SenseVoice with FunASR: {sv_model}")
                                realtime_models[key] = FunASRAutoModel(
                                    model=sv_model,
                                    device=device,
                                    disable_update=True,
                                    hub="ms"  # Force ModelScope hub
                                )
                                realtime_models[f"{key}_type"] = "funasr"
                                print(f"SenseVoice loaded successfully with FunASR: {sv_model}")
                                model_loaded = True
                                break
                            except Exception as e:
                                print(f"FunASR loading failed for {sv_model}: {e}")
                                continue
                    else:
                        print("FunASR not available, skipping FunASR loading attempt")
                    
                    # Fallback to transformers
                    if not model_loaded:
                        try:
                            print(f"Loading SenseVoice with transformers: {model_id}")
                            realtime_models[key] = AutoModel.from_pretrained(
                                model_id, 
                                trust_remote_code=True, 
                                torch_dtype=torch.float32,
                                device_map=device
                            )
                            realtime_models[f"{key}_type"] = "transformers"
                            print(f"SenseVoice loaded successfully with transformers: {model_id}")
                            model_loaded = True
                        except Exception as e:
                            print(f"Transformers loading failed: {e}")
                    
                    if not model_loaded:
                        print("All SenseVoice model paths failed, falling back to Whisper")
                        # Fallback to whisper small for Chinese
                        realtime_models[key] = whisper.load_model("small")
                        try:
                            realtime_models[key].to(DEVICE)
                        except Exception:
                            pass
                        print(f"Fallback to Whisper small for Chinese: {key}")
                else:
                    raise Exception("SenseVoice only available for Chinese language")
            elif model_name == 'large-v3-turbo':
                # Use Whisper Large-v3 Turbo with optimized pipeline
                if DISTIL_AVAILABLE:  # We use the same transformers library
                    device = get_device()
                    # 使用 float16 for CUDA/MPS，float32 for CPU
                    torch_dtype = torch.float16 if device in ['cuda', 'mps'] else torch.float32
                    
                    print(f"Loading Whisper Large-v3 Turbo with chunked algorithm optimization...")
                    from transformers import pipeline
                    
                    # Create pipeline with chunked algorithm optimization
                    pipe_kwargs = {
                        "model": "openai/whisper-large-v3-turbo",
                        "dtype": torch_dtype,  # 使用 dtype 替代 torch_dtype
                        "device": device,
                        "chunk_length_s": 30,  # 30秒分块，官方推荐
                        "batch_size": 8 if device in ['cuda', 'mps'] else 2,  # 批处理优化
                        "ignore_warning": True,  # 忽略分块实验性功能的警告
                    }
                    
                    # 不使用 flash attention，避免依赖问题
                    # 如果需要更快的速度，可以手动安装 flash-attn 包
                    
                    try:
                        realtime_models[key] = pipeline("automatic-speech-recognition", **pipe_kwargs)
                        realtime_models[f"{key}_type"] = "turbo_pipeline"
                        print(f"Whisper Large-v3 Turbo loaded successfully with chunked algorithm")
                    except Exception as e:
                        print(f"Failed to load Whisper Large-v3 Turbo: {e}")
                        raise Exception(f"Whisper Large-v3 Turbo not available: {e}")
                else:
                    raise Exception("Transformers not available for Whisper Large-v3 Turbo")
            elif model_name.startswith('distil-'):
                # Use distil-whisper with consistent dtype
                if DISTIL_AVAILABLE:
                    device = get_device()
                    # Force float32 to avoid dtype mismatch (MPS doesn't support float16 for some models)
                    torch_dtype = torch.float32
                    model_id = f"distil-whisper/{model_name}"
                    try:
                        realtime_models[key] = AutoModelForSpeechSeq2Seq.from_pretrained(
                            model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
                        )
                        realtime_models[key].to(device)
                        # Also load processor
                        realtime_models[f"{key}_processor"] = AutoProcessor.from_pretrained(model_id)
                    except Exception as e:
                        print(f"Failed to load distil-whisper model {model_id}: {e}")
                        raise Exception(f"Distil-Whisper model {model_id} not available. Try distil-small.en or use standard Whisper models.")
                else:
                    raise Exception("Transformers not available for Distil-Whisper")
            elif FASTER_AVAILABLE:
                # Use faster-whisper for real-time transcription (CUDA only)
                device = "cuda" if torch.cuda.is_available() else "cpu"  # faster-whisper 不支持 MPS
                compute_type = "float16" if device == "cuda" else "int8"
                realtime_models[key] = FasterWhisperModel(
                    model_name,
                    device=device,
                    compute_type=compute_type
                    # Removed language parameter to let it auto-detect and avoid Traditional Chinese
                )
            else:
                # Fallback to regular whisper
                realtime_models[key] = whisper.load_model(model_name)
                try:
                    realtime_models[key].to(DEVICE)
                except Exception:
                    pass
            print(f"Real-time model loaded: {key}")
        except Exception as e:
            print(f"Failed to load real-time model {key}: {e}")
            return None
    return realtime_models[key]


def resample_audio(audio_array: np.ndarray, src_rate: int, target_rate: int = 16000) -> np.ndarray:
    """Resample a mono float32 audio array to the target sample rate."""
    if audio_array.size == 0:
        return audio_array.astype(np.float32, copy=False)

    if src_rate == target_rate:
        return audio_array.astype(np.float32, copy=False)

    src_rate = float(src_rate)
    target_rate = float(target_rate)

    target_length = max(1, int(round(audio_array.shape[0] * target_rate / src_rate)))
    if target_length == audio_array.shape[0]:
        return audio_array.astype(np.float32, copy=False)

    x_old = np.linspace(0.0, 1.0, num=audio_array.shape[0], endpoint=False)
    x_new = np.linspace(0.0, 1.0, num=target_length, endpoint=False)
    resampled = np.interp(x_new, x_old, audio_array.astype(np.float32, copy=False))
    return resampled.astype(np.float32, copy=False)

def process_realtime_audio(sid, audio_queue, model, language, model_name):
    """Process audio chunks for real-time transcription"""
    buffer = []
    buffer_duration = 0
    min_chunk_duration = 3.0  # Process every 3 seconds of audio

    try:
        while True:
            # Get audio chunk from queue
            try:
                audio_chunk = audio_queue.get(timeout=1.0)
                if audio_chunk is None:  # Stop signal
                    break
            except queue.Empty:
                continue

            # Normalize chunk into float32 audio data and sample rate metadata
            if isinstance(audio_chunk, np.ndarray):
                audio_data = audio_chunk.astype(np.float32, copy=False)
                sample_rate = 16000
            elif isinstance(audio_chunk, tuple) and len(audio_chunk) == 2:
                raw_audio, sample_rate = audio_chunk
                audio_data = np.frombuffer(raw_audio, dtype=np.int16).astype(np.float32) / 32768.0
            else:
                sample_rate = 16000
                audio_data = np.frombuffer(audio_chunk, dtype=np.int16).astype(np.float32) / 32768.0

            # Add to buffer
            buffer.extend(audio_data.tolist())
            buffer_duration += len(audio_data) / float(sample_rate or 16000.0)

            # Process when we have enough audio
            if buffer_duration >= min_chunk_duration:
                try:
                    buffer_array = np.array(buffer, dtype=np.float32)
                    # Transcribe the buffered audio
                    if model_name == 'sensevoice':
                        # Use SenseVoice or fallback to Whisper
                        try:
                            model_type = realtime_models.get(f"realtime_{model_name}_{language}_type", "unknown")
                            
                            if model_type == "funasr":
                                # FunASR SenseVoice
                                with torch.no_grad():
                                    result = model.generate(
                                        input=buffer_array,
                                        cache={},
                                        language="auto",
                                        use_itn=True
                                    )
                                # Extract text from FunASR result
                                if isinstance(result, list) and len(result) > 0:
                                    transcription = result[0].get("text", "") if isinstance(result[0], dict) else str(result[0])
                                else:
                                    transcription = str(result) if result else ""
                                    
                            elif model_type == "transformers" and hasattr(model, 'inference'):
                                # Transformers SenseVoice
                                with torch.no_grad():
                                    result = model.inference(
                                        data_in=buffer_array,
                                        language="auto",
                                        use_itn=True,
                                    )
                                transcription = result[0][0]['text'] if result and len(result) > 0 and len(result[0]) > 0 else ""
                            else:
                                # Whisper fallback
                                result = model.transcribe(buffer_array, language=language)
                                transcription = result["text"]
                        except Exception as e:
                            print(f"SenseVoice inference error: {e}")
                            transcription = ""
                    elif model_name == 'large-v3-turbo':
                        # Use Whisper Large-v3 Turbo with chunked algorithm
                        model_type = realtime_models.get(f"realtime_{model_name}_{language}_type", "unknown")
                        if model_type == "turbo_pipeline":
                            try:
                                # 使用分块算法进行快速转录
                                result = model(
                                    buffer_array,
                                    chunk_length_s=30,  # 30秒分块
                                    batch_size=8 if get_device() in ['cuda', 'mps'] else 2,
                                    return_timestamps=False  # 实时转录不需要时间戳
                                )
                                transcription = result.get("text", "") if isinstance(result, dict) else ""
                            except Exception as e:
                                print(f"Turbo pipeline error: {e}")
                                transcription = ""
                        else:
                            transcription = ""
                    elif model_name.startswith('distil-'):
                        # Use distil-whisper
                        processor = realtime_models.get(f"realtime_{model_name}_{language}_processor")
                        if processor:
                            # Ensure consistent dtype for distil-whisper
                            inputs = processor(buffer_array, sampling_rate=16000, return_tensors="pt")
                            # Convert inputs to float32 to match model dtype
                            inputs = {k: v.to(model.device).float() if v.dtype == torch.float16 else v.to(model.device) for k, v in inputs.items()}
                            with torch.no_grad():
                                # Simple generation without custom config
                                generated_ids = model.generate(**inputs, max_length=448)
                            transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
                        else:
                            transcription = ""
                    elif FASTER_AVAILABLE and isinstance(model, FasterWhisperModel):
                        segments, _ = model.transcribe(
                            buffer_array,
                            language=language,
                            beam_size=5,
                            vad_filter=True,
                            vad_parameters=dict(threshold=0.5, min_speech_duration_ms=250)
                        )
                        transcription = ""
                        for segment in segments:
                            transcription += segment.text + " "
                    else:
                        # Regular whisper
                        result = model.transcribe(buffer_array, language=language)
                        transcription = result["text"]

                    if transcription.strip():
                        # Convert traditional Chinese to simplified if available
                        final_text = transcription.strip()
                        if OPENCC_AVAILABLE and language == 'zh':
                            try:
                                final_text = converter.convert(final_text)
                            except Exception as e:
                                print(f"OpenCC conversion failed: {e}")
                        
                        # Send transcription to client
                        socketio.emit('transcription', {
                            'text': final_text,
                            'timestamp': int(time.time() * 1000)
                        }, room=sid)

                    # Clear buffer
                    buffer = []
                    buffer_duration = 0

                except Exception as e:
                    print(f"Transcription error for {sid}: {e}")
                    socketio.emit('error', {'message': f'Transcription failed: {str(e)}'}, room=sid)

    except Exception as e:
        print(f"Real-time processing error for {sid}: {e}")
    finally:
        # Cleanup
        lock = realtime_locks.get(sid)
        if lock:
            with lock:
                realtime_audio_queues.pop(sid, None)
                realtime_threads.pop(sid, None)
        else:
            realtime_audio_queues.pop(sid, None)
            realtime_threads.pop(sid, None)

        realtime_locks.pop(sid, None)
        socketio.emit('transcription_stopped', {'status': 'stopped'}, room=sid)

# WebSocket event handlers
@socketio.on('connect')
def handle_connect():
    print(f"Client connected: {request.sid}")
    socketio.emit('connected', {'status': 'success'})

@socketio.on('disconnect')
def handle_disconnect():
    print(f"Client disconnected: {request.sid}")
    sid = request.sid

    # Stop real-time transcription for this client
    lock = realtime_locks.get(sid)
    if lock is None:
        lock = threading.Lock()
        realtime_locks[sid] = lock

    with lock:
        if sid in realtime_audio_queues:
            try:
                realtime_audio_queues[sid].put(None)
            except Exception:
                pass
        if sid in realtime_threads:
            realtime_threads[sid].join(timeout=2.0)
            realtime_threads.pop(sid, None)
        realtime_audio_queues.pop(sid, None)

    realtime_locks.pop(sid, None)

@socketio.on('start_transcription')
def handle_start_transcription(data):
    sid = request.sid
    model_name = data.get('model', 'base')
    language = data.get('language', 'zh')

    print(f"Starting real-time transcription for {sid}: model={model_name}, language={language}")

    try:
        # Get or load the model
        model = get_realtime_model(model_name, language)
        if not model:
            message = 'Failed to load transcription model'
            socketio.emit('error', {'message': message}, room=sid)
            return {'status': 'error', 'message': message}

        # Initialize audio queue and processing thread
        if sid not in realtime_locks:
            realtime_locks[sid] = threading.Lock()
        
        with realtime_locks[sid]:
            # Clean up existing queue/thread if present
            if sid in realtime_audio_queues:
                try:
                    realtime_audio_queues[sid].put_nowait(None)
                except Exception:
                    pass
            if sid in realtime_threads:
                realtime_threads[sid].join(timeout=3.0)  # Increased timeout
                realtime_threads.pop(sid, None)

            realtime_audio_queues[sid] = queue.Queue()
            worker = threading.Thread(
                target=process_realtime_audio,
                args=(sid, realtime_audio_queues[sid], model, language, model_name),
                daemon=True
            )
            realtime_threads[sid] = worker
            worker.start()

        socketio.emit('transcription_started', {'status': 'success'}, room=sid)
        return {'status': 'success', 'message': 'Transcription started successfully.'}

    except Exception as e:
        print(f"Failed to start transcription for {sid}: {e}")
        message = f'Failed to start transcription: {str(e)}'
        socketio.emit('error', {'message': message}, room=sid)
        return {'status': 'error', 'message': message}

@socketio.on('audio_chunk')
def handle_audio_chunk(data):
    sid = request.sid
    chunk = data.get('audio')
    sample_rate = data.get('sampleRate', 16000)

    if not chunk:
        return {'status': 'error', 'message': 'No audio chunk provided'}

    if sid not in realtime_audio_queues:
        message = 'Transcription not started for this session'
        socketio.emit('error', {'message': message}, room=sid)
        return {'status': 'error', 'message': message}

    try:
        # Decode the base64-encoded PCM chunk and convert to float32
        audio_bytes = base64.b64decode(chunk)
        int_samples = np.frombuffer(audio_bytes, dtype=np.int16)
        float_samples = int_samples.astype(np.float32) / 32768.0

        if not sample_rate or sample_rate <= 0:
            sample_rate = 16000

        processed_chunk = resample_audio(float_samples, int(sample_rate), 16000)

        lock = realtime_locks.get(sid)
        if not lock:
            lock = threading.Lock()
            realtime_locks[sid] = lock

        with lock:
            audio_queue = realtime_audio_queues.get(sid)
            if not audio_queue:
                message = 'Audio queue unavailable for this session'
                socketio.emit('error', {'message': message}, room=sid)
                return {'status': 'error', 'message': message}

            audio_queue.put(processed_chunk)

        socketio.emit('chunk_received', {'status': 'success'}, room=sid)
        return {'status': 'success'}
    except Exception as e:
        message = f'Failed to process audio chunk: {str(e)}'
        print(f"Error processing audio chunk for {sid}: {e}")
        socketio.emit('error', {'message': message}, room=sid)
        return {'status': 'error', 'message': message}

@socketio.on('stop_transcription')
def handle_stop_transcription(data=None):
    sid = request.sid
    print(f"Stopping real-time transcription for {sid}")
    response = {'status': 'success', 'message': 'Transcription stopping'}

    try:
        lock = realtime_locks.setdefault(sid, threading.Lock())

        with lock:
            audio_queue = realtime_audio_queues.get(sid)
            if audio_queue:
                try:
                    audio_queue.put_nowait(None)
                except Exception:
                    audio_queue.put(None)

            worker = realtime_threads.get(sid)

        if worker:
            worker.join(timeout=2.0)
            realtime_threads.pop(sid, None)

        socketio.emit('transcription_stopped', {'status': 'success'}, room=sid)

    except Exception as e:
        message = f'Failed to stop transcription: {str(e)}'
        print(message)
        socketio.emit('error', {'message': message}, room=sid)
        response = {'status': 'error', 'message': message}

    return response


# Route to serve the new frontend
@app.route('/')
def new_index():
    return send_file(os.path.join(os.getcwd(), 'app.html'))

# Route to serve app.html directly
@app.route('/app.html')
def app_page():
    return send_file(os.path.join(os.getcwd(), 'app.html'))

# Route to serve the real-time transcription page
@app.route('/realtime.html')
def realtime_page():
    return send_file(os.path.join(os.getcwd(), 'realtime.html'))

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
    print("Starting AI Subtitle Generator with real-time transcription support...")
    socketio.run(app, debug=True, port=5001, host='0.0.0.0')
