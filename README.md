# AI Subtitle Studio

A professional-grade, local, web-based tool for generating, translating, and editing subtitles for videos using state-of-the-art AI models.

## Introduction

AI Subtitle Studio is a standalone tool designed to streamline the subtitle creation process for content creators. It leverages extensive AI capabilities to:
- **Transcribe** audio from videos using OpenAI's **Whisper** (or Faster-Whisper) models.
- **Translate** subtitles into multiple languages using **Helsinki-NLP** models.
- **Refine** and polish text using **Qwen** large language models (optional).
- **Edit** subtitles in a modern, Google-style web interface with real-time video seeking.

All processing happens locally on your machine, ensuring privacy and zero data leakage.

## Tech Stack

This project is built using the following technologies:

- **Backend / Web Framework**: Python, Flask, Flask-SocketIO
- **AI / ML**:
    - **ASR**: `openai-whisper`, `faster-whisper`
    - **Translation**: `transformers` (Helsinki-NLP/OPUS-MT)
    - **LLM Refinement**: `transformers` (Qwen/Qwen3-4B)
    - **Deep Learning Framework**: `PyTorch` (GPU accelerated)
- **Utilities**: `ffmpeg` (Audio processing), `yt-dlp` (Video downloading)
- **Frontend**: HTML5, Tailwind CSS, Google Material Design (Custom CSS), JavaScript (Vanilla)

## Installation

### Prerequisites
- Python 3.10+
- FFmpeg installed and added to system PATH.
- (Optional) NVIDIA GPU with drivers installed for acceleration.

### Manual Setup (Conda/Pip)

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-repo/AI-Subtitle-Studio.git
    cd AI-Subtitle-Studio
    ```

2.  **Create a Conda environment (Recommended):**
    ```bash
    conda create -n ai-subtitle python=3.10
    conda activate ai-subtitle
    ```

3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *Note: Ensure you install the GPU version of PyTorch if you have a compatible NVIDIA card: [PyTorch Get Started](https://pytorch.org/get-started/locally/)*

## Usage

### Running Locally

1.  **Start the Server:**
    ```bash
    # chmod +x run.sh
    ./run.sh
    ```
    Or manually:
    ```bash
    python src/ai_subtitle_generator/web/server.py
    ```

2.  **Access the Interface:**
    Open your browser and navigate to: `http://localhost:7860`

3.  **Generate Subtitles:**
    - Paste a YouTube URL or upload a local video file.
    - Select your Model (e.g., `medium`, `large-v2`).
    - Choose Display Mode (Original/Translated/Bilingual).
    - Click "Generate Subtitles".

## Docker Support (Optimized)
 
 This application is packaged with a lightweight, GPU-ready Docker image based on `python:3.10-slim`.
 
 ### Prerequisites
 - Docker Engine
 - NVIDIA Container Toolkit (for GPU acceleration)
 
 ### Build & Run
 
 1.  **Build the Image:**
     ```bash
     docker build -t ai-subtitle-studio .
     ```
     *Note: The first build handles large AI dependencies (Torch, CUDA libs) so it may take some time depending on your network.*
 
 2.  **Run with GPU (Recommended):**
     ```bash
     docker run --gpus all -d -p 7860:7860 --name subtitle-studio ai-subtitle-studio
     ```
 
 3.  **Run on CPU (Optional):**
     If you don't have a GPU, you can run it in CPU mode (slower):
     ```bash
     docker run -d -p 7860:7860 --name subtitle-studio ai-subtitle-studio
     ```
 
 4.  **Access:**
     Open `http://localhost:7860` in your browser.

## Implementation Details

The core logic is modularized within the `src/ai_subtitle_generator` package:

### 1. Audio Transcription (`transcriber.py`)
- **Class**: `SubtitleGenerator`
- **Method**: `generate_subtitles`
- **Logic**:
    - Extracts audio from the input video using `ffmpeg`.
    - Slices audio into manageable segments (default 300s) to avoid memory issues.
    - Loads the Whisper model (either `openai-whisper` or `faster-whisper`).
    - Iterates through chunks, transcribing speech to text with timestamps.
    - Merges segment results into a unified WebVTT format.

### 2. Machine Translation (`translator.py`)
- **Class**: `SubtitleTranslator`
- **Method**: `translate_text` / `refine_subtitle`
- **Logic**:
    - Uses `Helsinki-NLP` models via Hugging Face `transformers` pipelines for efficient translation between specific language pairs (e.g., En-Zh).
    - Supports **LLM Refinement** using `Qwen/Qwen3-4B`: It constructs prompts tailored for subtitle correction (removing ASR errors, fixing grammar) and feeds the text to the local LLM for polish.

### 3. Video Downloading (`downloader.py`)
- **Class**: `VideoDownloader`
- **Logic**:
    - Wraps `yt-dlp` to fetch videos from YouTube or other supported platforms.
    - Handles caching locally to prevent re-downloading the same content.
    - Returns a distinct local filepath for processing.

### 4. GPU Optimization (`gpu_detector.py`)
- Automatically detects available hardware (`CUDA`, `MPS` for Mac, or `CPU`).
- Optimizes memory usage by loading models only when needed and using thread locks (`Lock`) for model access to prevent race conditions during concurrent requests.
