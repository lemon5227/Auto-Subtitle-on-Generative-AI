from flask import Flask, request, jsonify, send_file
from urllib.parse import unquote
import subprocess
import whisper
import os
import time  
app = Flask(__name__, static_folder='./save')

# 根路径，发送 index.html 页面
@app.route('/')
def index():
    return send_file(os.path.join(os.getcwd(), 'index.html'))

# 返回字幕文件
@app.route('/save/subtitles.vtt')
def get_subtitles():
    subtitles_path = './save/subtitles.vtt'
    return send_file(subtitles_path, as_attachment=True)

# 返回合并后的视频文件
@app.route('/save/merged_video.mp4')
def get_video():
    video_path = './save/merged_video.mp4'
    return send_file(video_path, as_attachment=True)

# 上传视频并保存原视频文件
@app.route('/upload', methods=['POST'])
def upload_video():
    uploaded_file = request.files['file']
    if uploaded_file.filename != '':
        # 保存上传的视频文件
        video_path = './save/' + uploaded_file.filename
        uploaded_file.save(video_path)

        return jsonify({'original_video_path': video_path})
    else:
        return jsonify({'error': '未找到上传的文件'})

# 提取字幕并生成字幕文件
@app.route('/extract_subtitles')
def extract_subtitles():
    video_path = request.args.get('video_path')
    if video_path:
        # 解码 video_path
        video_path = unquote(video_path)

        # 使用 os.path.join 正确构建视频文件路径
        video_file_path = os.path.join('./save', os.path.basename(video_path))

        # 使用FFmpeg提取视频音频
        audio_output_path = os.path.splitext(video_file_path)[0] + '.mp3'
        ffmpeg_command = f"ffmpeg -i '{video_file_path}' -vn -ar 44100 -ac 2 -b:a 192k '{audio_output_path}'"
        subprocess.run(ffmpeg_command, shell=True)

        # 开始时间戳
        start_time = time.time()

        # 使用Whisper进行音频转文字
        model = whisper.load_model("tiny")
        result = model.transcribe(audio_output_path, language='en')
        segments = result['segments']

        # 结束时间戳
        end_time = time.time()

        # 计算并打印时间差
        elapsed_time = end_time - start_time
        print(f"Whisper 转录时间: {elapsed_time} 秒")

        # 生成 WebVTT 格式的字幕
        vtt_content = "WEBVTT\n\n"
        for i, seg in enumerate(segments, start=1):
            start_time = format_time(seg['start'])
            end_time = format_time(seg['end'])
            text = seg['text']
            vtt_content += f"{i}\n{start_time} --> {end_time}\n{text}\n\n"

        # 保存到文件
        subtitles_path = './save/subtitles.vtt'
        with open(subtitles_path, "w") as f:
            f.write(vtt_content)
        
        # 返回字幕文件路径
        return jsonify({'subtitles_path': subtitles_path})
    else:
        return jsonify({'error': '视频文件路径不能为空'})

# 辅助函数，将时间格式化为 WebVTT 格式
def format_time(seconds):
    hours = int(seconds / 3600)
    minutes = int((seconds % 3600) / 60)
    seconds = seconds % 60
    milliseconds = int((seconds - int(seconds)) * 1000)
    return f"{hours:02}:{minutes:02}:{int(seconds):02}.{milliseconds:03}"

if __name__ == '__main__':
    app.run(debug=True)
