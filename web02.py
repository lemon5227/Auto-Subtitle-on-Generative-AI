from flask import Flask, request, jsonify, send_file
import subprocess
import whisper
import webvtt
import os

app = Flask(__name__)

# 根路径，发送 index.html 页面
@app.route('/')
def index():
    return send_file(os.path.join(os.getcwd(), 'index.html'))

# 返回合并后的视频文件
@app.route('/save/merged_video.mp4')
def get_video():
    video_path = './save/merged_video.mp4'
    return send_file(video_path, as_attachment=True)

# 上传视频并提取字幕
@app.route('/upload', methods=['POST'])
def upload_video():
    uploaded_file = request.files['file']
    if uploaded_file.filename != '':
        # 保存上传的视频文件
        video_path = './save/' + uploaded_file.filename
        uploaded_file.save(video_path)

        # 使用FFmpeg提取视频音频
        audio_output_path = './save/audio.mp3'
        ffmpeg_command = f"ffmpeg -i {video_path} -vn -ar 44100 -ac 2 -b:a 192k {audio_output_path}"
        subprocess.run(ffmpeg_command, shell=True)

        # 使用Whisper进行音频转文字
        model = whisper.load_model("tiny")
        result = model.transcribe(audio_output_path, language='en')
        segments = result['segments']

        # 生成webvtt字幕文件
        vtt_content = webvtt.WebVTT()
        for seg in segments:
            print(type(seg['start']))
            print(seg['start'])

            print(type(seg['end']))
            print(seg['end'])
            caption = webvtt.Caption(start=seg['start'], end=seg['end'], text=seg['text'])
            vtt_content.captions.append(caption)

        # 保存vtt字幕到文件
        vtt_path = './save/subtitles.vtt'
        vtt_content.save(vtt_path)

        return jsonify({'subtitles_path': vtt_path})
    else:
        return jsonify({'error': '未找到上传的文件'})

if __name__ == '__main__':
    app.run(debug=True)
