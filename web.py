from flask import Flask, request, jsonify, send_file
import subprocess
import whisper
import srt
import os
app = Flask(__name__)

# 根路径，发送 index.html 页面
@app.route('/')
def index():
    return send_file(os.path.join(os.getcwd(), 'index.html'))

# 返回字幕文件
@app.route('/save/subtitles.srt')
def get_subtitles():
    subtitles_path = './save/subtitles.srt'
    return send_file(subtitles_path, as_attachment=True)

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
        audio_output_path = './save/'+uploaded_file.filename.split('.')[0]+'.mp3'
        ffmpeg_command = f"ffmpeg -i '{video_path}' -vn -ar 44100 -ac 2 -b:a 192k '{audio_output_path}'"
        subprocess.run(ffmpeg_command, shell=True)

        # 使用Whisper进行音频转文字
        model = whisper.load_model("tiny")
        result = model.transcribe(audio_output_path, language='en')
        segments = result['segments']

        # 转换成字幕
        subs = [
            srt.Subtitle(
                index=i,
                start=srt.timedelta(seconds=seg['start']),
                end=srt.timedelta(seconds=seg['end']),
                content=seg['text'],
            )
            for i, seg in enumerate(segments, start=1)
        ]

        # 转换成字幕文件的格式
        srt_content = srt.compose(subs)

        # 保存到文件
        subtitles_path = './save/subtitles.srt'
        with open(subtitles_path, "w") as f:
            f.write(srt_content)

        # 使用FFmpeg将字幕合并到视频中
        merged_video_path = './save/merged_subtitles'+uploaded_file.filename
        subprocess.run(f"ffmpeg -i '{video_path}' -i '{subtitles_path}' -c:v copy -c:a copy -c:s mov_text -metadata:s:s:0 language=eng -disposition:s:0 default '{merged_video_path}'", shell=True)

        # 返回文件路径
        return jsonify({'merged_video_path': merged_video_path})
    else:
        return jsonify({'error': '未找到上传的文件'})

if __name__ == '__main__':
    app.run(debug=True)
