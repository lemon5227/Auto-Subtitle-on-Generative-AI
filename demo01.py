import whisper
import srt
model = whisper.load_model("tiny")
result = model.transcribe("output.mp3",language='en')

whisper_dict = result
segments  = whisper_dict['segments']

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
with open("output.srt", "w") as f:
    f.write(srt_content)
