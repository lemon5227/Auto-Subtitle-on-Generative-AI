import whisper
model = whisper.load_model("tiny")
result = model.transcribe("output.mp3",language='en')

whisper_dict = result
segments  = whisper_dict['segments']
print(segments)
def convert_to_vtt(segments):
    vtt_content = "WEBVTT\n\n"

    for segment in segments:
        start_time = segment['start']
        end_time = segment['end']
        text = segment['text']

        vtt_content += f"{start_time:.3f} --> {end_time:.3f}\n{text}\n\n"

    return vtt_content

# 使用提取的字幕数据segments，将其转换为VTT格式
vtt_data = convert_to_vtt(segments)

# 将VTT数据保存到文件中
with open("output.vtt", "w", encoding="utf-8") as file:
    file.write(vtt_data)

print("字幕已保存到 output.vtt 文件中。")