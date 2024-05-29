import os
import tkinter as tk
from tkinter import filedialog

def play_video(video_path):
    os.system(f'ffplay {video_path}')

def browse_files():
    filename = filedialog.askopenfilename(initialdir="/",
                                          title="Select a video file",
                                          filetypes=(("video files", "*.mp4 *.avi"), ("all files", "*.*")))
    play_video(filename)

window = tk.Tk()
window.title('Video Player')

button_explore = tk.Button(window,
                           text="Browse video files",
                           command=browse_files)
button_explore.pack()

window.mainloop()