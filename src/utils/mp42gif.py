from moviepy.editor import *

def make_gif(video_dir: str, gif_file_dir: str):
    video = VideoFileClip(video_dir)
    video.write_gif(gif_file_dir)