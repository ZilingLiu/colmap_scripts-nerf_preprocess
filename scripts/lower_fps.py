from moviepy.editor import VideoFileClip
import os
video_dir = "/hhd2/home/Code/lzl_gen/nerf_code/mingqi_videos"
videos = os.listdir(video_dir)
new_fps = 24
for video in videos:
    clip = VideoFileClip(os.path.join(video_dir, video))
    clip.write_videofile(os.path.join(video_dir, "new_" + video), fps=new_fps)

