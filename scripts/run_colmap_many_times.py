import subprocess
import os
import time
from video2images import video2images
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
videos_path = "/hhd2/home/Code/lzl_gen/nerf_code/retry_videos"
images_path = "/hhd2/home/Code/lzl_gen/nerf_code/retry_images"
videos = os.listdir(videos_path)
# Shell 脚本的路径
script_path = '/hhd2/home/Code/lzl_gen/nerf_code/scripts/local_colmap_and_resize.sh'  # 替换为你的 Shell 脚本路径
log_path = '/hhd2/home/Code/lzl_gen/nerf_code/scripts/log.txt'  # 替换为你的 Shell 脚本路径 
for video in videos:
    video2images(os.path.join(videos_path, video), images_path+'/'+video[:-4]+"/images")
    # 执行 Shell 脚本
    video_args = images_path +'/'+ video[:-4]
    start_time = time.time()    
    subprocess.run(['bash', script_path, video_args])
    end_time = time.time() 
    # 删掉images文件夹
    subprocess.run(['rm', '-rf', video_args + '/images'])
    with open(log_path, 'a') as f:
        f.write(f"delete {video_args + '/images'}\n")
        f.write(f"{video} spend time: {end_time-start_time}\n") 
    

# 执行完成
print("所有执行已完成!")
