import cv2
import time
import psutil
import os
def video2images(video_input,images_path):
    if not os.path.exists(images_path):
        os.makedirs(images_path)
    video_path = video_input
    frames = []
    user_name = time.time()
    operation_log = [("",""),("Upload video already. Try click the image for adding targets to track and inpaint.","Normal")]
    try:
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        index = 0
        while cap.isOpened():
            index += 1
            ret, frame = cap.read()
            if ret == True:
                current_memory_usage = psutil.virtual_memory().percent
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                if current_memory_usage > 90:
                    operation_log = [("Memory usage is too high (>90%). Stop the video extraction. Please reduce the video resolution or frame rate.", "Error")]
                    print("Memory usage is too high (>90%). Please reduce the video resolution or frame rate.")
                    break
            else:
                break
    except (OSError, TypeError, ValueError, KeyError, SyntaxError) as e:
        print("read_frame_source:{} error. {}\n".format(video_path, str(e)))
    # for i in range(len(frames)):
    #     cv2.imwrite(images_path + '\\' + '{:08d}.png'.format(i), frames[i])
    index = 0
    # factor = len(frames) // 150
    
    for i in range(len(frames)):
        # if i % factor == 0:
            cv2.imwrite(images_path + '/' + '{:08d}.png'.format(index), cv2.cvtColor(frames[i], cv2.COLOR_RGB2BGR))
            index += 1
    print(f'{video_path} has been extracted to {images_path}.')

# videos_path = "/hhd2/home/Code/lzl_gen/nerf_code/videos"
# videos = os.listdir(videos_path)
# for video in videos:
#     video2images(os.path.join(videos_path, video), "/hhd2/home/Code/lzl_gen/nerf_code/images/"+video[:-4]+"/images")
# video2images("/hhd2/home/Code/lzl_gen/nerf_code/videos/helan.mp4","/hhd2/home/Code/lzl_gen/nerf_code/images/closing/images")