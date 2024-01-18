import cv2
import numpy as np
import json
import os

def calculate_relative_pose(img1_path, img2_path, ratio=0.8, fx=1375.52, fy=1374.49, cx=554.558, cy=965.268):
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    # 读取图像
    img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)

    # 创建SIFT特征提取器
    sift = cv2.SIFT_create()

    # 提取特征并匹配
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    # 使用BFMatcher进行匹配
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    # 使用Lowe's ratio测试筛选好的匹配
    good = []
    for m, n in matches:
        if m.distance < ratio * n.distance:
            good.append(m)
    print(f'good matches: {len(good)}')
    # 假设所有匹配都是好的匹配，将他们转换为NumPy数组
    pts1 = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    # 计算本质矩阵
    E, mask = cv2.findEssentialMat(pts1, pts2, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)

    # 从本质矩阵中恢复旋转和平移
    _, R, t, _ = cv2.recoverPose(E, pts1, pts2, K)
    # 把x轴和y轴反向
    t[0] = -t[0]
    t[1] = -t[1]
    t = 0.1 * t  # 乘以一个系数，使平移向量更小
    # 把R，t转换为4x4矩阵
    T = np.hstack((R, t))
    T = np.vstack((T, np.array([0, 0, 0, 1])))
    return T, t

def find_min_max_translation(frames):
    # 统计平移向量在x，y，z方向上的最小值和最大值
    min_translation = (1000, 1000, 1000)
    max_translation = (-1000, -1000, -1000)
    for frame in frames:
        transformation = np.array(frame["transform_matrix"])
        translation = np.array(transformation[:3, 3])
        min_translation = tuple(map(min, zip(min_translation, translation)))
        max_translation = tuple(map(max, zip(max_translation, translation)))    
    return min_translation, max_translation

    

if __name__ == '__main__':
    fx = 1375.52
    fy = 1374.49
    cx = 554.558
    cy = 965.268
    img_path1 = "/hhd2/home/Code/lzl_gen/nerf_code/images/moving_computer/00000001.png"
    img_path2 = "/hhd2/home/Code/lzl_gen/nerf_code/images/moving_computer/00000002.png"
    T,t = calculate_relative_pose(img_path1, img_path2, fx=fx, fy=fy, cx=cx, cy=cy, ratio=0.8)
    # file_path = os.path.dirname(os.path.abspath(__file__))
    # img_dir = "/hhd2/home/Code/lzl_gen/instant-ngp/data/nerf/demo_infer/images"
    # json_path = "JSONS/milk_transforms.json"
    # img_dir = os.path.join(file_path, img_dir)
    # json_path = os.path.join(file_path, json_path)
    # output_path = os.path.join(file_path, "JSONS/demo_scale0.1.json")
    
    # img_list = os.listdir(img_dir)
    # img_list.sort()
    # result = dict()
    # result["frames"] = list()
    
    # with open(json_path, 'r') as f:
    #     data = json.load(f)
    
    # # 统计训练集中平移向量的最小值和最大值
    # min_translation, max_translation = find_min_max_translation(data["frames"])
    
    # # 从json中选定初始位姿选定初始位姿
    # initial_pose = data["frames"][0]["transform_matrix"]
    # initial_pose = np.array(initial_pose)
    # result["frames"].append({"file_path": "images/"+img_list[0], "transform_matrix": initial_pose.tolist()})
    
    # # 获取初始位姿的平移向量
    # initial_translation = initial_pose[:3, 3]
    
    # # 从第二张图片开始计算相对位姿
    # current_pose = initial_pose
    # translations = list()
    # for i in range(1, len(img_list)):
    #     img_path1 = os.path.join(img_dir, img_list[i-1])
    #     img_path2 = os.path.join(img_dir, img_list[i])
    #     # 计算相对位姿, t相对平移向量
    #     T,t = calculate_relative_pose(img_path1, img_path2, fx=fx, fy=fy, cx=cx, cy=cy, ratio=0.8)
    #     translations.append(t)
    #     # 计算绝对位姿
    #     T = np.dot(current_pose,T )
    #     current_pose = T
    #     # 把绝对位姿添加到result中
    #     result["frames"].append({"file_path": "images/"+ img_list[i], "transform_matrix": T.tolist()})
        
    #     # if i == 20:
    #     #     break
        
    # # 统计结果中的平移向量的最小值和最大值
    # # result_min_translation, result_max_translation = find_min_max_translation(result["frames"])
    
    
    # # with open('output.txt', 'w') as f:
    # #     for item in translations:
    # #         item = item.flatten()
    # #         length = np.sum(item**2)
    # #         print(length)
    # #         f.write(f'length:{length}, t: {item}\n')
   
    # # 后处理，保持初始平移向量大小不变将结果的平移向量根据比例缩放到(min_translation, initial_translation)和(max_translation, initial_translation)之间之间
    # # for frame in result["frames"]:
    # #     transformation = np.array(frame["transform_matrix"])
    # #     translation = np.array(transformation[:3, 3])
    # #     for i in range(3):
    # #         if translation[i] > initial_translation[i]:
    # #             translation[i] = (translation[i] - initial_translation[i]) / (result_max_translation[i] - initial_translation[i]) * (max_translation[i] - initial_translation[i]) + initial_translation[i]
    # #         elif translation[i] < initial_translation[i]:
    # #             translation[i] = ((translation[i] - initial_translation[i]) / (result_min_translation[i] - initial_translation[i])) * (min_translation[i] - initial_translation[i]) + initial_translation[i]
            
    # #     transformation[:3, 3] = translation
    # #     frame["transform_matrix"] = transformation.tolist()
    
    # # 把data中除了frames之外的内容复制到result中
    # for key in data:
    #     if key != "frames":
    #         result[key] = data[key]
    
    
    # with open(output_path, 'w') as f:
    #     json.dump(result, f)
    
    
