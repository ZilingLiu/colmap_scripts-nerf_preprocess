import cv2
import numpy as np
import json
import os

def calculate_relative_pose(trans_matrix1, trans_matrix2):
    # m1 * relative_pose = m2 -> relative_pose = m1^-1 * m2
    # 计算第一个变换矩阵的逆
    inv_trans_matrix1 = np.linalg.inv(trans_matrix1)
    
    # 计算相对位姿
    relative_pose = np.dot(inv_trans_matrix1, trans_matrix2)
    
    return relative_pose

if __name__ == '__main__':
    file_path = os.path.dirname(os.path.abspath(__file__))

    img_dir = os.path.join(file_path, "/hhd2/home/Code/lzl_gen/Video-Poster/images/demo")
    nerf_json_path = os.path.join(file_path, "JSONS/milk_transforms.json")
    infer_json_path = os.path.join(file_path, "JSONS/infer_demo.json")
    
    output_path = os.path.join(file_path, "JSONS/demo_video1.json")
    
    result = dict()
    result["frames"] = list()
    
    with open(nerf_json_path, 'r') as f:
        data = json.load(f)
    
    # 从json中选定初始位姿选定初始位姿, 找frames中file_path为images/00000000.png的元素
    index = 0
    for i in range(len(data["frames"])):
        if data["frames"][i]["file_path"] == "images/00000109.png":
            index =  i
            break
    initial_pose = data["frames"][index]["transform_matrix"]
    initial_pose = np.array(initial_pose)
    img_list = os.listdir(img_dir)
    img_list.sort()
    result["frames"].append({"file_path": "images/"+img_list[0], "transform_matrix": initial_pose.tolist()})
    
    # 从infer json中取出所有的位姿
    with open(infer_json_path, 'r') as f:
        data_infer = json.load(f)
    infer_poses = data_infer["frames"]
    # 根据infer_poses每个元素的file_path对infer_poses进行排序
    infer_poses.sort(key=lambda x: x["file_path"])
    
        
    # 获取初始位姿的平移向量
    initial_translation = initial_pose[:3, 3]
    
    # 从第二张图片开始计算相对位姿
    current_pose = initial_pose
    translations = list()
    for i in range(1, len(img_list)):
        trans_matrix1 = infer_poses[i-1]["transform_matrix"]
        trans_matrix2 = infer_poses[i]["transform_matrix"]

        # 计算相对位姿, t相对平移向量
        relative_pose = calculate_relative_pose(trans_matrix1, trans_matrix2)
        
        # 计算绝对位姿
        current_pose = np.dot(current_pose, relative_pose)
        # 把绝对位姿添加到result中
        result["frames"].append({"file_path": "images/"+ img_list[i], "transform_matrix": current_pose.tolist()})
    
    # 把data中除了frames之外的内容复制到result中
    for key in data:
        if key != "frames":
            result[key] = data[key]
    
    
    with open(output_path, 'w') as f:
        json.dump(result, f)
    
    
