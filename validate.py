import numpy as np
import cv2
import json

# 读取 JSON 数据
with open('transforms.json', 'r') as f:
    data = json.load(f)
P1 = data["frames"][0]["transform_matrix"]
P2 = data["frames"][1]["transform_matrix"]
P1 = np.array(P1)
P2 = np.array(P2)
P1_inverse = np.linalg.inv(P1)
relative_pose = np.dot(P1_inverse, P2)
print("Relative pose: ", relative_pose)
print("P1: ")
print(P1)
print("P2: ")
print(P2)
print("P1 * relative_pose: ")
print(np.dot(P1, relative_pose))