import argparse
from glob import glob
import os
from pathlib import Path, PurePosixPath

import numpy as np
import json
import sys
import math
import cv2
import os
import shutil

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
SCRIPTS_FOLDER = os.path.join(ROOT_DIR, "scripts")

def parse_args():
    parser = argparse.ArgumentParser(description="Convert a text colmap export to nerf format transforms.json; optionally convert video to images, and optionally run colmap in the first place.")

    parser.add_argument("--video_in", default="", help="Run ffmpeg first to convert a provided video file into a set of images. Uses the video_fps parameter also.")
    parser.add_argument("--video_fps", default=2)
    parser.add_argument("--time_slice", default="", help="Time (in seconds) in the format t1,t2 within which the images should be generated from the video. E.g.: \"--time_slice '10,300'\" will generate images only from 10th second to 300th second of the video.")
    parser.add_argument("--run_colmap", action="store_true", help="run colmap first on the image folder")
    parser.add_argument("--colmap_matcher", default="exhaustive", choices=["exhaustive","sequential","spatial","transitive","vocab_tree"], help="Select which matcher colmap should use. Sequential for videos, exhaustive for ad-hoc images.")
    parser.add_argument("--colmap_db", default="colmap.db", help="colmap database filename")
    parser.add_argument("--colmap_camera_model", default="OPENCV", choices=["SIMPLE_PINHOLE", "PINHOLE", "SIMPLE_RADIAL", "RADIAL", "OPENCV", "SIMPLE_RADIAL_FISHEYE", "RADIAL_FISHEYE", "OPENCV_FISHEYE"], help="Camera model")
    parser.add_argument("--colmap_camera_params", default="", help="Intrinsic parameters, depending on the chosen model. Format: fx,fy,cx,cy,dist")
    parser.add_argument("--images", default="/hhd2/home/Code/lzl_gen/colmap_scripts-nerf_preprocess/tapall_colmap_data/toy/images", help="Input path to the images.")
    parser.add_argument("--text", default="/hhd2/home/Code/lzl_gen/colmap_scripts-nerf_preprocess/tapall_colmap_data/toy/text", help="Input path to the colmap text files (set automatically if --run_colmap is used).")
    parser.add_argument("--aabb_scale", default=32, choices=["1", "2", "4", "8", "16", "32", "64", "128"], help="Large scene scale factor. 1=scene fits in unit cube; power of 2 up to 128")
    parser.add_argument("--skip_early", default=0, help="Skip this many images from the start.")
    parser.add_argument("--keep_colmap_coords", action="store_true", help="Keep transforms.json in COLMAP's original frame of reference (this will avoid reorienting and repositioning the scene for preview and rendering).")
    parser.add_argument("--out", default="/hhd2/home/Code/lzl_gen/colmap_scripts-nerf_preprocess/tapall_colmap_data/toy/transforms.json", help="Output path.")
    parser.add_argument("--vocab_path", default="", help="Vocabulary tree path.")
    parser.add_argument("--overwrite", action="store_true", help="Do not ask for confirmation for overwriting existing images and COLMAP data.")
    parser.add_argument("--mask_categories", nargs="*", type=str, default=[], help="Object categories that should be masked out from the training images. See `scripts/category2id.json` for supported categories.")
    parser.add_argument("--seg_path", default = None, help="四通道图存储路径")
    args = parser.parse_args()
    return args

def variance_of_laplacian(image):
	return cv2.Laplacian(image, cv2.CV_64F).var()

def sharpness(imagePath):
	image = cv2.imread(imagePath)
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	fm = variance_of_laplacian(gray)
	return fm

def qvec2rotmat(qvec):
	return np.array([
		[
			1 - 2 * qvec[2]**2 - 2 * qvec[3]**2,
			2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
			2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]
		], [
			2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
			1 - 2 * qvec[1]**2 - 2 * qvec[3]**2,
			2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]
		], [
			2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
			2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
			1 - 2 * qvec[1]**2 - 2 * qvec[2]**2
		]
	])

def rotmat(a, b):
	a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
	v = np.cross(a, b)
	c = np.dot(a, b)
	# handle exception for the opposite direction input
	if c < -1 + 1e-10:
		return rotmat(a + np.random.uniform(-1e-2, 1e-2, 3), b)
	s = np.linalg.norm(v)
	kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
	return np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2 + 1e-10))

def closest_point_2_lines(oa, da, ob, db): # returns point closest to both rays of form o+t*d, and a weight factor that goes to 0 if the lines are parallel
	da = da / np.linalg.norm(da)
	db = db / np.linalg.norm(db)
	c = np.cross(da, db)
	denom = np.linalg.norm(c)**2
	t = ob - oa
	ta = np.linalg.det([t, db, c]) / (denom + 1e-10)
	tb = np.linalg.det([t, da, c]) / (denom + 1e-10)
	if ta > 0:
		ta = 0
	if tb > 0:
		tb = 0
	return (oa+ta*da+ob+tb*db) * 0.5, denom
args = parse_args()
AABB_SCALE = int(args.aabb_scale)
SKIP_EARLY = int(args.skip_early)
IMAGE_FOLDER = args.images
TEXT_FOLDER = args.text
OUT_PATH = args.out
print(f"outputting to {OUT_PATH}...")
cameras = {}
with open(os.path.join(TEXT_FOLDER,"cameras.txt"), "r") as f:
    camera_angle_x = math.pi / 2
    for line in f:
        # 1 SIMPLE_RADIAL 2048 1536 1580.46 1024 768 0.0045691
        # 1 OPENCV 3840 2160 3178.27 3182.09 1920 1080 0.159668 -0.231286 -0.00123982 0.00272224
        # 1 RADIAL 1920 1080 1665.1 960 540 0.0672856 -0.0761443
        if line[0] == "#":
            continue
        els = line.split(" ")
        camera = {}
        camera_id = int(els[0])
        camera["w"] = float(els[2])
        camera["h"] = float(els[3])
        camera["fl_x"] = float(els[4])
        camera["fl_y"] = float(els[4])
        camera["k1"] = 0
        camera["k2"] = 0
        camera["k3"] = 0
        camera["k4"] = 0
        camera["p1"] = 0
        camera["p2"] = 0
        camera["cx"] = camera["w"] / 2
        camera["cy"] = camera["h"] / 2
        camera["is_fisheye"] = False
        if els[1] == "SIMPLE_PINHOLE":
            camera["cx"] = float(els[5])
            camera["cy"] = float(els[6])
        elif els[1] == "PINHOLE":
            camera["fl_y"] = float(els[5])
            camera["cx"] = float(els[6])
            camera["cy"] = float(els[7])
        elif els[1] == "SIMPLE_RADIAL":
            camera["cx"] = float(els[5])
            camera["cy"] = float(els[6])
            camera["k1"] = float(els[7])
        elif els[1] == "RADIAL":
            camera["cx"] = float(els[5])
            camera["cy"] = float(els[6])
            camera["k1"] = float(els[7])
            camera["k2"] = float(els[8])
        elif els[1] == "OPENCV":
            camera["fl_y"] = float(els[5])
            camera["cx"] = float(els[6])
            camera["cy"] = float(els[7])
            camera["k1"] = float(els[8])
            camera["k2"] = float(els[9])
            camera["p1"] = float(els[10])
            camera["p2"] = float(els[11])
        elif els[1] == "SIMPLE_RADIAL_FISHEYE":
            camera["is_fisheye"] = True
            camera["cx"] = float(els[5])
            camera["cy"] = float(els[6])
            camera["k1"] = float(els[7])
        elif els[1] == "RADIAL_FISHEYE":
            camera["is_fisheye"] = True
            camera["cx"] = float(els[5])
            camera["cy"] = float(els[6])
            camera["k1"] = float(els[7])
            camera["k2"] = float(els[8])
        elif els[1] == "OPENCV_FISHEYE":
            camera["is_fisheye"] = True
            camera["fl_y"] = float(els[5])
            camera["cx"] = float(els[6])
            camera["cy"] = float(els[7])
            camera["k1"] = float(els[8])
            camera["k2"] = float(els[9])
            camera["k3"] = float(els[10])
            camera["k4"] = float(els[11])
        else:
            print("Unknown camera model ", els[1])
        # fl = 0.5 * w / tan(0.5 * angle_x);
        camera["camera_angle_x"] = math.atan(camera["w"] / (camera["fl_x"] * 2)) * 2
        camera["camera_angle_y"] = math.atan(camera["h"] / (camera["fl_y"] * 2)) * 2
        camera["fovx"] = camera["camera_angle_x"] * 180 / math.pi
        camera["fovy"] = camera["camera_angle_y"] * 180 / math.pi

        print(f"camera {camera_id}:\n\tres={camera['w'],camera['h']}\n\tcenter={camera['cx'],camera['cy']}\n\tfocal={camera['fl_x'],camera['fl_y']}\n\tfov={camera['fovx'],camera['fovy']}\n\tk={camera['k1'],camera['k2']} p={camera['p1'],camera['p2']} ")
        cameras[camera_id] = camera

if len(cameras) == 0:
    print("No cameras found!")
    sys.exit(1)

with open(os.path.join(TEXT_FOLDER,"images.txt"), "r") as f:
    i = 0
    bottom = np.array([0.0, 0.0, 0.0, 1.0]).reshape([1, 4])
    if len(cameras) == 1:
        camera = cameras[camera_id]
        out = {
            "camera_angle_x": camera["camera_angle_x"],
            "camera_angle_y": camera["camera_angle_y"],
            "fl_x": camera["fl_x"],
            "fl_y": camera["fl_y"],
            "k1": camera["k1"],
            "k2": camera["k2"],
            "k3": camera["k3"],
            "k4": camera["k4"],
            "p1": camera["p1"],
            "p2": camera["p2"],
            "is_fisheye": camera["is_fisheye"],
            "cx": camera["cx"],
            "cy": camera["cy"],
            "w": camera["w"],
            "h": camera["h"],
            "aabb_scale": AABB_SCALE,
            "frames": [],
        }
    else:
        out = {
            "frames": [],
            "aabb_scale": AABB_SCALE
        }

    up = np.zeros(3)
    for line in f:
        line = line.strip()
        if line[0] == "#":
            continue
        i = i + 1
        if i < SKIP_EARLY*2:
            continue
        if  i % 2 == 1:
            elems=line.split(" ") # 1-4 is quat, 5-7 is trans, 9ff is filename (9, if filename contains no spaces)
            #name = str(PurePosixPath(Path(IMAGE_FOLDER, elems[9])))
            # why is this requireing a relitive path while using ^
            image_rel = os.path.relpath(IMAGE_FOLDER)
            name = str(f"./{image_rel}/{'_'.join(elems[9:])}")
            b = sharpness(name)
            print(name, "sharpness=",b)
            image_id = int(elems[0])
            qvec = np.array(tuple(map(float, elems[1:5])))
            tvec = np.array(tuple(map(float, elems[5:8])))
            R = qvec2rotmat(-qvec)
            t = tvec.reshape([3,1])
            m = np.concatenate([np.concatenate([R, t], 1), bottom], 0)
            c2w = np.linalg.inv(m)
            if not args.keep_colmap_coords:
                c2w[0:3,2] *= -1 # flip the y and z axis
                c2w[0:3,1] *= -1
                c2w = c2w[[1,0,2,3],:]
                c2w[2,:] *= -1 # flip whole world upside down

                up += c2w[0:3,1]

            frame = {"file_path":name,"sharpness":b,"transform_matrix": c2w}
            if len(cameras) != 1:
                frame.update(cameras[int(elems[8])])
            out["frames"].append(frame)
nframes = len(out["frames"])

if args.keep_colmap_coords:
    flip_mat = np.array([
        [1, 0, 0, 0],
        [0, -1, 0, 0],
        [0, 0, -1, 0],
        [0, 0, 0, 1]
    ])

    for f in out["frames"]:
        f["transform_matrix"] = np.matmul(f["transform_matrix"], flip_mat) # flip cameras (it just works)
else:
    # don't keep colmap coords - reorient the scene to be easier to work with

    up = up / np.linalg.norm(up)
    print("up vector was", up)
    R = rotmat(up,[0,0,1]) # rotate up vector to [0,0,1]
    R = np.pad(R,[0,1])
    R[-1, -1] = 1

    for f in out["frames"]:
        f["transform_matrix"] = np.matmul(R, f["transform_matrix"]) # rotate up to be the z axis

    # find a central point they are all looking at
    print("computing center of attention...")
    totw = 0.0
    totp = np.array([0.0, 0.0, 0.0])
    for f in out["frames"]:
        mf = f["transform_matrix"][0:3,:]
        for g in out["frames"]:
            mg = g["transform_matrix"][0:3,:]
            p, w = closest_point_2_lines(mf[:,3], mf[:,2], mg[:,3], mg[:,2])
            if w > 0.00001:
                totp += p*w
                totw += w
    if totw > 0.0:
        totp /= totw
    print(totp) # the cameras are looking at totp
    for f in out["frames"]:
        f["transform_matrix"][0:3,3] -= totp

    avglen = 0.
    for f in out["frames"]:
        avglen += np.linalg.norm(f["transform_matrix"][0:3,3])
    avglen /= nframes
    print("avg camera distance from origin", avglen)
    for f in out["frames"]:
        f["transform_matrix"][0:3,3] *= 4.0 / avglen # scale to "nerf sized"

for f in out["frames"]:
    f["transform_matrix"] = f["transform_matrix"].tolist()
print(nframes,"frames")
print(f"writing {OUT_PATH}")
with open(OUT_PATH, "w") as outfile:
    json.dump(out, outfile, indent=2)

if len(args.mask_categories) > 0:
    # Check if detectron2 is installed. If not, install it.
    try:
        import detectron2
    except ModuleNotFoundError:
        try:
            import torch
        except ModuleNotFoundError:
            print("PyTorch is not installed. For automatic masking, install PyTorch from https://pytorch.org/")
            sys.exit(1)

        input("Detectron2 is not installed. Press enter to install it.")
        import subprocess
        package = 'git+https://github.com/facebookresearch/detectron2.git'
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        import detectron2

    import torch
    from pathlib import Path
    from detectron2.config import get_cfg
    from detectron2 import model_zoo
    from detectron2.engine import DefaultPredictor

    category2id = json.load(open(SCRIPTS_FOLDER / "category2id.json", "r"))
    mask_ids = [category2id[c] for c in args.mask_categories]

    cfg = get_cfg()
    # Add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
    # Find a model from detectron2's model zoo.
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    predictor = DefaultPredictor(cfg)

    for frame in out["frames"]:
        img = cv2.imread(frame["file_path"])
        outputs = predictor(img)

        output_mask = np.zeros((img.shape[0], img.shape[1]))
        for i in range(len(outputs["instances"])):
            if outputs["instances"][i].pred_classes.cpu().numpy()[0] in mask_ids:
                pred_mask = outputs["instances"][i].pred_masks.cpu().numpy()[0]
                output_mask = np.logical_or(output_mask, pred_mask)

        rgb_path = Path(frame["file_path"])
        mask_name = str(rgb_path.parents[0] / Path("dynamic_mask_" + rgb_path.name.replace(".jpg", ".png")))
        cv2.imwrite(mask_name, (output_mask*255).astype(np.uint8))