import torch
import numpy as np

from body_visualizer.tools.vis_tools import render_smpl_params
from body_visualizer.tools.vis_tools import imagearray2file, show_image

# ====================================================================================================

# This tutorial requires "vposer_v2_05"

from os import path as osp

current_file_path = osp.abspath(__file__)
current_dir = osp.dirname(current_file_path)
project_dir = osp.dirname(current_dir)

support_dir = osp.join(project_dir, "support_data")
download_dir = osp.join(support_dir, "downloads")

bm_fname = osp.join(download_dir, "models/smplx/neutral/model.npz")  # "PATH_TO_SMPLX_model.npz"  obtain from https://smpl-x.is.tue.mpg.de/downloads
vposer_dir = osp.join(download_dir, "vposer_v2_05")  # "TRAINED_MODEL_DIRECTORY"  in this directory the trained model along with the model code exist

print(
    f"bm_fname = {bm_fname}\n"
    f"vposer_dir = {vposer_dir}\n"
)

# ====================================================================================================

# Loading SMPLx Body Model
from human_body_prior.body_model.body_model import BodyModel

device = "cpu"  # run on CPU

bm = BodyModel(bm_fname=bm_fname).to(device)

# ====================================================================================================

# Loading VPoser Body Pose Prior
from human_body_prior.tools.model_loader import load_model
from human_body_prior.models.vposer_model import VPoser

vp, ps = load_model(vposer_dir,
                    model_code=VPoser,
                    remove_words_in_model_weights="vp_model.",
                    disable_grad=True)
vp = vp.to(device)

# ====================================================================================================

# number of body poses in each batch
num_poses = 9

# will a generate Nx1x21x3 tensor of body poses
sampled_pose_body = vp.sample_poses(num_poses=num_poses)["pose_body"].contiguous().view(num_poses, -1)
images = render_smpl_params(bm, {"pose_body": sampled_pose_body}).reshape(3, 3, 1, 800, 800, 3)

img = imagearray2file(images)
show_image(np.array(img[0]))

# ====================================================================================================
