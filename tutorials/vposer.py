import torch
import numpy as np

from body_visualizer.tools.vis_tools import render_smpl_params
from body_visualizer.tools.vis_tools import imagearray2file
from body_visualizer.tools.vis_tools import show_image

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
sample_amass_fname = osp.join(download_dir, "amass_sample.npz")  # a sample npz file from AMASS

print(
    f"bm_fname = {bm_fname}\n"
    f"vposer_dir = {vposer_dir}\n"
    f"sample_amass_fname = {sample_amass_fname}\n"
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

# return the model and the model parameters
vp, ps = load_model(vposer_dir,
                    model_code=VPoser,
                    remove_words_in_model_weights="vp_model.",
                    disable_grad=True)
vp = vp.to(device)  # run on CPU

# ====================================================================================================

# Prepare the pose_body from amass sample
amass_body_pose = np.load(sample_amass_fname)["poses"][:, 3:66]
amass_body_pose = torch.from_numpy(amass_body_pose).type(torch.float).to(device)

print(
    f"amass_body_pose.shape = {amass_body_pose.shape}"
)

# raw data -> vp -> latent space
amass_body_poZ = vp.encode(amass_body_pose).mean

print(
    f"amass_body_poZ.shape = {amass_body_poZ.shape}"
)

# ====================================================================================================

# latent space -> vp -> reconstructed data
amass_body_pose_rec = vp.decode(amass_body_poZ)["pose_body"].contiguous().view(-1, 63)

print(
    f"amass_body_pose_rec.shape = {amass_body_pose_rec.shape}"
)

# Let"s visualize the original pose and the reconstructed one:

t = np.random.choice(len(amass_body_pose))

all_pose_body = torch.stack([amass_body_pose[t], amass_body_pose_rec[t]])

print(
    f"all_pose_body.shape = {all_pose_body.shape}"
)

# images = render_smpl_params(bm, {"pose_body": all_pose_body}).reshape(1, 2, 1, 400, 400, 3)

# 获取渲染后的图像
images = render_smpl_params(bm, {"pose_body": all_pose_body})
images = images.reshape(1, 2, 1, 800, 800, 3)

img = imagearray2file(images)
show_image(img[0])
