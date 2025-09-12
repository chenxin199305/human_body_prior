import torch
import numpy as np

from body_visualizer.tools.vis_tools import render_smpl_params
from body_visualizer.tools.vis_tools import imagearray2file
from body_visualizer.tools.vis_tools import show_image

# ====================================================================================================

# This tutorial requires 'vposer_v2_05'

from os import path as osp

support_dir = '../support_data/dowloads'
expr_dir = osp.join(support_dir, 'vposer_v2_05')  # 'TRAINED_MODEL_DIRECTORY'  in this directory the trained model along with the model code exist
bm_fname = osp.join(support_dir, 'models/smplx/neutral/model.npz')  # 'PATH_TO_SMPLX_model.npz'  obtain from https://smpl-x.is.tue.mpg.de/downloads
sample_amass_fname = osp.join(support_dir, 'amass_sample.npz')  # a sample npz file from AMASS

print(
    f"expr_dir = {expr_dir}\n"
    f"bm_fname = {bm_fname}\n"
    f"sample_amass_fname = {sample_amass_fname}\n"
)

# ====================================================================================================


# Loading SMPLx Body Model
from human_body_prior.body_model.body_model import BodyModel

bm = BodyModel(bm_fname=bm_fname).to('cuda')

# ====================================================================================================

# Loading VPoser Body Pose Prior
from human_body_prior.tools.model_loader import load_model
from human_body_prior.models.vposer_model import VPoser

vp, ps = load_model(expr_dir,
                    model_code=VPoser,
                    remove_words_in_model_weights='vp_model.',
                    disable_grad=True)
vp = vp.to('cuda')

# ====================================================================================================

# Prepare the pose_body from amass sample
amass_body_pose = np.load(sample_amass_fname)['poses'][:, 3:66]
amass_body_pose = torch.from_numpy(amass_body_pose).type(torch.float).to('cuda')

print(
    f"amass_body_pose.shape = {amass_body_pose.shape}"
)

amass_body_poZ = vp.encode(amass_body_pose).mean

print(
    f"amass_body_poZ.shape = {amass_body_poZ.shape}"
)

# ====================================================================================================

amass_body_pose_rec = vp.decode(amass_body_poZ)['pose_body'].contiguous().view(-1, 63)

print(
    f"amass_body_pose_rec.shape = {amass_body_pose_rec.shape}"
)

# Let's visualize the original pose and the reconstructed one:

t = np.random.choice(len(amass_body_pose))

all_pose_body = torch.stack([amass_body_pose[t], amass_body_pose_rec[t]])

print(
    f"all_pose_body.shape = {all_pose_body.shape}"
)

images = render_smpl_params(bm, {'pose_body': all_pose_body}).reshape(1, 2, 1, 400, 400, 3)
img = imagearray2file(images)
show_image(img[0])
