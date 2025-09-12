# VPoser: Variational Human Pose Prior for Body Inverse Kinematics

![alt text](../support_data/vposer_samples.png "Novel Human Poses Sampled From the VPoser.")

## Description

The articulated 3D pose of the human body is high-dimensional and complex.
Many applications make use of a prior distribution over valid human poses, but modeling this distribution is difficult.
Here we present VPoser, a learning based variational human pose prior trained from a large dataset of human poses represented as SMPL bodies.
This body prior can be used as an Inverse Kinematics (IK) solver for many tasks such as fitting a body model to images
as the main contribution of this repository for [SMPLify-X](https://smpl-x.is.tue.mpg.de/).
VPoser has the following features:

- defines a prior of SMPL pose parameters
- is end-to-end differentiable
- provides a way to penalize impossible poses while admitting valid ones
- effectively models correlations among the joints of the body
- introduces an efficient, low-dimensional, representation for human pose
- can be used to generate valid 3D human poses for data-dependent tasks

### Environment Setup

Conda environment is needed at first, then run the following command to install required packages.

```bash
# Create conda environment
conda create -n hbp python=3.7 -y
conda activate hbp

# Install required packages
bash install_env.sh
```

## Tutorials

![alt text](../support_data/latent_interpolation_1.gif "Interpolation of novel poses on the smoother VPoser latent space.")
![alt text](../support_data/latent_interpolation_2.gif "Interpolation of novel poses on the smoother VPoser latent space.")

* [VPoser Body poZ Space for SMPL Body Model Family](../tutorials/vposer.ipynb)
* [Sampling Novel Body Poses with VPoser](../tutorials/vposer_sampling.ipynb)

## Advanced IK Capabilities

![alt text](../support_data/SMPL_inverse_kinematics.gif "Batched SMPL Inverse Kinematics With Learned Body Prior")

Given position of some key points one can find the necessary body joints' rotation configurations via inverse kinematics (IK).
The keypoints could either be 3D (joint locations, 3D mocap markers on body surface) or 2D (as in [SMPLify-X](https://smpl-x.is.tue.mpg.de/)).
We provide a comprehensive IK engine with flexible key point definition interface demonstrated in tutorials:

- [IK for 3D joints](../tutorials/ik_example_joints.py)
- [IK for mocap markers](../tutorials/ik_example_mocap.py)

One can define keypoints on the SMPL body, e.g. joints, or any locations relative to the body surface
and fit body model parameters to them while utilizing the efficient learned pose parameterization of
[VPoser](https://github.com/nghorbani/human_body_prior). The supported features are:

- Batch enabled
- Flexible key point definition
- LBFGS with wolfe line-search and ADAM optimizer already enabled
- No need for initializing the body (always starts from zero)
- Optimizes body pose, translation and body global orientation jointly and iteratively

## Train VPoser

We train VPoser, as a [variational autoencoder](https://arxiv.org/abs/1312.6114)
that learns a latent representation of human pose and regularizes the distribution of the latent code
to be a normal distribution.
We train our prior on data from the [AMASS](https://amass.is.tue.mpg.de/) dataset,
that holds the SMPL pose parameters of various publicly available human motion capture datasets.
