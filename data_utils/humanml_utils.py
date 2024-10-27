import numpy as np
import torch
from typing import List
import os.path as osp
import einops

from utils import dist_util
from utils.misc import Normalizer

HML_JOINT_NAMES = [
    'pelvis',
    'left_hip',
    'right_hip',
    'spine1',
    'left_knee',
    'right_knee',
    'spine2',
    'left_ankle',
    'right_ankle',
    'spine3',
    'left_foot',
    'right_foot',
    'neck',
    'left_collar',
    'right_collar',
    'head',
    'left_shoulder',
    'right_shoulder',
    'left_elbow',
    'right_elbow',
    'left_wrist',
    'right_wrist',
]

NUM_HML_JOINTS = len(HML_JOINT_NAMES)  # 22 SMPLH body joints

HML_LOWER_BODY_JOINTS = [HML_JOINT_NAMES.index(name) for name in ['pelvis', 'left_hip', 'right_hip', 'left_knee', 'right_knee', 'left_ankle', 'right_ankle', 'left_foot', 'right_foot',]]
SMPL_UPPER_BODY_JOINTS = [i for i in range(len(HML_JOINT_NAMES)) if i not in HML_LOWER_BODY_JOINTS]


# Recover global angle and positions for rotation data
# root_rot_velocity (B, seq_len, 1)
# root_linear_velocity (B, seq_len, 2)
# root_y (B, seq_len, 1)
# ric_data (B, seq_len, (joint_num - 1)*3)
# rot_data (B, seq_len, (joint_num - 1)*6)
# local_velocity (B, seq_len, joint_num*3)
# foot contact (B, seq_len, 4)
HML_ROOT_BINARY = np.array([True] + [False] * (NUM_HML_JOINTS-1))
HML_ROOT_MASK = np.concatenate(([True]*(1+2+1),
                                HML_ROOT_BINARY[1:].repeat(3),
                                HML_ROOT_BINARY[1:].repeat(6),
                                HML_ROOT_BINARY.repeat(3),
                                [False] * 4))
HML_LOWER_BODY_JOINTS_BINARY = np.array([i in HML_LOWER_BODY_JOINTS for i in range(NUM_HML_JOINTS)])
HML_LOWER_BODY_MASK = np.concatenate(([True]*(1+2+1),
                                     HML_LOWER_BODY_JOINTS_BINARY[1:].repeat(3),
                                     HML_LOWER_BODY_JOINTS_BINARY[1:].repeat(6),
                                     HML_LOWER_BODY_JOINTS_BINARY.repeat(3),
                                     [True]*4))
HML_UPPER_BODY_MASK = ~HML_LOWER_BODY_MASK

def load_motions(motion_paths: List[str]):
    var_len_motions = []
    prompts = []
    
    for motion_path in motion_paths:
        # todo: use code from dataset.py. This usage must depend on the selected dataset
        
        # get the motion and convert it to what the dataset loader outputs
        motion = np.load(motion_path, allow_pickle=True)  
        motion = einops.rearrange(motion, 'time features -> features 1 time')
        motion = torch.from_numpy(motion)
        var_len_motions.append(motion)
    
        # get the corresponding prompt
        folder, file = osp.split(motion_path)
        base_path = osp.split(folder)[0]
        file_no_suffix = osp.splitext(file)[0]
        prompt_path = osp.join(base_path, 'texts', file_no_suffix + '.txt')
        with open(prompt_path, 'r') as f:
            first_line = f.readline().strip()
        prompt = first_line.split('#')[0]

        prompts.append(prompt)  
    
    lengths = [motion.shape[-1] for motion in var_len_motions]
    return var_len_motions, prompts, lengths

class HumanMlNormalizer(Normalizer):
    def __init__(self):
        mean = torch.from_numpy(np.load('dataset/HumanML3D/Mean.npy')).to(dist_util.dev())
        std = torch.from_numpy(np.load('dataset/HumanML3D/Std.npy')).to(dist_util.dev())
        super().__init__(mean, std)
