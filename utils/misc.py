import torch
import collections
import os
import random
import numpy as np


def fixseed(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def wrapped_getattr(self, name, default=None, wrapped_member_name='model'):
    ''' should be called from wrappers of model classes such as ClassifierFreeSampleModel'''

    if isinstance(self, torch.nn.Module):
        # for descendants of nn.Module, name may be in self.__dict__[_parameters/_buffers/_modules] 
        # so we activate nn.Module.__getattr__ first.
        # otherwise, we might encounter an infinite loop
        try:
            attr = torch.nn.Module.__getattr__(self, name)
        except AttributeError:
            wrapped_member = torch.nn.Module.__getattr__(self, wrapped_member_name)
            attr = getattr(wrapped_member, name, default)
    else:
        # the easy case, where self is not derived from nn.Module
        wrapped_member = getattr(self, wrapped_member_name)
        attr = getattr(wrapped_member, name, default)
    return attr        


def to_numpy(tensor):
    if torch.is_tensor(tensor):
        return tensor.cpu().numpy()
    elif type(tensor).__module__ != 'numpy':
        raise ValueError("Cannot convert {} to numpy array".format(
            type(tensor)))
    return tensor


def to_torch(ndarray):
    if type(ndarray).__module__ == 'numpy':
        return torch.from_numpy(ndarray)
    elif not torch.is_tensor(ndarray):
        raise ValueError("Cannot convert {} to torch tensor".format(
            type(ndarray)))
    return ndarray


def cleanexit():
    import sys
    import os
    try:
        sys.exit(0)
    except SystemExit:
        os._exit(0)

def load_model_wo_clip(model, state_dict):
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    assert len(unexpected_keys) == 0
    assert all([k.startswith('clip_model.') for k in missing_keys])

def freeze_joints(x, joints_to_freeze):
    # Freezes selected joint *rotations* as they appear in the first frame
    # x [bs, [root+n_joints], joint_dim(6), seqlen]
    frozen = x.detach().clone()
    frozen[:, joints_to_freeze, :, :] = frozen[:, joints_to_freeze, :, :1]
    return frozen


def tensor_to_device(x, device):
    if isinstance(x, torch.Tensor):
        return x.to(device)
    else:
        return x


def recursive_op2(x, y, op):
    assert type(x) == type(y)
    if isinstance(x, collections.Mapping):
        assert x.keys() == y.keys()
        return {k: recursive_op2(v1, v2, op) for (k, v1), v2 in zip(x.items(), y.values())}
    elif isinstance(x, collections.Sequence) and not isinstance(x, str):
        Warning('recursive_op2 on a sequence has never been tested')
        return [recursive_op2(v1, v2, op) for v1, v2 in zip(x, y)]
    elif isinstance(x, tuple):
        Warning('recursive_op2 on a tuple has never been tested')
        return tuple([recursive_op2(v1, v2, op) for v1, v2 in zip(x, y)])
    else:
        return op(x, y)


def recursive_op1(x, op, **kwargs):
    if isinstance(x, collections.Mapping):
        return {k: recursive_op1(v, op, **kwargs) for (k, v) in x.items()}
    elif isinstance(x, collections.Sequence) and not isinstance(x, str):
        return [recursive_op1(v, op, **kwargs) for v in x]
    elif isinstance(x, tuple):
        Warning('recursive_op1 on a tuple has never been tested')
        return tuple([recursive_op1(v, op, **kwargs) for v in x])
    else:
        return op(x, **kwargs)


def normalize(data, axis):
    return (data - data.min(axis=axis, keepdims=True)) / (data.max(axis=axis, keepdims=True) - data.min(axis=axis, keepdims=True) + 1e-7)


def get_project_root_path():
    return os.path.dirname(os.path.dirname(__file__))


def rename_files(directory, from_str, to_str):
    for root, _, files in os.walk(directory):
        for file in files:
            if from_str in file:
                old_file_path = os.path.join(root, file)
                new_file_name = file.replace(from_str, to_str)
                new_file_path = os.path.join(root, new_file_name)
                # print(f'renaming {old_file_path} to {new_file_path}')
                os.rename(old_file_path, new_file_path)


class Normalizer():
    def __init__(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        self.motion_mean = mean
        self.motion_std = std

    def forward(self, x, feature_idx=-1):
        mean, std = self.adjust_dim(x, feature_idx)
        x = (x - mean) / std
        return x

    def backward(self, x, feature_idx=-1):
        mean, std = self.adjust_dim(x, feature_idx)
        x = x * std + mean
        return x

    def adjust_dim(self, x, feature_idx=-1):
        mean = self.motion_mean
        std = self.motion_std
        if feature_idx != -1:
            for _ in range(feature_idx):  
                mean = mean.unsqueeze_(0)
                std = std.unsqueeze_(0)
            for _ in range(feature_idx+1, x.ndim):  
                mean = mean.unsqueeze(-1)
                std = std.unsqueeze(-1)
        return mean, std

