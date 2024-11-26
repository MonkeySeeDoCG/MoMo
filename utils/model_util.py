from typing import List, Tuple

import torch

from model.mdm import MDM
from diffusion import gaussian_diffusion as gd
from diffusion.respace import SpacedDiffusion, space_timesteps
from data_utils.tensors import collate
from utils import dist_util
from utils.misc import recursive_op1, tensor_to_device


def load_model_wo_clip(model, state_dict):
    del state_dict['sequence_pos_encoder.pe']  # no need to load it (fixed), and causes size mismatch for older models
    del state_dict['embed_timestep.sequence_pos_encoder.pe']  # no need to load it (fixed), and causes size mismatch for older models
    
    # backward compitability: change self_attn to self_frame_attn
    self_attn_keys = [k for k in state_dict.keys() if 'self_attn' in k]
    for k in self_attn_keys:
        state_dict[k.replace('self_attn', 'self_frame_attn')] = state_dict[k]
        # del state_dict[k]  # do NOT del 'self_attn' objects, since they too exist in the new model
    
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    
    # backward compitability #todo: remove after a new model is trained
    unexpected_keys = [key for key in unexpected_keys if 'self_frame_attn' not in key] # self_frame_attn is equal to self_attn so not harm done
    
    assert len(unexpected_keys) == 0
    assert all([k.startswith('clip_model.') or k.startswith('bert_model.') or 'sequence_pos_encoder' in k for k in missing_keys])


def create_model_and_diffusion(args, data, additional_model_args={}):
    model = MDM(**get_model_args(args, data, additional_model_args))
    diffusion = create_gaussian_diffusion(args)      
    return model, diffusion


def get_model_args(args, data, additional_args={}):

    # default args
    clip_version = 'ViT-B/32'
    njoints = data.n_features  # currently only humanml data is used, where its features contain both joints and feature data
    nfeats = 1

    model_args =  {'njoints': njoints, 'nfeats': nfeats, 
            'translation': True, 'glob': True, 'glob_rot': True,
            'latent_dim': args.latent_dim, 'ff_size': 1024, 'num_layers': args.layers, 'num_heads': 4,
            'dropout': 0.1, 'activation': "gelu", 
            'cond_mask_prob': args.cond_mask_prob, 'arch': args.arch,
            'emb_trans_dec': args.emb_trans_dec, 'clip_version': clip_version, 'dataset': args.dataset,
            'emb_before_mask': args.emb_before_mask, 'text_encoder_type': args.text_encoder_type,
            'diffusion_steps': args.diffusion_steps}
    model_args.update(additional_args)
    return model_args


def create_gaussian_diffusion(args):
    # default params
    predict_xstart = True  # we always predict x_start (a.k.a. x0), that's our deal!
    steps = args.diffusion_steps
    scale_beta = 1.  # no scaling
    timestep_respacing = ''  # can be used for ddim sampling, we don't use it.
    learn_sigma = False
    rescale_timesteps = False

    betas = gd.get_named_beta_schedule(args.noise_schedule, steps, scale_beta)
    loss_type = gd.LossType.MSE

    if not timestep_respacing:
        timestep_respacing = [steps]

    return SpacedDiffusion(
        use_timesteps=space_timesteps(steps, timestep_respacing),
        betas=betas,
        model_mean_type=(
            gd.ModelMeanType.EPSILON if not predict_xstart else gd.ModelMeanType.START_X
        ),
        model_var_type=(
            (
                gd.ModelVarType.FIXED_LARGE
                if not args.sigma_small
                else gd.ModelVarType.FIXED_SMALL
            )
            if not learn_sigma
            else gd.ModelVarType.LEARNED_RANGE
        ),
        loss_type=loss_type,
        rescale_timesteps=rescale_timesteps,
        lambda_vel=args.lambda_vel,
        lambda_rcxyz=args.lambda_rcxyz,
        lambda_fc=args.lambda_fc,
        data_rep=args.repr,
    )

def load_saved_model(model, model_path, use_avg: bool=False):  # use_avg_model
    state_dict = torch.load(model_path, map_location='cpu')
    # Use average model when possible
    if use_avg and 'model_avg' in state_dict.keys():
    # if use_avg_model:
        print('loading avg model')
        state_dict = state_dict['model_avg']
    else:
        if 'model' in state_dict:
            print('loading model without avg')
            state_dict = state_dict['model']
        else:
            print('checkpoint has no avg model, loading as usual.')
    load_model_wo_clip(model, state_dict)
    return model


def load_into_model_format(motion_paths, normalizer_class, load_motions_func):
    """ load motions from file and convert them into the format used by the model (currently mdm)"""
    
    motions, prompts, lengths = load_motions_func(motion_paths)
    n_motions = len(motions)
    collate_args = [{'inp': motion, 'tokens': None, 'lengths': len, 'text':txt} for motion, len, txt in zip(motions, lengths, prompts)]
    motions, motions_kwargs = collate(collate_args)
    
    # move to device
    motions = motions.to(dist_util.dev())
    motions_kwargs = recursive_op1(motions_kwargs, tensor_to_device, device=dist_util.dev())
    motions_kwargs['y']['scale'] = torch.ones(n_motions).to(dist_util.dev())
            
    motions = normalizer_class().forward(motions, feature_idx=1)
    return motions, motions_kwargs
