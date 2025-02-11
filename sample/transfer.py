import os
from typing import List, Tuple
import numpy as np
import torch
import torch.nn.functional as F

from utils.parser_util import transfer_args
import data_utils.humanml.utils.paramUtil as paramUtil
from data_utils.humanml_utils import HumanMlNormalizer, load_motions
from sample.sample_utils import get_niter, get_sample_vars, init_main, save_results, sample_motions, get_xyz_rep, prepare_plot, get_max_frames
from utils.visualize import save_multiple_samples
import data_utils.humanml.utils.paramUtil as paramUtil
from utils.model_util import load_into_model_format
from utils import dist_util
from collections.abc import Iterable


def main():
    args = transfer_args()
    
    get_feat_idx, transfer_idx = get_transfer_args(args)
    additional_model_args = {'get_feat_idx': get_feat_idx, 'transfer_idx': transfer_idx}
    
    data, model, diffusion = init_main(args, additional_model_args)   
    return transfer(args, data, model, diffusion)


def transfer(args, data, model, diffusion):
    given_motion_kwargs = getattr(args, 'given_motion_kwargs', None)
    assert args.text_leader or args.leader_motion_path or given_motion_kwargs
    assert args.text_follower or args.follower_motion_path or given_motion_kwargs
    is_inv = args.leader_motion_path or args.follower_motion_path or given_motion_kwargs
    
    # enable real motions via inversion
    if is_inv:
        assert args.n_follower_mult == 1, 'Inversion not implemented for n_follower_mult > 1'  # todo: rotate follower motions n_follower_mult times before inversion
        if not given_motion_kwargs:
            # args.leader_motion_path = data.t2m_dataset.idx_to_path(given_motion_kwargs['y']['leader_idx'])            
            # args.follower_motion_path = [data.t2m_dataset.idx_to_path(given_motion_kwargs['y']['follower_idx'])]            
            # load motions to invert
            motion_paths = [args.leader_motion_path] if args.leader_motion_path is not None else []
            motion_paths += args.follower_motion_path
            motions_to_invert, motions_to_invert_kwargs = load_into_model_format(motion_paths, HumanMlNormalizer, load_motions)
        else:
            motions_to_invert = np.stack([given_motion_kwargs['y']['leader_motion'], given_motion_kwargs['y']['follower_motion']], axis=0)
            motions_to_invert = torch.from_numpy(motions_to_invert).to(dist_util.dev())
            motions_to_invert_kwargs = {'y': 
                {'text': [given_motion_kwargs['y']['leader_text'], given_motion_kwargs['y']['follower_text']], 
                'tokens': [given_motion_kwargs['y']['leader_tokens'], given_motion_kwargs['y']['follower_tokens']], 
                'lengths': torch.tensor([given_motion_kwargs['y']['leader_lengths'], given_motion_kwargs['y']['follower_lengths']]),
                'mask': torch.stack([given_motion_kwargs['y']['leader_mask'], given_motion_kwargs['y']['follower_mask']], axis=0), 
                'scale': torch.ones(2).to(dist_util.dev())}}
        show_progress = getattr(args, 'show_progress', True)
        inv_motions = diffusion.ddim_reverse_sample_loop(model, motions_to_invert, clip_denoised=False, 
                                                         model_kwargs=motions_to_invert_kwargs, progress=show_progress)['sample']        
        text_leader, text_follower, n_frames, inv_idx = merge_inverse_args(args, data, motions_to_invert_kwargs['y']['text'], motions_to_invert_kwargs['y']['lengths'])
        max_tensor_frames = max(n_frames + [motions_to_invert[0].shape[-1]])
    else:
        text_leader = args.text_leader
        text_follower = args.text_follower
        if isinstance(args.motion_length, Iterable):
            # currently this path is reached only during evaluation
            n_frames = args.motion_length  + [args.motion_length[0]]  # assign leader length to output - might need adaptatin to n_follower_mult one day
        else: 
            n_frames = None
        max_tensor_frames = None

    text_follower_mult = [x for x in text_follower for _ in range(args.n_follower_mult)]
    texts = [text_leader] + text_follower_mult + text_follower  # follower <-- text_follower_mult; out <-- args.text_follower
    args.num_samples = len(texts)

    out_path, shape, model_kwargs, max_frames = get_sample_vars(args, data, model, texts, 
                                                                GetOutName(text_leader, text_follower), n_frames=n_frames, max_tensor_frames=max_tensor_frames)
    model_kwargs['features_mode'] = 'transfer'

    # handle initial noise
    if is_inv:
        max_inv_frames = inv_motions.shape[-1]
        init_noise = torch.zeros(shape, dtype=inv_motions.dtype, device=dist_util.dev())
        init_noise[inv_idx, :, :, :max_inv_frames] = inv_motions  # max_inv_frames might be smaller than shape[-1]
        non_inv_idx = [i for i in range(shape[0]) if i not in inv_idx]
        init_noise[non_inv_idx] = torch.randn((len(non_inv_idx),)+shape[1:], device=dist_util.dev())   
        model_kwargs['y']['scale'][inv_idx] = 1.0
        model_kwargs['y']['scale'][model.transfer_idx['out']] = 1.0
    else:
        init_noise = None

    # sample is a list of (leader, followers, transferred) samples.
    all_motions, all_lengths, all_text = sample_motions(args, model, shape, model_kwargs, max_frames, init_noise, sample_func=diffusion.ddim_sample_loop)
   
    all_motions = assign_leader_root_rot(args, data.n_features, model.transfer_idx, all_motions)  
        
    # get xyz abs locations
    xyz_motions = get_xyz_rep(data, all_motions)

    # save outputs in an xyz format
    save_results(args, out_path, xyz_motions, all_lengths, all_text)

    visualize_motions(out_path, args, xyz_motions, all_text, all_lengths, data.fps, max_frames, model.transfer_idx, len(text_follower))
    return out_path, all_motions, xyz_motions


def assign_leader_root_rot(args, n_features, transfer_idx, all_motions):
    # assign the root rotation from the leader to the outputs 
    assert all_motions.shape[-3] == n_features, f'Expected {n_features} features, got {all_motions.shape[-3]}'
    if args.assign_root_rot:
        for rep_i in range(args.num_repetitions):
            out_idx = [args.num_samples*rep_i + i for i in transfer_idx['out']]
            all_motions[out_idx, :1, :, :] = all_motions[args.num_samples*rep_i, :1, :, :]
    return all_motions


def visualize_motions(out_path: List[str], args, all_motions: List[np.ndarray],
                            all_text: List[List[str]], all_lengths: List[np.ndarray],
                            fps: float, max_frames: int, transfer_idx: dict, n_follower: int):
    
    if not getattr(args, 'render', True):
        return out_path
    
    # get kinematic chain
    kinematic_chain = paramUtil.t2m_kinematic_chain

    n_follower_mult = len(transfer_idx['follower'])
    all_file_template = 'samples_{:02d}_to_{:02d}.mp4'
    anim_leader_and_out = np.empty(shape=(args.num_repetitions, 1+n_follower), dtype=object)
    anim_follower = np.empty(shape=(args.num_repetitions, n_follower_mult), dtype=object)

    idx_sample_i = 0
    for rep_i in range(args.num_repetitions):
        # plot leader motion
        anim_leader_and_out[rep_i, transfer_idx['leader']] = \
            prepare_plot(transfer_idx['leader'], rep_i, args, fps, all_motions, all_text, all_lengths, kinematic_chain, crop=True)
        # plot output motion
        for sample_i in transfer_idx['out']:
            # plot transferred motion near leader
            anim_leader_and_out[rep_i, sample_i-n_follower_mult] = \
                prepare_plot(sample_i, rep_i, args, fps, all_motions, all_text, all_lengths, kinematic_chain, crop=True)
        # plot follower motion in a separate figure
        for sample_i in transfer_idx['follower']:
            anim_follower[rep_i, sample_i-1] = \
                prepare_plot(sample_i, rep_i, args, fps, all_motions, all_text, all_lengths, kinematic_chain)  # we do not crop the follower because there could be multiple followers with varying lengths
                
        idx_sample_i += args.num_samples

    # save_multiple_samples should be called outside the args loop
    save_multiple_samples(out_path, {'all': all_file_template}, anim_leader_and_out, fps, all_lengths[0], prefix='transfer_', n_rows_in_out_file=args.n_rows_in_out_file)
    save_multiple_samples(out_path, {'all': all_file_template}, anim_follower, fps, max_frames, prefix='follower_', n_rows_in_out_file=args.n_rows_in_out_file)

    abs_path = os.path.abspath(out_path)
    print(f'[Done] Results are at {abs_path}')

    return out_path


class GetOutName():
    def __init__(self, text_leader=None, text_follower=None):
        self.text_leader = text_leader
        self.text_follower = text_follower
    def __call__(self, args):
        if self.text_leader is None:
            self.text_leader = args.text_leader
        if self.text_follower is None:
            self.text_follower = args.text_follower
        niter = get_niter(args.model_path)
        mult = f'x{args.n_follower_mult}' if args.n_follower_mult > 1 else ''
            
        out_name = (f'transfer_{niter}_seed{args.seed}_'
                        f'leader_{self.text_leader[:60].replace(" ", "-")}_'
                        f'follower{mult}_{self.text_follower[0][:60].replace(" ", "-")}')

        n_follower = len(self.text_follower)
        if n_follower > 1:
            out_name += '_and_{}_more'.format(n_follower-1)
        return out_name


def transfer_indices(ind_range: Tuple[int], max_elements: int) -> List[int]:
    if len(ind_range) == 2:
        start, end = ind_range
        skip = 1
    elif len(ind_range) == 3:
        start, end, skip = ind_range
    else:
        raise BaseException(f'Invalid number of arguments got for indices Expecting 2 or 3, got {len(ind_range)}')
    
    last_idx = end if end >= 0 else max_elements + end + 1
    assert last_idx > start, f'Invalid indices: {start} - {end} - (out of {max_elements})'
    return list(range(start, last_idx, skip))


def get_transfer_args(args):
    vis_layer = transfer_indices((args.transfer_layers_start, args.transfer_layers_end), args.layers)
    vis_step = transfer_indices((args.transfer_diff_step_start, args.transfer_diff_step_end, args.transfer_diff_step_step),
                                  args.diffusion_steps)

    # inject_leader_step = [step for step in vis_step if step % 5 == 0 and step < args.diffusion_steps-40]  # not used right now
    get_feat_idx = {'layer': vis_layer, 'step': vis_step}  # , 'inject_leader_step': inject_leader_step}    
    
    transfer_idx = get_transfer_idx(args)
    return get_feat_idx, transfer_idx


def get_transfer_idx(args):
    # indices of transfer elements (leader, follower, out) within the batch 
    # if neither text nor path are given, it means we are at an evaluation phase, where a single motion is transferred
    n_follower = len(args.follower_motion_path) if len(args.follower_motion_path) > 0 else len(args.text_follower) if args.text_follower else 1
    n_follower_total = n_follower*args.n_follower_mult
    follower_idx = list(range(1, 1+n_follower_total))
    out_idx = list(range(1+n_follower_total, 1+n_follower_total+n_follower))
    transfer_idx = {'leader': 0, 'follower': follower_idx, 'out': out_idx}
    return transfer_idx


def merge_inverse_args(args, data, inversion_texts, inversion_lengths):

    if args.leader_motion_path and args.text_leader: 
        Warning('leader_motion_path is given, ignoring text_leader')
    if args.follower_motion_path and args.text_follower: 
        Warning('follower_motion_path is given, ignoring text_follower')
    
    device=inversion_lengths.device
    running_idx = 0
    n_frames = torch.empty(0, device=device)  # todo: better hold n_frames as an array (or np.array of ints)
    given_motion_kwargs = getattr(args, 'given_motion_kwargs', None)
    
    # non_inverted_len will be used only for motions to be generated, in case some motions are inverted and some generated
    if not given_motion_kwargs:  # given_motion_kwargs exist only during evaluation, in which no generation is done
        non_inverted_len = get_max_frames(data, motion_length=args.motion_length)
    
    if args.leader_motion_path or given_motion_kwargs:
        text_leader =  inversion_texts[0]
        running_idx = 1
        n_frames = torch.cat((n_frames, inversion_lengths[:1]))
    else:
        n_frames = torch.cat((n_frames, torch.tensor([non_inverted_len], device=device)))
        text_leader = args.text_leader
    if args.follower_motion_path or given_motion_kwargs:
        text_follower = inversion_texts[running_idx:]
        n_frames = torch.cat((n_frames, inversion_lengths[running_idx:]))
    else:       
        text_follower = args.text_follower
        n_frames = torch.cat((n_frames, torch.tensor([non_inverted_len]*len(text_follower), device=device)))
    
    frames_follower_mult = n_frames[1:].repeat_interleave(args.n_follower_mult)
    n_frames = torch.cat((n_frames[:1], frames_follower_mult, n_frames[:1].repeat_interleave(len(text_follower))))
    n_frames = n_frames.to(int)
    
    inv_idx = []
    if args.leader_motion_path or given_motion_kwargs:
        inv_idx.append(0)
    if args.follower_motion_path or given_motion_kwargs:
        inv_idx.extend(range(1, 1+len(text_follower)*args.n_follower_mult))

    return text_leader, text_follower, n_frames.tolist(), inv_idx


if __name__ == "__main__":
    main()
