import torch
from data_utils.humanml.networks.modules import *
from data_utils.humanml.networks.trainers import CompTrainerV6
from torch.utils.data import Dataset, DataLoader
from os.path import join as pjoin
from tqdm import tqdm
from utils import dist_util
from data_utils.humanml.motion_loaders.comp_v6_model_dataset import CompMDMGeneratedDataset
from sample.transfer import transfer_from_noise
from sample.edit import upper_body_edit_from_noise
from data_utils.humanml.scripts.motion_process import recover_from_ric
from utils.test_utils import save_3motion
from copy import deepcopy
from utils.inverse import HumanMlNormalizer
import os
import json
from pathlib import Path

Transfer_LEGEND = ['leader', 'follower']

class CompTransferGeneratedDataset(CompMDMGeneratedDataset):

    def __init__(self, model, diffusion, dataloader, mm_num_samples, mm_num_repeats, max_motion_length, num_samples_limit, args):
                #  scale=1., mode='gen', render=False):
        self.dataloader = dataloader
        self.dataset = dataloader.dataset
        assert mm_num_samples < len(dataloader.dataset)
        self.max_motion_length = max_motion_length
        self.mode = args.eval_mode
        self.root_mode = args.root_mode
        self.render = args.render
        self.scale = args.guidance_param

        if self.mode == 'inversion':
            print('WARNING - For inversion mode - forcing scale=1')
            self.scale=1.
        
        normalizer = HumanMlNormalizer()
        
        real_num_batches = len(dataloader)
        if num_samples_limit is not None:
            real_num_batches = num_samples_limit // dataloader.batch_size + 1
        print('real_num_batches', real_num_batches)

        generated_motion = []
        mm_generated_motions = []
        if mm_num_samples > 0:
            mm_idxs = np.random.choice(real_num_batches, mm_num_samples // dataloader.batch_size +1, replace=False)
            mm_idxs = np.sort(mm_idxs)
        else:
            mm_idxs = []
        print('mm_idxs', mm_idxs)

        model.eval()

        self.aux_metrics = []

        with torch.no_grad():
            for i, (_, model_kwargs) in tqdm(enumerate(dataloader)):

                if num_samples_limit is not None and len(generated_motion) >= num_samples_limit:
                    break

                tokens = [t.split('_') for t in model_kwargs['y']['follower_tokens']]  # TODO - support leader tokens as well
                texts = model_kwargs['y']['follower_text']  # TODO - support leader texts as well
                lengths = []

                cur_bs = model_kwargs['y']['leader_motion'].shape[0]
                
                # add CFG scale to batch
                if self.scale != 1.:
                    model_kwargs['y']['scale'] = torch.ones(cur_bs,
                                                            device=dist_util.dev()) * self.scale

                mm_num_now = len(mm_generated_motions) // dataloader.batch_size
                is_mm = i in mm_idxs
                repeat_times = mm_num_repeats if is_mm else 1
                mm_motions = []
                for t in range(repeat_times):
                    
                    sample = []  # FIXME - for now, running example by example
                    for sample_i in tqdm(range(cur_bs)):
                        cur_len = [model_kwargs['y']['leader_lengths'][sample_i], model_kwargs['y']['follower_lengths'][sample_i]]
                        cur_len = [motion_len.item() for motion_len in cur_len]
                        lengths.append(model_kwargs['y']['leader_lengths'][sample_i])
                        prompts = [model_kwargs['y']['leader_text'][sample_i], model_kwargs['y']['follower_text'][sample_i]]
                        in_motions = None
                        in_motions = torch.cat([model_kwargs['y']['leader_motion'][[sample_i]], model_kwargs['y']['follower_motion'][[sample_i]]], dim=0)
                        # in_motions = in_motions[..., :cur_len]  # cut to min len  # FIXME - can we do something better here?
                        in_motions = in_motions.to(dist_util.dev())
                        if self.mode == 'upper_body':
                            all_outputs = upper_body_edit_from_noise(model, diffusion, [prompts[1]],
                                                                    in_motions[:1].clone(),
                                                                    motion_len=cur_len[:1],
                                                                    progress=False, scale=self.scale)
                        else:
                            _motions = in_motions if self.mode == 'inversion' else None
                            all_outputs = transfer_from_noise(model, diffusion, prompts, _motions,
                                                              motion_len=cur_len,
                                                              max_motion_len=in_motions.shape[-1],
                                                              progress=False, scale=self.scale)
                        
                        
                        if self.mode == 'gen' or self.mode == 'debug':
                            in_motions = all_outputs[:2]
                        
                        transfer_output = all_outputs[-1:]

                        # calc similarity metrics
                        transfer_proc = self.get_motion_proc(transfer_output)
                        in_proc = self.get_motion_proc(in_motions)
                        cur_metrics = self.calc_metrics(transfer_proc, in_proc, cur_len)
                        self.aux_metrics.append(cur_metrics)
                        sample.append(transfer_output)
                        
                        if self.render and i==0:
                            eval_prompts = [prompt for prompt in prompts] # deepcopy(prompts)
                            for k, v in cur_metrics.items():
                                idx_to_add = 0 if Transfer_LEGEND[0] in k else 1
                                _k = k.replace('follower_', '').replace('leader_', '')
                                eval_prompts[idx_to_add] += f' | {_k}={v:.2f}'
                                

                            save_3motion(motions=torch.cat([in_motions, transfer_output], dim=0).detach(), 
                                         motion_name='{}_{}_{}'.format(self.mode, model_kwargs['y']['leader_idx'][sample_i], model_kwargs['y']['follower_idx'][sample_i]),
                                        prompts=eval_prompts,
                                        normalizer=normalizer)
    
                    
                    # extract outputs
                    sample = torch.cat(sample, dim=0)  # -1 index is the output motion

                    # sample = sample_fn(
                    #     model,
                    #     motion.shape,
                    #     clip_denoised=clip_denoised,
                    #     model_kwargs=model_kwargs,
                    #     skip_timesteps=0,  # 0 is the default value - i.e. don't skip any step
                    #     init_image=None,
                    #     progress=False,
                    #     dump_steps=None,
                    #     noise=None,
                    #     const_noise=False,
                    #     # when experimenting guidance_scale we want to nutrileze the effect of noise on generation
                    # )

                    if t == 0:
                        sub_dicts = [{'motion': sample[bs_i].squeeze().permute(1,0).cpu().numpy(),
                                    'length': lengths[bs_i],
                                    'caption': texts[bs_i],
                                    'tokens': tokens[bs_i],
                                    'cap_len': len(tokens[bs_i]),
                                    } for bs_i in range(cur_bs)]
                        generated_motion += sub_dicts

                    if is_mm:
                        mm_motions += [{'motion': sample[bs_i].squeeze().permute(1, 0).cpu().numpy(),
                                        'length': lengths[bs_i],
                                        } for bs_i in range(cur_bs)]

                if is_mm:
                    mm_generated_motions += [{
                                    'caption': model_kwargs['y']['text'][bs_i],
                                    'tokens': tokens[bs_i],
                                    'cap_len': len(tokens[bs_i]),
                                    'mm_motions': mm_motions[bs_i::dataloader.batch_size],  # collect all 10 repeats from the (32*10) generated motions
                                    } for bs_i in range(dataloader.batch_size)]


        self.generated_motion = generated_motion
        self.mm_generated_motion = mm_generated_motions
        self.w_vectorizer = dataloader.dataset.w_vectorizer
    
    # root_rot_velocity (B, seq_len, 1)
    # root_linear_velocity (B, seq_len, 2)
    # root_y (B, seq_len, 1)
    # ric_data (B, seq_len, (joint_num - 1)*3)
    # rot_data (B, seq_len, (joint_num - 1)*6)
    # local_velocity (B, seq_len, joint_num*3)
    # foot contact (B, seq_len, 4)
    def get_motion_proc(self, motion):
        n_joints = 22 if motion.shape[1] == 263 else 21
        unnormed_motion = self.dataset.t2m_dataset.inv_transform(motion.cpu().permute(0, 2, 3, 1)).float()
        foot_contact = unnormed_motion[:,0,:, -4:] > 0.5
        motion_xyz = recover_from_ric(unnormed_motion, n_joints)
        motion_xyz = motion_xyz.view(-1, *motion_xyz.shape[2:]).permute(0, 2, 3, 1)
        traj = motion_xyz[:, 0, [0,2]].clone()
        local_loc = unnormed_motion[:,0,:, 4:4+(n_joints - 1)*3]
        rot = unnormed_motion[:,0,:, 4+(n_joints - 1)*3:4+(n_joints - 1)*3+(n_joints - 1)*6]
        return {'foot_contact': foot_contact, 'motion_xyz': motion_xyz, 'traj': traj, 'rot': rot, 'local_loc': local_loc}

    def calc_metrics(self, transfer_proc, in_proc, lengths):
        metrics = {}

        n_frames_out = lengths[0]

        for idx, name in enumerate(Transfer_LEGEND):

            n_frames = min(n_frames_out, lengths[idx])

            foot_contact_acc = (transfer_proc['foot_contact'][0, :n_frames] == in_proc['foot_contact'][idx, :n_frames]).sum() / transfer_proc['foot_contact'][0, :n_frames].numel()
            metrics[f'{name}_fc_acc'] = foot_contact_acc.cpu().numpy()

            traj_dist = torch.linalg.vector_norm(transfer_proc['traj'][0, :, :n_frames] - in_proc['traj'][idx, :, :n_frames], dim=0).mean()
            metrics[f'{name}_traj_dist'] = traj_dist.cpu().numpy()

        
        for key in ['rot', 'local_loc']:
            acum = []
            for frame_i in range(n_frames_out):
                frame = transfer_proc[key][0, frame_i]
                frame_dist = torch.linalg.vector_norm(in_proc[key] - frame, dim=-1)  # note calculating distance also to frames out of the mask
                closest_motion = torch.argmin(frame_dist.min(dim=-1)[0])
                acum.append(closest_motion[None])
            follower_sim = torch.sum(torch.cat(acum)).cpu().numpy() / float(n_frames_out)
            metrics[f'follower_{key}_sim'] = follower_sim
        
        return metrics
    
# For evaluating generated results by external models, saved to npy files
class CompTransferExternalDataset(CompTransferGeneratedDataset):

    def __init__(self, model, diffusion, dataloader, mm_num_samples, mm_num_repeats, max_motion_length, num_samples_limit, args):
                #  scale=1., mode='gen', render=False):
        self.dataloader = dataloader
        self.dataset = dataloader.dataset
        assert mm_num_samples < len(dataloader.dataset)
        self.max_motion_length = max_motion_length
        self.mode = args.eval_mode
        self.root_mode = args.root_mode
        self.render = args.render
        self.scale = args.guidance_param
        self.DATASET = Path('dataset/HumanML3D')
        self.aux_metrics = []

        normalizer = HumanMlNormalizer()
       
        # Load motion from files
        generated_motions = np.load(os.path.join(args.external_results_dir, 'benchmark_samples.npy'))  # [n_motions, 3[leader, folower, out], n_frames, dim]
        generated_motions = generated_motions.transpose(0, 1, 3, 2)[:, :, :, None]  # [n_motions, 3[leader, follower, out], dim, 1, n_frames]
        generated_motions = (generated_motions - dataloader.dataset.mean[None, None, :, None, None]) / dataloader.dataset.std[None, None, :, None, None]  # assuming data is unnormalized
        apear_mask = np.load(os.path.join(args.external_results_dir, 'benchmark_style_masks.npy'))  # [n_motions n_frames]
        struct_mask = np.load(os.path.join(args.external_results_dir, 'benchmark_content_masks.npy'))  # [n_motions n_frames]
        # mask = np.concatenate([struct_mask, apear_mask, struct_mask], axis=1)  # [n_motions, 3, 1, 1, n_frames]
        assert generated_motions.shape[0] == struct_mask.shape[0] == apear_mask.shape[0]
        n_samples = generated_motions.shape[0]

        # Retrieve texts
        with open(args.benchmark_path, 'r') as f:
            motion_id_pairs = json.load(f)
        assert len(motion_id_pairs) == n_samples
        struct_files = [Path(f'{f[0]}.txt') for f in motion_id_pairs]
        appear_files = [Path(f'{f[1]}.txt') for f in motion_id_pairs]
        struct_text, struct_tokens = self.load_text_tokens(struct_files)
        appear_text, appear_tokens = self.load_text_tokens(appear_files)


        # Construct batches
        real_num_batches = num_samples_limit // dataloader.batch_size + 1
        real_num_samples = real_num_batches * dataloader.batch_size
        print('real_num_batches', real_num_batches)

        self.generated_motion = []
        self.mm_generated_motion = []
        self.w_vectorizer = dataloader.dataset.w_vectorizer

        for sample_i in range(min(n_samples, real_num_samples)):
            sample_dict = {'motion': generated_motions[sample_i, -1].squeeze().transpose(1,0),
                        'length': torch.tensor(struct_mask[sample_i].sum()),
                        'caption': appear_text[sample_i],
                        'tokens': appear_tokens[sample_i],
                        'cap_len': len(appear_tokens[sample_i]),
                        }
            self.generated_motion.append(sample_dict)


            # calc similarity metrics
            cur_len = [struct_mask[sample_i].sum(), apear_mask[sample_i].sum()]
            out_proc = self.get_motion_proc(torch.tensor(generated_motions[sample_i, 2][None]))
            in_proc = self.get_motion_proc(torch.tensor(generated_motions[sample_i, :2]))
            cur_metrics = self.calc_metrics(out_proc, in_proc, cur_len)
            self.aux_metrics.append(cur_metrics)

            if self.render and sample_i < dataloader.batch_size:
                # eval_prompts = [prompt for prompt in prompts] # deepcopy(prompts)
                eval_prompts = ['', '', '']
                for k, v in cur_metrics.items():
                    idx_to_add = 0 if Transfer_LEGEND[0] in k else 1
                    _k = k.replace('follower_', '').replace('leader_', '')
                    eval_prompts[idx_to_add] += f' | {_k}={v:.2f}'
                    
                save_3motion(motions=torch.tensor(generated_motions[sample_i], device=dist_util.dev()).float().detach(), 
                             motion_name='{}_{}_{}'.format('external', motion_id_pairs[sample_i][0], motion_id_pairs[sample_i][0]),
                             prompts=eval_prompts,
                             normalizer=normalizer)
        
        
    def load_text_tokens(self, motion_files):
        tokens = []
        text = []
        for file in motion_files:
            text_file = self.DATASET / 'texts' / file
            with open(text_file, 'r') as f:
                data = f.read()
            _text, _tokens = data.split('#')[:2]
            text.append(_text)
            tokens.append(process_tokens(_tokens.split(' ')))
        return text, tokens


def process_tokens(tokens, max_text_len=20):
    if len(tokens) < max_text_len:
        # pad with "unk"
        tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
        sent_len = len(tokens)
        tokens = tokens + ['unk/OTHER'] * (max_text_len + 2 - sent_len)
    else:
        # crop
        tokens = tokens[:max_text_len]
        tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
        sent_len = len(tokens) 
    return tokens