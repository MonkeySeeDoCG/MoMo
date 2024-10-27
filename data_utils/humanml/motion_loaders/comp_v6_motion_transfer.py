import torch
from data_utils.humanml.networks.modules import *
from tqdm import tqdm
from utils import dist_util
from data_utils.humanml.motion_loaders.comp_v6_model_dataset import CompMDMGeneratedDataset
from sample.transfer import transfer
from data_utils.humanml.scripts.motion_process import recover_from_ric
from data_utils.humanml_utils import HumanMlNormalizer
import copy


Transfer_LEGEND = ['leader', 'follower']

class CompTransferGeneratedDataset(CompMDMGeneratedDataset):

    def __init__(self, model, diffusion, dataloader, mm_num_samples, mm_num_repeats, max_motion_length, num_samples_limit, args):

        self.dataloader = dataloader
        self.dataset = dataloader.dataset
        self.max_motion_length = max_motion_length
        self.mode = args.eval_mode
        self.render = args.render
        self.save = args.save
        self.scale = args.guidance_param

        if self.mode == 'inversion':
            print('WARNING - For inversion mode - forcing scale=1')
            self.scale=1.
                
        real_num_batches = len(dataloader)
        if num_samples_limit is not None:
            real_num_batches = num_samples_limit // dataloader.batch_size + 1
        print('real_num_batches', real_num_batches)

        generated_motion = []
        model.eval()

        self.aux_metrics = []

        with torch.no_grad():
            for i, (_, model_kwargs) in tqdm(enumerate(dataloader)):

                if num_samples_limit is not None and len(generated_motion) >= num_samples_limit:
                    break

                tokens = [t.split('_') for t in model_kwargs['y']['follower_tokens']]
                texts = model_kwargs['y']['follower_text']
                lengths = model_kwargs['y']['leader_lengths']


                cur_bs = model_kwargs['y']['leader_motion'].shape[0]
                
                # add CFG scale to batch
                if self.scale != 1.:
                    model_kwargs['y']['scale'] = torch.ones(cur_bs,
                                                            device=dist_util.dev()) * self.scale
                
                sample = []  # FIXME - for now, running example by example
                for sample_i in tqdm(range(cur_bs)):
                    cur_len = self.args_to_tupple(model_kwargs, sample_i, '_lengths')
                    cur_len = [motion_len.item() for motion_len in cur_len]
                    if args.eval_mode in ['gen', 'debug']:
                        prompts = self.args_to_tupple(model_kwargs, sample_i, '_text')
                        given_motion_kwargs = None
                    else:
                        prompts = [None, None]
                        given_motion_kwargs = {'y': {k: v[sample_i] for k, v in model_kwargs['y'].items()}} 
                    in_motions = None
                    in_motions = torch.stack(self.args_to_tupple(model_kwargs, sample_i, '_motion'), dim=0)
                    in_motions = in_motions.to(dist_util.dev())
                    transfer_args = copy.deepcopy(args)
                    vars(transfer_args).update({'text_leader': prompts[0], 'text_follower': [prompts[1]], 
                                                'given_motion_kwargs': given_motion_kwargs,
                                                'output_dir': None, 'motion_length': cur_len,
                                                'num_repetitions': 1, 'n_rows_in_out_file': 3, 
                                                'render': args.render and i==0, 'save': args.save and i==0,  # render/save only for 1st batch
                                                'show_progress': False})
                    _, all_outputs, _ = transfer(transfer_args, dataloader.dataset, model, diffusion)
                    
                    if self.mode == 'gen' or self.mode == 'debug':
                        in_motions = all_outputs[:2]
                    
                    transfer_output = all_outputs[-1:]

                    # calc similarity metrics
                    transfer_proc = self.get_motion_proc(transfer_output)
                    in_proc = self.get_motion_proc(in_motions)
                    cur_metrics = self.calc_metrics(transfer_proc, in_proc, cur_len)
                    self.aux_metrics.append(cur_metrics)
                    sample.append(transfer_output)
                

                # outputs might differ in frame count, as their length is determined by the leader motion. 
                # must convert all outputs to having the same frame count. Otherwise, reading a batch (in evaluate_matching_score_negative_samples()) crashes
                max_frames = max([out.shape[-1] for out in sample])
                equal_length_sample = np.zeros((len(sample),) + sample[0].shape[1:3] + (max_frames,))
                for i, s in enumerate(sample):
                    equal_length_sample[i, :, :, :s.shape[-1]] = s

                sub_dicts = [{'motion': equal_length_sample[bs_i].squeeze().transpose(1,0),
                            'length': lengths[bs_i],
                            'caption': texts[bs_i],
                            'tokens': tokens[bs_i],
                            'cap_len': len(tokens[bs_i]),
                            } for bs_i in range(cur_bs)]
                generated_motion += sub_dicts

        self.generated_motion = generated_motion
        self.mm_generated_motion = []  # created to please general classes which expect this field
        self.w_vectorizer = dataloader.dataset.w_vectorizer

    def args_to_tupple(self, model_kwargs, sample_i, suffix):
        tupple = [model_kwargs['y'][f'{id}{suffix}'][sample_i] for id in ['leader', 'follower']]
        return tupple
        
    # root_rot_velocity (B, seq_len, 1)
    # root_linear_velocity (B, seq_len, 2)
    # root_y (B, seq_len, 1)
    # ric_data (B, seq_len, (joint_num - 1)*3)
    # rot_data (B, seq_len, (joint_num - 1)*6)
    # local_velocity (B, seq_len, joint_num*3)
    # foot contact (B, seq_len, 4)
    def get_motion_proc(self, motion):
        n_joints = self.dataloader.dataset.n_joints
        torch_motion = torch.from_numpy(motion).to(dist_util.dev()) if not isinstance(motion, torch.Tensor) else motion
        unnormed_motion = self.dataset.t2m_dataset.inv_transform(torch_motion.cpu().permute(0, 2, 3, 1)).float()
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