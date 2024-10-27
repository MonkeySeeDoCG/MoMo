from argparse import ArgumentParser
import argparse
import os
import json
import copy


def parse_and_load_from_model(parser):
    # args according to the loaded model
    # do not specify them from cmd line since they will be overwritten
    add_data_options(parser)
    add_model_options(parser)
    add_diffusion_options(parser)
    args = parser.parse_args()
    args_to_overwrite = []
    for group_name in ['dataset', 'model', 'diffusion']:
        args_to_overwrite += get_args_per_group_name(parser, args, group_name)
    
    # load args from model
    args = extract_args(copy.deepcopy(args), args_to_overwrite, args.model_path)
    return args
        
def extract_args(args, args_to_overwrite, model_path):
    args_path = os.path.join(os.path.dirname(model_path), 'args.json')
    assert os.path.exists(args_path), 'Arguments json file was not found!'
    with open(args_path, 'r') as fr:
        model_args = json.load(fr)

    for a in args_to_overwrite:
        if a in model_args.keys():
            setattr(args, a, model_args[a])
        else:
            print('Warning: was not able to load [{}], using default value [{}] instead.'.format(a, args.__dict__[a]))

    if args.cond_mask_prob == 0:
        args.guidance_param = 1
        
    # backward compatibility
    if isinstance(args.emb_trans_dec, bool):
        if args.emb_trans_dec:
            args.emb_trans_dec = 'cls_tcond_cross_tcond'
        else: 
            args.emb_trans_dec = 'cls_none_cross_tcond'
    return args


def get_args_per_group_name(parser, args, group_name):
    for group in parser._action_groups:
        if group.title == group_name:
            group_dict = {a.dest: getattr(args, a.dest, None) for a in group._group_actions}
            return list(argparse.Namespace(**group_dict).__dict__.keys())
    return ValueError('group_name was not found.')


def add_base_options(parser):
    group = parser.add_argument_group('base')
    group.add_argument("--cuda", default=True, type=bool, help="Use cuda device, otherwise use CPU.")
    group.add_argument("--device", default=0, type=int, help="Device id to use.")
    group.add_argument("--seed", default=10, type=int, help="For fixing random seed.")
    group.add_argument("--batch_size", default=64, type=int, help="Batch size during training.")

    group.add_argument("--ml_platform_type", default='NoPlatform', choices=['NoPlatform', 'ClearmlPlatform', 'TensorboardPlatform', 'WandBPlatform'], type=str,
                    help="Choose platform to log results. NoPlatform means no logging.")


def add_diffusion_options(parser):
    group = parser.add_argument_group('diffusion')
    group.add_argument("--noise_schedule", default='cosine', choices=['linear', 'cosine'], type=str,
                       help="Noise schedule type")
    group.add_argument("--diffusion_steps", default=1000, type=int,
                       help="Number of diffusion steps (denoted T in the paper)")
    group.add_argument("--sigma_small", default=True, type=bool, help="Use smaller sigma values.")


def add_model_options(parser):
    group = parser.add_argument_group('model')
    group.add_argument("--arch", default='trans_enc',
                       choices=['trans_enc', 'trans_dec', 'gru'], type=str,
                       help="Architecture types as reported in the paper.")
    group.add_argument("--text_encoder_type", default='clip',
                       choices=['clip', 'bert'], type=str, help="Text encoder type.")
    group.add_argument("--emb_trans_dec", default='cls_none_cross_tcond', 
                       choices=['cls_none_cross_tcond', 'cls_t_cross_tcond', 'cls_t_cross_cond', 'cls_tcond_cross_tcond', 'cls_tcond_cross_cond'],
                       help="For trans_dec architecture only, if true, will inject condition as a class token"
                            " (in addition to cross-attention).")
    group.add_argument("--layers", default=8, type=int,
                       help="Number of layers.")
    group.add_argument("--latent_dim", default=512, type=int,
                       help="Transformer/GRU width.")
    group.add_argument("--cond_mask_prob", default=.1, type=float,
                       help="The probability of masking the condition during training."
                            " For classifier-free guidance learning.")
    group.add_argument("--mask_frames", action='store_true', help="If true, will fix Rotem's bug and mask invalid frames.")
    group.add_argument("--root_sep_io", action='store_true',
                       help="Root joint get separated input/output process (since the data is different). Relevant only for mat representation.")
    group.add_argument("--frame_averaging", default='no',
                       help="Options are `no`, `all` (i.e. into a single vector), or `int` indicating the number of frames avraged into a single vector.")
    group.add_argument("--lambda_rcxyz", default=0.0, type=float, help="Joint positions loss.")
    group.add_argument("--lambda_vel", default=0.0, type=float, help="Joint velocity loss.")
    group.add_argument("--lambda_fc", default=0.0, type=float, help="Foot contact loss.")
    group.add_argument("--unconstrained", action='store_true',
                       help="Model is trained unconditionally. That is, it is constrained by neither text nor action. "
                            "Currently tested on HumanAct12 only.")
    group.add_argument("--emb_before_mask", action='store_true',
                       help="If true - for the cond branch - flip between mask and linear blocks (as described in Fig.2).")
    group.add_argument("--agg_features", action='store_true',
                       help="Run attention on all joints together (when attending frames), or on all frames together (when attending joints).")


def add_data_options(parser):
    group = parser.add_argument_group('dataset')
    group.add_argument("--dataset", default='humanml', choices=['humanml', 'kit', 'humanact12', 'uestc', 'bvh_general'], type=str,
                       help="Dataset name (choose from list).")
    group.add_argument("--repr", default='rot6d', choices=['quat', 'rot6d'], type=str,
                       help="Motion representation (choose from list).")
    group.add_argument("--data_dir", default=None, type=str,
                       help="If empty, will use defaults according to the specified dataset.")


def add_training_options(parser):
    group = parser.add_argument_group('training')
    group.add_argument("--save_dir", required=True, type=str,
                       help="Path to save checkpoints and results.")
    group.add_argument("--overwrite", action='store_true',
                       help="If True, will enable to use an already existing save_dir.")
    group.add_argument("--lr", default=1e-4, type=float, help="Learning rate.")
    group.add_argument("--weight_decay", default=0.0, type=float, help="Optimizer weight decay.")
    group.add_argument("--lr_anneal_steps", default=0, type=int, help="Number of learning rate anneal steps.")
    group.add_argument("--eval_batch_size", default=32, type=int,
                       help="Batch size during evaluation loop. Do not change this unless you know what you are doing. "
                            "T2m precision calculation is based on fixed batch size 32.")
    group.add_argument("--eval_split", default='test', choices=['val', 'test'], type=str,
                       help="Which split to evaluate on during training.")
    group.add_argument("--eval_during_training", action='store_true',
                       help="If True, will run evaluation during training.")
    group.add_argument("--eval_rep_times", default=3, type=int,
                       help="Number of repetitions for evaluation loop during training.")
    group.add_argument("--eval_num_samples", default=1_000, type=int,
                       help="If -1, will use all samples in the specified split.")
    group.add_argument("--log_interval", default=1_000, type=int,
                       help="Log losses each N steps")
    group.add_argument("--save_interval", default=50_000, type=int,
                       help="Save checkpoints and run evaluation each N steps")
    group.add_argument("--num_steps", default=600_000, type=int,
                       help="Training will stop after the specified number of steps.")
    group.add_argument("--num_frames", default=60, type=int,
                       help="Limit for the maximal number of frames. In HumanML3D and KIT this field is ignored.")
    group.add_argument("--resume_checkpoint", default="", type=str,
                       help="If not empty, will start from the specified checkpoint (path to model###.pt file).")
    group.add_argument("--use_ema", action='store_true',
                       help="If True, will use EMA model averaging.")
    group.add_argument("--avg_model_beta", default=0.9999, type=float, help="Average model beta.")
    group.add_argument("--adam_beta2", default=0.999, type=float, help="Adam beta2.")
    group.add_argument("--gen_during_training", action='store_true',
                       help="If True, will generate motions during training, on each save interval.")
    group.add_argument("--gen_num_samples", default=3, type=int,
                       help="Number of samples to sample while generating")
    group.add_argument("--gen_num_repetitions", default=2, type=int,
                       help="Number of repetitions, per sample (text prompt/action)")
    group.add_argument("--gen_guidance_param", default=2.5, type=float,
                       help="For classifier-free sampling - specifies the s parameter, as defined in the paper.")

def add_sampling_options(parser):
    group = parser.add_argument_group('sampling')
    group.add_argument("--model_path", required=True, type=str,
                       help="Path to model####.pt file to be sampled.")
    group.add_argument("--output_dir", default='', type=str,
                       help="Path to results dir (auto created by the script). "
                            "If empty, will create dir in parallel to checkpoint.")
    group.add_argument("--num_samples", default=10, type=int,
                       help="Maximal number of prompts to sample, "
                            "if loading dataset from file, this field will be ignored.")
    group.add_argument("--num_repetitions", default=3, type=int,
                       help="Number of repetitions, per sample (text prompt/action)")
    group.add_argument("--guidance_param", default=2.5, type=float,
                       help="For classifier-free sampling - specifies the s parameter, as defined in the paper.")
    group.add_argument("--keywords", default='', type=str,
                       help="Coma separated for attention maps. (sample/gen_attention)")
    group.add_argument("--rank_coloring", action='store_true', help="")
    group.add_argument("--overwrite", action='store_true',
                       help="VIS FEATURES ONLY: If True and directory already exists, will delete it and recreate with the output of this run.")
    group.add_argument("--resume", action='store_true',
                       help="VIS FEATURES ONLY: If True and directory already exists, will proceed from where previous run has stopped, "
                            "i.e., create samples that were not created yet.")
    group.add_argument("--ddim", action='store_true',
                       help="VIS FEATURES ONLY: apply ddim instead of ddpm.")
    group.add_argument("--motion_length", default=6.0, type=float,
                       help="The length of the sampled motion [in seconds]. "
                            "Maximum is 9.8 for HumanML3D (text-to-motion), and 2.0 for HumanAct12 (action-to-motion)")
    group.add_argument("--n_rows_in_out_file", default=3, type=int,
                       help="Number of motion rows in the output mp4 file.")


def add_generate_options(parser):
    group = parser.add_argument_group('generate')
    group.add_argument("--input_text", default='', type=str,
                       help="Path to a text file lists text prompts to be synthesized. If empty, will take text prompts from dataset.")
    group.add_argument("--action_file", default='', type=str,
                       help="Path to a text file that lists names of actions to be synthesized. Names must be a subset of dataset/uestc/info/action_classes.txt if sampling from uestc, "
                            "or a subset of [warm_up,walk,run,jump,drink,lift_dumbbell,sit,eat,turn steering wheel,phone,boxing,throw] if sampling from humanact12. "
                            "If no file is specified, will take action names from dataset.")
    group.add_argument("--text_prompt", default='', type=str,
                       help="A text prompt to be generated. If empty, will take text prompts from dataset.")
    group.add_argument("--action_name", default='', type=str,
                       help="An action name to be generated. If empty, will take text prompts from dataset.")


def add_vis_feat_options(parser):
    group = parser.add_argument_group('vis_feat')
    group.add_argument("--vis_type", default=['pca', 'text', 'corresp'], choices=['pca', 'text', 'corresp'], type=str, nargs='+',
                       help="Type of visualization")
    group.add_argument("--text_prompt", default='', type=str, nargs='+',
                       help="A text prompt to be generated. If empty, will take text prompts from dataset.")
    group.add_argument("--num_corresp_frames", default=3, type=int,
                       help="Number of frames against which to visualize correspondence.")


def add_transfer_options(parser):
    group = parser.add_argument_group('transfer')
    group.add_argument("--text_leader", default=None, type=str,
                       help="The prompt that describes the leader motion.")
    group.add_argument("--text_follower", default=[], type=str, nargs='*',
                       help="The prompt that describes the follower motion.")
    group.add_argument("--leader_motion_path", default=None, type=str,
                       help="The leader motion to inverse - in npy format.")
    group.add_argument("--follower_motion_path", default=[], type=str, nargs='*',
                       help="The follower motion to inverse - in npy format.")
    group.add_argument("--n_follower_mult", default=1, type=int,
                       help="Make the follower motion by synthesizing it several times. The parameter tells us how many time to synthesize it.")
    group.add_argument("--transfer_layers_start", default=1, type=int,
                       help="A range of layers *start* to transfer")
    group.add_argument("--transfer_layers_end", default=-1, type=int,
                       help="A range of layers *end* to transfer")
    group.add_argument("--transfer_diff_step_start", default=10, type=int,
                       help="A range of diffusion steps *start* to transfer")
    group.add_argument("--transfer_diff_step_end", default=-10, type=int,
                       help="A range of diffusion steps *end* to transfer")
    group.add_argument("--transfer_diff_step_step", default=1, type=int,
                       help="A range of diffusion steps *step* to transfer")
    group.add_argument("--assign_root_rot", action='store_true', help="If true, will assign the root rotation to the output motion.")


def add_evaluation_options(parser):
    group = parser.add_argument_group('eval')
    group.add_argument("--model_path", required=True, type=str,
                       help="Path to model####.pt file to be sampled.")
    group.add_argument("--eval_name", default='unnamed_experiment', type=str, help="Optional for wandb. if empty will use the model name instead.")
    group.add_argument("--eval_mode", default='gen', choices=['gen', 'inversion', 'debug'], type=str,
                       help="gen - generate from text; "
                            "inversion - invert real dataset motions; "
                            "debug - short run, less accurate results.")
    group.add_argument("--guidance_param", default=2.5, type=float,
                       help="For classifier-free sampling - specifies the s parameter, as defined in the paper.")
    group.add_argument("--render", action='store_true', help="If true, will render mp4 for the first batch.")
    group.add_argument("--save", action='store_true', help="If true, will store npy files for the first batch.")
    group.add_argument("--benchmark_path", default=None, type=str,
                    help="Path to csv file defining the Motion Transfer Benchmark.")
    group.add_argument("--samples_limit", default=512, type=int,
                    help="")
    

def train_args():
    parser = ArgumentParser()
    add_base_options(parser)
    add_data_options(parser)
    add_model_options(parser)
    add_diffusion_options(parser)
    add_training_options(parser)
    return parser.parse_args()


def generate_args():
    parser = ArgumentParser()
    # args specified by the user: (all other will be loaded from the model)
    add_base_options(parser)
    add_sampling_options(parser)
    add_generate_options(parser)
    return  parse_and_load_from_model(parser)


def transfer_args():
    parser = ArgumentParser()
    # args specified by the user: (all other will be loaded from the model)
    add_base_options(parser)
    add_sampling_options(parser)
    add_transfer_options(parser)
    return parse_and_load_from_model(parser)


def vis_feat_args():
    parser = ArgumentParser()
    # args specified by the user: (all other will be loaded from the model)
    add_base_options(parser)
    add_sampling_options(parser)
    add_vis_feat_options(parser)
    return parse_and_load_from_model(parser)


def evaluation_parser():
    parser = ArgumentParser()
    # args specified by the user: (all other will be loaded from the model)
    add_base_options(parser)
    add_evaluation_options(parser)
    add_transfer_options(parser)
    return parse_and_load_from_model(parser)
