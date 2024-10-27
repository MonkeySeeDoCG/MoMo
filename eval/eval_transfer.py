from utils.parser_util import evaluation_parser
from utils.misc import fixseed
from data_utils.humanml.motion_loaders.model_motion_loaders import get_mdm_loader
from data_utils.humanml.utils.metrics import *
from data_utils.humanml.networks.evaluator_wrapper import EvaluatorMDMWrapper
from data_utils.humanml.scripts.motion_process import *
from data_utils.humanml.utils.utils import *
from utils.model_util import create_model_and_diffusion, load_model_wo_clip

from diffusion import logger
from utils import dist_util
from data_utils.get_data import get_dataset_loader
from model.cfg_sampler import ClassifierFreeSampleModel
from utils.ml_platforms import ClearmlPlatform, TensorboardPlatform, NoPlatform, WandBPlatform  # required for the eval operation
from sample.transfer import get_transfer_args, init_main

from eval.eval_humanml import *

torch.multiprocessing.set_sharing_strategy('file_system')


if __name__ == '__main__':
    args = evaluation_parser()
    ml_platform_type = eval(args.ml_platform_type)
    ml_platform = ml_platform_type(name=args.eval_name)
    ml_platform.report_args(args, name='Args')

    assert args.model_path is not None and args.benchmark_path is not None, 'model_path and benchmark_path must be provided'

    args.batch_size = 32 # This must be 32! Don't change it! otherwise it will cause a bug in R precision calc!
    
    niter = os.path.basename(args.model_path).replace('model', '').replace('.pt', '')
    log_file = os.path.join(os.path.dirname(args.model_path), 'eval_transfer_{}_seed{}'.format(niter, args.seed))
    if args.guidance_param != 1.:
        log_file += f'_gscale{args.guidance_param}'
    log_file += f'_{args.eval_mode}'
    log_file += '.log'

    print(f'Will save to log file [{log_file}]')

    print(f'Eval mode [{args.eval_mode}]')
    if args.eval_mode in ['inversion', 'gen', 'upper_body']:
        num_samples_limit = args.samples_limit
    elif args.eval_mode == 'debug':
        num_samples_limit = args.batch_size # a single batch
    else:
        raise ValueError()
    
    diversity_times = 300
    replication_times = 1  # motion transfer with inversion is (almost) deterministic, no need for repetitions.

    # dist_util.setup_dist(args.device)
    logger.configure()

    logger.log("creating data loader...")
    split = 'test'
    unfiltered_gt_loader = get_dataset_loader(name=args.dataset, batch_size=args.batch_size, num_frames=None, split=split, hml_mode='gt')
    gt_loader = get_dataset_loader(name=args.dataset, batch_size=args.batch_size, num_frames=None, split=split, hml_mode='gt', filter_path=args.benchmark_path)
    gen_loader = get_dataset_loader(name=args.dataset, batch_size=args.batch_size, num_frames=None, split=split, hml_mode='eval', benchmark_path=args.benchmark_path)
    
    get_feat_idx, transfer_idx = get_transfer_args(args)
    additional_model_args = {'get_feat_idx': get_feat_idx, 'transfer_idx': transfer_idx}    
    _, model, diffusion = init_main(args, additional_model_args, data=gen_loader.dataset)   

    eval_motion_loaders = {
        ################
        ## HumanML3D Dataset##
        ################
        'vald': lambda: get_mdm_loader(
            model, diffusion, args.batch_size, gen_loader, 
            mm_num_samples=0, mm_num_repeats=0, max_motion_length=gt_loader.dataset.opt.max_motion_length, num_samples_limit=num_samples_limit,
            args=args, is_transfer=True,
        )
    }

    eval_wrapper = EvaluatorMDMWrapper(args.dataset, dist_util.dev())
    evaluation(eval_wrapper, gt_loader, eval_motion_loaders, log_file, 
               replication_times, diversity_times, mm_num_times=0, 
               run_mm=False, run_div=False, logger=ml_platform,
               unfiltered_gt_loader=unfiltered_gt_loader)
