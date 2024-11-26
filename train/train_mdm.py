# This code is based on https://github.com/openai/guided-diffusion
"""
Train a diffusion model on images.
"""

import os
import json
from utils.misc import fixseed
from utils.parser_util import train_args
from utils import dist_util
from train.training_loop import TrainLoop
from data_utils.get_data import get_dataset_loader
from utils.model_util import create_model_and_diffusion
from utils.ml_platforms import ClearmlPlatform, TensorboardPlatform, NoPlatform, WandBPlatform  # required for the eval operation

def main():
    args = train_args()
    fixseed(args.seed)
    ml_platform_type = eval(args.ml_platform_type)
    ml_platform = ml_platform_type(save_dir=args.save_dir)
    ml_platform.report_args(args, name='Args')

    if args.save_dir is None:
        raise FileNotFoundError('save_dir was not specified.')
    elif os.path.exists(args.save_dir) and not args.overwrite:
        raise FileExistsError('save_dir [{}] already exists.'.format(args.save_dir))
    elif not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    args_path = os.path.join(args.save_dir, 'args.json')
    with open(args_path, 'w') as fw:
        json.dump(vars(args), fw, indent=4, sort_keys=True)

    dist_util.setup_dist(args.device)

    print("creating data loader...")
    data = get_dataset_loader(name=args.dataset, batch_size=args.batch_size, num_frames=args.num_frames, 
                              datapath=args.data_dir, pose_rep=args.repr)

    print("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(args, data.dataset)
    model.to(dist_util.dev())
    # model.rot2xyz.smpl_model.eval()

    print('Total params: %.2fM' % (sum(p.numel() for p in model.parameters_wo_clip()) / 1000000.0))
    print("Training...")
    TrainLoop(args, ml_platform, model, diffusion, data).run_loop()
    ml_platform.close()

if __name__ == "__main__":
    main()
