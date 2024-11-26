# This code is based on https://github.com/openai/guided-diffusion
"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""
import os
import numpy as np
from utils.parser_util import generate_args
import data_utils.humanml.utils.paramUtil as paramUtil
from utils.visualize import save_multiple_samples
from sample.sample_utils import init_main, get_niter, get_sample_vars, save_results, sample_motions, prepare_plot, get_xyz_rep


def main():
    args = generate_args()

    # this block must be called BEFORE the dataset is loaded
    texts = None
    if args.text_prompt != '':
        texts = [args.text_prompt]
        args.num_samples = 1
    elif args.input_text != '':
        assert os.path.exists(args.input_text)
        with open(args.input_text, 'r') as fr:
            texts = fr.readlines()
        texts = [s.replace('\n', '') for s in texts]
        args.num_samples = len(texts)

    data, model, diffusion = init_main(args)   
    
    return generate(args, data, model, diffusion, texts)


def generate(args, data, model, diffusion, texts = None):

    args.num_samples = min(args.num_samples, len(data))
    is_using_data = not any([args.input_text, args.text_prompt])
    out_path, shape, model_kwargs, max_frames = get_sample_vars(args, data, model, texts, get_out_name=get_out_name, 
                                                                is_using_data=is_using_data)

    all_motions, all_lengths, all_text = sample_motions(args, model, shape, model_kwargs, max_frames, 
                                                        init_noise=None, sample_func=diffusion.p_sample_loop)

    # get xyz abs locations
    all_motions = get_xyz_rep(data, all_motions)

    save_results(args, out_path, all_motions, all_lengths, all_text)

    visualize_motions(args, out_path, max_frames, all_motions, all_lengths, all_text, data.fps)

    return out_path

def visualize_motions(args, out_path, max_frames, all_motions, all_lengths, all_text, fps):
    # print(f"saving visualizations to {out_path}...")
    
    # get kinematic chain
    kinematic_chain = paramUtil.t2m_kinematic_chain

    # get xyz samples
    all_file_template = 'samples_{:02d}_to_{:02d}.mp4'
    animations = np.empty(shape=(args.num_samples, args.num_repetitions), dtype=object)

    for sample_i in range(args.num_samples):
        for rep_i in range(args.num_repetitions):
            # caption = all_text[rep_i*args.batch_size + sample_i]
            # length = all_lengths[rep_i*args.batch_size + sample_i]
            # motion = all_motions[rep_i*args.batch_size + sample_i]
            # motion = motion.transpose(2, 0, 1)  # [:length]
            # motion[length:-1] = motion[length-1]  # duplicate the last frame to end of motion, so all motions will be in equal length
            # animations[sample_i, rep_i] = plot_3d_motion(kinematic_chain, motion, dataset=args.dataset, title=caption, fps=fps)
            animations[sample_i, rep_i] = prepare_plot(sample_i, rep_i, args, fps, all_motions, all_text, all_lengths, kinematic_chain)

    save_multiple_samples(out_path, {'all': all_file_template}, animations, fps, max_frames, 
                          n_rows_in_out_file=getattr(args, 'n_rows_in_out_file', 3))

    abs_path = os.path.abspath(out_path)
    print(f'[Done] Results are at [{abs_path}]')


def get_out_name(args):
    out_path = args.output_dir
    if out_path == '':
        niter = get_niter(args.model_path)
        out_name = 'samples_{}_seed{}'.format(niter, args.seed)
        if args.text_prompt != '':
            out_name += '_' + args.text_prompt.replace(' ', '_').replace('.', '')
        elif args.input_text != '':
            out_name += '_' + os.path.basename(args.input_text).replace('.txt', '').replace(' ', '_').replace('.', '')
    return out_name


if __name__ == "__main__":
    main()
