import copy
import functools
import os
import time
from types import SimpleNamespace
import numpy as np

import re
from os.path import join as pjoin
from typing import Optional

import blobfile as bf
import torch
from torch.optim import AdamW

from diffusion import logger
from utils import dist_util
from diffusion.fp16_util import MixedPrecisionTrainer
from diffusion.resample import LossAwareSampler, UniformSampler
from tqdm import tqdm
from diffusion.resample import create_named_schedule_sampler
from data_utils.humanml.networks.evaluator_wrapper import EvaluatorMDMWrapper
from eval import eval_humanml
from sample.generate import generate
from data_utils.get_data import get_dataset_loader
from utils.model_util import load_model_wo_clip
import sys, signal  # for slurm recovery


class TrainLoop:
    def __init__(self, args, ml_platform, model, diffusion, data):
        self.args = args
        self.dataset = args.dataset
        self.ml_platform = ml_platform
        self.model = model
        self.model_avg = None
        if self.args.use_ema:
            self.model_avg = copy.deepcopy(self.model)
        self.model_for_eval = self.model_avg if self.args.use_ema else self.model
        self.diffusion = diffusion
        self.cond_mode = model.cond_mode
        self.data = data
        self.batch_size = args.batch_size
        self.microbatch = args.batch_size  # deprecating this option
        self.lr = args.lr
        self.log_interval = args.log_interval
        self.save_interval = args.save_interval
        self.resume_checkpoint = args.resume_checkpoint
        self.use_fp16 = False  # deprecating this option
        self.fp16_scale_growth = 1e-3  # deprecating this option
        self.weight_decay = args.weight_decay
        self.lr_anneal_steps = args.lr_anneal_steps

        self.step = 0
        self.resume_step = 0
        self.global_batch = self.batch_size # * dist.get_world_size()
        self.num_steps = args.num_steps
        self.sync_cuda = torch.cuda.is_available()

        self._load_and_sync_parameters()
        self.mp_trainer = MixedPrecisionTrainer(
            model=self.model,
            use_fp16=self.use_fp16,
            fp16_scale_growth=self.fp16_scale_growth,
        )

        self.save_dir = args.save_dir
        self.overwrite = args.overwrite

        if self.args.use_ema:
            self.opt = AdamW(
                # with amp, we don't need to use the mp_trainer's master_params
                (self.model.parameters()
                 if self.use_fp16 else self.mp_trainer.master_params),
                lr=self.lr,
                weight_decay=self.weight_decay,
                betas=(0.9, self.args.adam_beta2),
            )
        else:
            self.opt = AdamW(
                self.mp_trainer.master_params, lr=self.lr, weight_decay=self.weight_decay
            )

        if self.resume_step:
            self._load_optimizer_state()
            # Model was resumed, either due to a restart or a checkpoint
            # being specified at the command line.

        self.device = torch.device("cpu")
        if torch.cuda.is_available() and dist_util.dev() != 'cpu':
            self.device = torch.device(dist_util.dev())

        self.schedule_sampler_type = 'uniform'
        self.schedule_sampler = create_named_schedule_sampler(self.schedule_sampler_type, diffusion)
        self.eval_wrapper, self.eval_data, self.eval_gt_data = None, None, None
        if args.dataset in ['kit', 'humanml'] and args.eval_during_training:
            mm_num_samples = 0  # mm is super slow hence we won't run it during training
            mm_num_repeats = 0  # mm is super slow hence we won't run it during training
            gen_loader = get_dataset_loader(name=args.dataset, batch_size=args.eval_batch_size, num_frames=None,
                                            split=args.eval_split,
                                            hml_mode='eval')

            self.eval_gt_data = get_dataset_loader(name=args.dataset, batch_size=args.eval_batch_size, num_frames=None,
                                                   split=args.eval_split,
                                                   hml_mode='gt')
            self.eval_wrapper = EvaluatorMDMWrapper(args.dataset, dist_util.dev())
            eval_args = SimpleNamespace(
                scale=1.,
            )
            self.eval_data = {
                'test': lambda: eval_humanml.get_mdm_loader(
                    self.model_for_eval, diffusion, args.eval_batch_size,
                    gen_loader, mm_num_samples, mm_num_repeats, gen_loader.dataset.opt.max_motion_length,
                    args.eval_num_samples, args=eval_args,
                )
            }
        self.use_ddp = False
        self.ddp_model = self.model
        
        signal.signal(signal.SIGUSR1, self.listen_to_signal_and_save)

    def listen_to_signal_and_save(self, signal_number, frame):      
        '''Save the model before task is killed... Relevant when using slurm.
        
        Listen to specific signal in my case SIGUSR1 since its not used by the system. on signal,
        save the model, exit Wandb and return exit code 16. Here is what's needed to be done:
        - Add a line on __init__() to listen to that signal `signal.signal(signal.SIGUSR1, self.listen_to_signal_and_save)`
        - Ask slurm to send the signal USR1 before ending.
        - Run python code (in slurm) using srun otherwise signals sent to the slurm script won't be forwarded to python.
        - Catch exit code on slurm script, and requeue your job :)
        '''
        
        print(f'Received signal number {signal_number}')
        if signal_number == signal.SIGUSR1:
            print(f'SIGUSR1 recieved after step {self.total_step()}')
            self.save()
            self.ml_platform.close()
            print('Platform closed, exiting...')
            sys.exit(signal.SIGUSR1)
            
    def _load_and_sync_parameters(self):
        self.resume_checkpoint = self.resume_checkpoint or self.find_resume_checkpoint()

        if self.resume_checkpoint:
            self.resume_step = parse_resume_step_from_filename(self.resume_checkpoint)
            logger.log(f"loading model from checkpoint: {self.resume_checkpoint}...")

            state_dict = dist_util.load_state_dict(
                self.resume_checkpoint, map_location=dist_util.dev())

            if 'model_avg' in state_dict:
                print('loading both model and model_avg')
                state_dict, state_dict_avg = state_dict['model'], state_dict[
                    'model_avg']
                load_model_wo_clip(self.model, state_dict)
                load_model_wo_clip(self.model_avg, state_dict_avg)
            else:
                load_model_wo_clip(self.model, state_dict)
                if self.args.use_ema:
                    # in case we load from a legacy checkpoint, just copy the model
                    print('loading model_avg from model')
                    self.model_avg.load_state_dict(self.model.state_dict())

    def _load_optimizer_state(self):
        main_checkpoint = self.resume_checkpoint  # was: self.find_resume_checkpoint() or self.resume_checkpoint
        opt_checkpoint = bf.join(
            bf.dirname(main_checkpoint), f"opt{self.resume_step:09}.pt"
        )
        if bf.exists(opt_checkpoint):
            logger.log(f"loading optimizer state from checkpoint: {opt_checkpoint}")
            state_dict = dist_util.load_state_dict(
                opt_checkpoint, map_location=dist_util.dev()
            )

            if self.use_fp16:
                if 'scaler' not in state_dict:
                    print("scaler state not found ... not loading it.")
                else:
                    # load grad scaler state
                    self.scaler.load_state_dict(state_dict['scaler'])
                    # for the rest
                    state_dict = state_dict['opt']

            tgt_wd = self.opt.param_groups[0]['weight_decay']
            print('target weight decay:', tgt_wd)
            self.opt.load_state_dict(state_dict)
            print('loaded weight decay (will be replaced):',
                  self.opt.param_groups[0]['weight_decay'])
            # preserve the weight decay parameter
            for group in self.opt.param_groups:
                group['weight_decay'] = tgt_wd

    def run_loop(self):
        print('train steps:', self.num_steps)
        while self.total_step() < self.num_steps:
            print(f'Starting a new epoch at step {self.total_step()}')
            for motion, cond in tqdm(self.data):
                if not (not self.lr_anneal_steps or self.total_step() < self.lr_anneal_steps):
                    break

                motion = motion.to(self.device)
                cond['y'] = {key: val.to(self.device) if torch.is_tensor(val) else val for key, val in cond['y'].items()}

                self.run_step(motion, cond)
                if self.total_step() % self.log_interval == 0:
                    for k,v in logger.get_current().dumpkvs().items():
                        if k == 'loss':
                            print('step[{}]: loss[{:0.5f}]'.format(self.total_step(), v))

                        if k in ['step', 'samples'] or '_q' in k:
                            continue
                        else:
                            self.ml_platform.report_scalar(name=k, value=v, iteration=self.total_step(), group_name='Loss')

                if self.total_step() % self.save_interval == 0 and self.total_step() != 0 or self.total_step() == self.num_steps - 1:
                    self.save()
                    self.model.eval()
                    self.evaluate()
                    self.generate_during_training()
                    self.model.train()

                    # Run for a finite amount of time in integration tests.
                    if os.environ.get("DIFFUSION_TRAINING_TEST", "") and self.step > 0:
                        return
                self.step += 1
                
                # if we had enough steps, exit loop without completing the epoch
                if self.total_step() == self.num_steps:
                    break
                
            if not (not self.lr_anneal_steps or self.total_step() < self.lr_anneal_steps):
                break

    def total_step(self):
        total_step = self.step
        if self.resume_step:
            # we add 1 because self.resume_step has already been done and we don't want to run it again
            # in particular we don't want to run the evaluation and generation again
            total_step += self.resume_step + 1
        return total_step

    def evaluate(self):
        if not self.args.eval_during_training:
            return
        start_eval = time.time()
        if self.eval_wrapper is not None:
            print('Running evaluation loop: [Should take about 90 min]')
            log_file = os.path.join(self.save_dir, f'eval_humanml_{(self.total_step()):09d}.log')
            diversity_times = 300
            mm_num_times = 0  # mm is super slow hence we won't run it during training
            eval_dict = eval_humanml.evaluation(
                self.eval_wrapper, self.eval_gt_data, self.eval_data, log_file,
                replication_times=self.args.eval_rep_times, diversity_times=diversity_times, mm_num_times=mm_num_times, run_mm=False)
            print(eval_dict)
            for k, v in eval_dict.items():
                if k.startswith('R_precision'):
                    for i in range(len(v)):
                        self.ml_platform.report_scalar(name=f'top{i + 1}_' + k, value=v[i],
                                                          iteration=self.total_step(),
                                                          group_name='Eval')
                else:
                    self.ml_platform.report_scalar(name=k, value=v, iteration=self.total_step(),
                                                      group_name='Eval')

        end_eval = time.time()
        print(f'Evaluation time: {round(end_eval-start_eval)/60}min')

    def run_step(self, batch, cond):
        self.forward_backward(batch, cond)
        self.mp_trainer.optimize(self.opt)
        self.update_average_model()
        self._anneal_lr()
        self.log_step()

    def update_average_model(self):
        # update the average model using exponential moving average
        if self.args.use_ema:
            # master params are FP32
            params = self.model.parameters(
            ) if self.use_fp16 else self.mp_trainer.master_params
            for param, avg_param in zip(params, self.model_avg.parameters()):
                # avg = avg + (param - avg) * (1 - alpha)
                # avg = avg + param * (1 - alpha) - (avg - alpha * avg)
                # avg = alpha * avg + param * (1 - alpha)
                avg_param.data.mul_(self.args.avg_model_beta).add_(
                    param.data, alpha=1 - self.args.avg_model_beta)

    def forward_backward(self, batch, cond):
        self.mp_trainer.zero_grad()
        for i in range(0, batch.shape[0], self.microbatch):
            # Eliminates the microbatch feature
            assert i == 0
            assert self.microbatch == self.batch_size
            micro = batch
            micro_cond = cond
            last_batch = (i + self.microbatch) >= batch.shape[0]
            t, weights = self.schedule_sampler.sample(micro.shape[0], dist_util.dev())

            compute_losses = functools.partial(
                self.diffusion.training_losses,
                self.ddp_model,
                micro,  # [bs, ch, image_size, image_size]
                t,  # [bs](int) sampled timesteps
                model_kwargs=micro_cond,
                dataset=self.data.dataset
            )

            if last_batch or not self.use_ddp:
                losses = compute_losses()
            else:
                with self.ddp_model.no_sync():
                    losses = compute_losses()

            if isinstance(self.schedule_sampler, LossAwareSampler):
                self.schedule_sampler.update_with_local_losses(
                    t, losses["loss"].detach()
                )

            loss = (losses["loss"] * weights).mean()
            log_loss_dict(
                self.diffusion, t, {k: v * weights for k, v in losses.items()}
            )
            self.mp_trainer.backward(loss)

    def _anneal_lr(self):
        if not self.lr_anneal_steps:
            return
        frac_done = (self.total_step()) / self.lr_anneal_steps
        lr = self.lr * (1 - frac_done)
        for param_group in self.opt.param_groups:
            param_group["lr"] = lr

    def log_step(self):
        logger.logkv("step", self.total_step())
        logger.logkv("samples", (self.total_step() + 1) * self.global_batch)


    def ckpt_file_name(self):
        return f"model{(self.total_step()):09d}.pt"

    def generate_during_training(self):
        if not self.args.gen_during_training:
            return
        gen_args = copy.deepcopy(self.args)
        gen_args.model_path = os.path.join(self.save_dir, self.ckpt_file_name())
        gen_args.output_dir = os.path.join(self.save_dir, f'{self.ckpt_file_name()}.samples')
        gen_args.num_samples = self.args.gen_num_samples
        gen_args.num_repetitions = self.args.gen_num_repetitions
        gen_args.guidance_param = self.args.gen_guidance_param
        gen_args.motion_length = None  # length is taken from the dataset
        gen_args.input_text = gen_args.text_prompt = gen_args.action_file = gen_args.action_name = ''
        all_sample_save_path = generate(gen_args, data=self.data.dataset, model=self.model, diffusion=self.diffusion)
        self.ml_platform.report_media(title='Motion', series='Predicted Motion', iteration=self.total_step(),
                                         local_path=all_sample_save_path)

    def save(self):
        def save_checkpoint():
            def del_submodel(state_dict, submodel_name):
                # Do not save unwanted submodel weights
                submodel_weights = [
                    e for e in state_dict.keys() if e.startswith(f'{submodel_name}.')
                ]
                for e in submodel_weights:
                    del state_dict[e]

            if self.use_fp16:
                state_dict = self.model.state_dict()
            else:
                state_dict = self.mp_trainer.master_params_to_state_dict(
                    self.mp_trainer.master_params)
            del_submodel(state_dict, 'clip_model')
            del_submodel(state_dict, 'bert_model')

            if self.args.use_ema:
                # save both the model and the average model
                state_dict_avg = self.model_avg.state_dict()
                del_submodel(state_dict_avg, 'clip_model')
                del_submodel(state_dict_avg, 'bert_model')
                state_dict = {'model': state_dict, 'model_avg': state_dict_avg}

            logger.log(f"saving model...")
            filename = self.ckpt_file_name()
            with bf.BlobFile(bf.join(self.save_dir, filename), "wb") as f:
                torch.save(state_dict, f)

        save_checkpoint()

        with bf.BlobFile(
            bf.join(self.save_dir, f"opt{(self.total_step()):09d}.pt"),
            "wb",
        ) as f:
            opt_state = self.opt.state_dict()
            if self.use_fp16:
                # with fp16 we also save the state dict
                opt_state = {
                    'opt': opt_state,
                    'scaler': self.scaler.state_dict(),
                }

            torch.save(opt_state, f)

    def find_resume_checkpoint(self) -> Optional[str]:
        '''look for all file in save directory in the pattent of model{number}.pt
            and return the one with the highest step number.

        TODO: Implement this function (alredy existing in MDM), so that find model will call it in case a ckpt exist.
        TODO: Change call for find_resume_checkpoint and send save_dir as arg.
        TODO: This means ignoring the flag of resume_checkpoint in case some other ckpts exists in that dir!
        '''

        matches = {file: re.match(r'model(\d+).pt$', file) for file in os.listdir(self.args.save_dir)}
        models = {int(match.group(1)): file for file, match in matches.items() if match}

        return pjoin(self.args.save_dir, models[max(models)]) if models else None


def parse_resume_step_from_filename(filename):
    """
    Parse filenames of the form path/to/modelNNNNNN.pt, where NNNNNN is the
    checkpoint's number of steps.
    """
    split = filename.split("model")
    if len(split) < 2:
        return 0
    split1 = split[-1].split(".")[0]
    try:
        return int(split1)
    except ValueError:
        return 0


def get_blob_logdir():
    # You can change this to be a separate path to save checkpoints to
    # a blobstore or some external drive.
    return logger.get_dir()




def log_loss_dict(diffusion, ts, losses):
    for key, values in losses.items():
        logger.logkv_mean(key, values.mean().item())
        # Log the quantiles (four quartiles, in particular).
        for sub_t, sub_loss in zip(ts.cpu().numpy(), values.detach().cpu().numpy()):
            quartile = int(4 * sub_t / diffusion.num_timesteps)
            logger.logkv_mean(f"{key}_q{quartile}", sub_loss)
