import os
import glob


class MLPlatform:
    def __init__(self, *args, **kwargs):
        save_dir = kwargs.get('save_dir', 'unnamed_path/unnamed_experiment')
        self.path, file = os.path.split(save_dir)
        self.name = kwargs.get('name', file)
        pass

    def report_scalar(self, name, value, iteration, group_name=None):
        pass

    def report_media(self, title, series, iteration, local_path):
        pass

    def report_args(self, args, name):
        pass

    def close(self):
        pass


class ClearmlPlatform(MLPlatform):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        from clearml import Task
        path, name = os.path.split(save_dir)
        self.task = Task.init(project_name='some_project',
                              task_name=name)
        self.logger = self.task.get_logger()

    def report_scalar(self, name, value, iteration, group_name):
        self.logger.report_scalar(title=group_name, series=name, iteration=iteration, value=value)

    def report_media(self, title, series, iteration, local_path):
        self.logger.report_media(title=title, series=series, iteration=iteration, local_path=local_path)

    def report_args(self, args, name):
        self.task.connect(args, name=name)

    def close(self):
        self.task.close()


class TensorboardPlatform(MLPlatform):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        from torch.utils.tensorboard import SummaryWriter
        self.writer = SummaryWriter(log_dir=save_dir)

    def report_scalar(self, name, value, iteration, group_name=None):
        self.writer.add_scalar(f'{group_name}/{name}', value, iteration)

    def close(self):
        self.writer.close()


class NoPlatform(MLPlatform):
    def __init__(self, *args, **kwargs):
        pass

class WandBPlatform(MLPlatform):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        import wandb
        self.wandb = wandb
        wandb.login(host=os.getenv("WANDB_BASE_URL"), key=os.getenv("WANDB_API_KEY"))
 
        # check if an experiment with the same id is already running
        api = wandb.Api()
        project = kwargs.get('project', 'unnamed_project')
        entity = kwargs.get('entity', 'unnamed_entity')
        config = kwargs.get('config', None)
        runs = api.runs(path=f'{entity}/{project}')
        for run in runs:
            # print(run.name, run.state)
            if run.name == self.name and run.state == 'running':
                raise Exception(f'Experiment with name {self.name} is already running')
        wandb.init(
            project=project,
            name=self.name,
            id=self.name,  # in order to send continued runs to the same record
            resume='allow',  # in order to send continued runs to the same record
            entity=entity,
            save_code=True,
            config=config)  # config can also be sent via report_args()

    def report_scalar(self, name, value, iteration, group_name=None):
        self.wandb.log({name: value}, step=iteration)

    def report_media(self, title, series, iteration, local_path):
        files = glob.glob(f'{local_path}/*.mp4')
        self.wandb.log({series: [self.wandb.Video(file, format='mp4', fps=20) for file in files]}, step=iteration)

    def report_args(self, args, name):
        self.wandb.config.update(args)  # , allow_val_change=True) # use allow_val_change ONLY if you want to change existing args (e.g., overwrite)

    def watch_model(self, *args, **kwargs):
        self.wandb.watch(args, kwargs)

    def close(self):
       self.wandb.finish()
