import logging
import os
import sys
import atexit

_PROGRESS_TTY = None


def _get_progress_file():
    global _PROGRESS_TTY
    if sys.stderr.isatty():
        return sys.stderr
    if _PROGRESS_TTY is not None and not _PROGRESS_TTY.closed:
        return _PROGRESS_TTY
    try:
        _PROGRESS_TTY = open("/dev/tty", "w")
        atexit.register(_PROGRESS_TTY.close)
        return _PROGRESS_TTY
    except OSError:
        return sys.stderr


def create_progress_bar(*args, **kwargs):
    from progress.bar import Bar
    kwargs.setdefault("file", _get_progress_file())
    if "width" not in kwargs:
        import shutil
        term_width = shutil.get_terminal_size((80, 20)).columns
        kwargs["width"] = 6 if term_width < 80 else 12
    return Bar(*args, **kwargs)


def format_duration_compact(seconds):
    if seconds is None:
        return "?"
    try:
        seconds = float(seconds)
    except (TypeError, ValueError):
        return "?"
    if seconds < 0:
        seconds = 0.0
    if seconds < 60:
        return f"{int(round(seconds))}s"
    if seconds < 3600:
        return f"{seconds / 60:.1f}m"
    if seconds < 86400:
        return f"{seconds / 3600:.1f}h"
    return f"{seconds / 86400:.1f}d"


def estimate_remaining_time(start_time, completed_units, total_units, now=None):
    if completed_units <= 0 or total_units <= completed_units:
        return 0.0
    import time
    if now is None:
        now = time.time()
    elapsed = now - start_time
    seconds_per_unit = elapsed / completed_units
    return seconds_per_unit * (total_units - completed_units)


def compact_progress_suffix(loss_value, epoch_eta_seconds, total_eta_seconds):
    return (
        f"loss {loss_value:.4f} | "
        f"eta {format_duration_compact(epoch_eta_seconds)} | "
        f"total {format_duration_compact(total_eta_seconds)}"
    )

def path_to_repo(*args):
    repo_root = os.path.dirname(os.path.dirname(__file__))
    return os.path.join(repo_root, *args)

def path_to_data(*args):
    return path_to_repo("data", *args)

def path_to_experiment(*args, base_dir=None):
    if base_dir is None:
        return path_to_repo("experiments", *args)
    if os.path.isabs(base_dir):
        return os.path.join(base_dir, *args)
    return path_to_repo(base_dir, *args)

def create_logger(logdir):
    head = '%(asctime)-15s %(message)s'
    if logdir != '':
        log_file = os.path.join(logdir, 'log.txt')
        logging.basicConfig(filename=log_file, format=head)
        logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    else:
        logging.basicConfig(format=head)

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    return logger


def init_output_dirs(exp_name="default", base_dir=None):
    log_dir = path_to_experiment(exp_name, base_dir=base_dir)
    ckpt_dir = os.path.join(log_dir, "checkpoints")
    runs_dir = os.path.join(log_dir, "tensorboard")

    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(runs_dir, exist_ok=True)
     
    return log_dir, ckpt_dir, runs_dir


def load_config(path, exp_name="default"):
    import yaml

    with open(path, "rt") as reader:
        config = yaml.safe_load(reader)

    if "OUTPUT" not in config:
        config["OUTPUT"] = {}
    base_dir = config["OUTPUT"].get("base_dir")
    config["OUTPUT"]["log_dir"], config["OUTPUT"]["ckpt_dir"], config["OUTPUT"]["runs_dir"] = init_output_dirs(exp_name=exp_name, base_dir=base_dir)

    with open(os.path.join(config["OUTPUT"]["ckpt_dir"], "config.yaml"), 'w') as f:
        yaml.dump(config, f)

    return config

class RunningAverage:
    """Accumulate a sample-weighted arithmetic mean."""

    def __init__(self):
        self.weighted_total = 0.0
        self.samples = 0

    @property
    def mean(self):
        if self.samples == 0:
            return 0.0
        return self.weighted_total / self.samples

    def add(self, value, samples=1):
        if samples < 0:
            raise ValueError("samples must be non-negative")
        self.weighted_total += value * samples
        self.samples += samples


def save_checkpoint(
    model,
    optimizer,
    planner_optimizer,
    forecaster_optimizer,
    epoch,
    config,
    filename,
    logger,
):
    import torch

    logger.info(f"Saving checkpoint to {filename}.")
    checkpoint = {
        "model": model.state_dict(),
        "epoch": epoch,
        "config": config,
    }
    if optimizer is not None:
        checkpoint["optimizer"] = optimizer.state_dict()
    if planner_optimizer is not None:
        checkpoint["opt_planner"] = planner_optimizer.state_dict()
    if forecaster_optimizer is not None:
        checkpoint["opt_forecaster"] = forecaster_optimizer.state_dict()
    torch.save(
        checkpoint, os.path.join(config["OUTPUT"]["ckpt_dir"], filename)
    )
