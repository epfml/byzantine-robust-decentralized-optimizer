import json
import numpy as np
import os
import shutil
import logging
import time
import torch
from contextlib import contextmanager


class BColors(object):
    HEADER = "\033[95m"
    OK_BLUE = "\033[94m"
    OK_CYAN = "\033[96m"
    OK_GREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    END_C = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


def touch(fname: str, times=None, create_dirs: bool = False):
    if create_dirs:
        base_dir = os.path.dirname(fname)
        if not os.path.exists(base_dir):
            os.makedirs(base_dir)
    with open(fname, "a"):
        os.utime(fname, times)


def touch_dir(base_dir: str) -> None:
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def top1_accuracy(output, target):
    return accuracy(output, target, topk=(1,))[0].item()


def log(*args, **kwargs):
    pass


def log_dict(*args, **kwargs):
    pass


def initialize_logger(log_root):
    if not os.path.exists(log_root):
        os.makedirs(log_root)
    else:
        shutil.rmtree(log_root)
        os.makedirs(log_root)

    print(f"Logging files to {log_root}")

    # Only to file; One dict per line; Easy to process
    json_logger = logging.getLogger("stats")
    json_logger.setLevel(logging.INFO)
    fh = logging.FileHandler(os.path.join(log_root, "stats"))
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter("%(message)s"))
    json_logger.addHandler(fh)

    debug_logger = logging.getLogger("debug")
    debug_logger.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("%(message)s"))
    debug_logger.addHandler(ch)
    fh = logging.FileHandler(os.path.join(log_root, "debug"))
    fh.setLevel(logging.INFO)
    debug_logger.addHandler(fh)


def vectorize_model(model):
    state_dict = model.state_dict()
    return torch.cat([state_dict[k].data.view(-1) for k in state_dict])


def unstack_vectorized_model(model, state_dict):
    beg = 0
    for k in state_dict:
        p = state_dict[k]
        end = beg + len(p.data.view(-1))
        state_dict[k] = model[beg:end].reshape_as(p.data)
        beg = end


def filter_entries_from_json(path, kw="validation"):
    """
    Load json file of `stats`.
    """
    print(f"Reading json file {path}")
    validation = []
    with open(path, "r") as f:
        for i, line in enumerate(f):
            line = line.strip().replace("'", '"')
            line = line.replace("nan", '"nan"')
            line = line.replace("inf", '"inf"')
            line = line.replace("'", '"')
            if "mozi" in line:
                # TODO: remove this in the future
                continue
            try:
                data = json.loads(line)
            except:
                print(path)
                print(i)
                print(line)
                raise
            if data["_meta"]["type"] == kw:
                validation.append(data)
    return validation


class Timer(object):
    def __init__(self):
        self._time = 0
        self._counter = 0
        self.t0 = 0

    def __enter__(self):
        self.t0 = time.time()

    def __exit__(self, type, value, traceback):
        self._time += time.time() - self.t0
        self._counter += 1

    @property
    def avg(self):
        return self._time / self._counter

    @property
    def counter(self):
        return self._counter


@contextmanager
def printoptions(*args, **kwargs):
    optional_options = np.get_printoptions()
    np.set_printoptions(*args, **kwargs)
    try:
        yield
    finally:
        np.set_printoptions(**optional_options)
