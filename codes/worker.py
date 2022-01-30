import logging
import torch
from collections import defaultdict
from typing import Union, Callable, Tuple
from .utils import Timer

debug_logger = logging.getLogger("debug")


class Worker(object):
    def __init__(
        self,
        index: int,
        data_loader: torch.utils.data.DataLoader,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        loss_func: torch.nn.modules.loss._Loss,
        device: Union[torch.device, str],
        lr_scheduler: None,
    ):
        self.index = index
        self.data_loader = data_loader
        self.model = model
        self.optimizer = optimizer
        self.loss_func = loss_func
        self.device = device
        self.lr_scheduler=lr_scheduler

        # self.running has attribute:
        #   - `train_loader_iterator`: data iterator
        #   - `data`: last data
        #   - `target`: last target
        self.running = {}
        self.metrics = {}
        self.state = defaultdict(dict)
        self.timer = Timer()

    def add_metric(
        self,
        name: str,
        callback: Callable[[torch.Tensor, torch.Tensor], float],
    ):
        """
        metric = callback(predicted, groundtruth)
        """
        if name in self.metrics or name in ["loss", "length"]:
            raise KeyError(f"Metrics ({name}) already added.")

        self.metrics[name] = callback

    def add_metrics(self, metrics: dict):
        for name in metrics:
            self.add_metric(name, metrics[name])

    def __str__(self) -> str:
        return f"Worker(index={self.index})"

    def train_epoch_start(self) -> None:
        self.running["train_loader_iterator"] = iter(self.data_loader)
        self.model.train()

    def compute_gradient(self) -> Tuple[float, int]:
        results = {}

        with self.timer:
            data, target = self.running["train_loader_iterator"].__next__()
            data, target = data.to(self.device), target.to(self.device)

        # if self.index == 0:
        #     print(
        #         f"DataLoading Avg Time = {self.timer.avg:.6f}s ({self.timer.counter} batches)"
        #     )

        self.optimizer.zero_grad()
        output = self.model(data)
        loss = self.loss_func(output, target)
        loss.backward()
        self._save_updates()

        self.running["data"] = data
        self.running["target"] = target

        results["loss"] = loss.item()
        results["length"] = len(target)
        results["metrics"] = {}
        for name, metric in self.metrics.items():
            results["metrics"][name] = metric(output, target)
        return results

    def get_gradient(self) -> torch.Tensor:
        return self._get_updates()

    def apply_gradient(self) -> None:
        self.optimizer.step()

    def set_gradient(self, gradient: torch.Tensor) -> None:
        beg = 0
        for p in self.model.parameters():
            end = beg + len(p.grad.view(-1))
            x = gradient[beg:end].reshape_as(p.grad.data)
            p.grad.data = x.clone().detach()
            beg = end

    def _save_updates(self) -> None:
        raise NotImplementedError

    def _get_updates(self) -> torch.Tensor:
        raise NotImplementedError

    def pre_aggr(self, epoch, batch):
        pass

    def post_aggr(self, epoch, batch):
        pass


class SGDWorker(Worker):
    def _save_updates(self) -> None:
        for group in self.optimizer.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                param_state = self.state[p]
                param_state["gradient_buffer"] = torch.clone(p.grad).detach()

    def _get_updates(self) -> torch.Tensor:
        layer_gradients = []
        for group in self.optimizer.param_groups:
            for p in group["params"]:
                param_state = self.state[p]
                layer_gradients.append(
                    param_state["gradient_buffer"].data.view(-1))
        return torch.cat(layer_gradients)


class SGDMWorker(Worker):
    def __init__(self, momentum, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.momentum = momentum

    def _save_updates(self) -> None:
        for group in self.optimizer.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                param_state = self.state[p]
                if "momentum_buffer" not in param_state:
                    param_state["momentum_buffer"] = torch.clone(
                        p.grad).detach()
                else:
                    param_state["momentum_buffer"].mul_(
                        self.momentum).add_(p.grad)

    def _get_updates(self) -> torch.Tensor:
        layer_gradients = []
        for group in self.optimizer.param_groups:
            for p in group["params"]:
                param_state = self.state[p]
                layer_gradients.append(
                    param_state["momentum_buffer"].data.view(-1))
        return torch.cat(layer_gradients)

    def __str__(self) -> str:
        return f"SGDMWorker(index={self.index}, momentum={self.momentum})"


class ByzantineWorker(SGDWorker):
    def __init__(self, simulator, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.simulator = simulator

    def compute_gradient(self) -> Tuple[float, int]:
        # Use self.simulator to get all other workers
        # Note that the byzantine worker does not modify the states directly.
        return super().compute_gradient()

    def get_gradient(self) -> torch.Tensor:
        # Use self.simulator to get all other workers
        return super().get_gradient()

    def __str__(self) -> str:
        return f"ByzantineWorker(index={self.index})"

    def pre_aggr(self, epoch, batch):
        pass

    def post_aggr(self, epoch, batch):
        pass
