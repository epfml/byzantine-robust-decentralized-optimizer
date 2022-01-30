import numpy as np
import torch
from scipy.stats import norm

from codes.worker import ByzantineWorker
from codes.aggregator import DecentralizedAggregator


class DecentralizedByzantineWorker(ByzantineWorker):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # The target of attack
        self.target = None
        self.tagg = None
        self.target_good_neighbors = None

    def _initialize_target(self):
        if self.target is None:
            assert len(self.running["neighbor_workers"]) == 1
            self.target = self.running["neighbor_workers"][0]
            self.tagg = self.target.running["aggregator"]
            self.target_good_neighbors = self.simulator.get_good_neighbor_workers(
                self.target.running["node"]
            )


class DissensusWorker(DecentralizedByzantineWorker):
    def __init__(self, epsilon, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.epsilon = epsilon

    def _attack_decentralized_aggregator(self, mixing=None):
        tm = self.target.running["flattened_model"]

        # Compute Byzantine weights
        partial_sum = []
        partial_byz_weights = []
        for neighbor in self.target.running["neighbor_workers"]:
            nm = neighbor.running["flattened_model"]
            nn = neighbor.running["node"]
            nw = mixing or self.tagg.weights[nn.index]
            if isinstance(neighbor, ByzantineWorker):
                partial_byz_weights.append(nw)
            else:
                partial_sum.append(nw * (nm - tm))

        partial_sum = sum(partial_sum)
        partial_byz_weights = sum(partial_byz_weights)

        return tm, partial_sum / partial_byz_weights

    def pre_aggr(self, epoch, batch):
        self._initialize_target()

        if isinstance(self.tagg, DecentralizedAggregator):
            # Dissensus using the gossip weight
            tm, v = self._attack_decentralized_aggregator()
            self.running["flattened_model"] = tm - self.epsilon * v
        else:
            # TODO: check
            # Dissensus using the gossip weight
            mixing = 1 / (len(self.target.running["neighbor_workers"]) + 1)
            tm, v = self._attack_decentralized_aggregator(mixing)
            self.running["flattened_model"] = tm - self.epsilon * v


class BitFlippingWorker(ByzantineWorker):
    def __str__(self) -> str:
        return "BitFlippingWorker"

    def pre_aggr(self, epoch, batch):
        self.running["flattened_model"] = -self.running["flattened_model"]


class LabelFlippingWorker(ByzantineWorker):
    def __init__(self, revertible_label_transformer, *args, **kwargs):
        """
        Args:
            revertible_label_transformer (callable):
                E.g. lambda label: 9 - label
        """
        super().__init__(*args, **kwargs)
        self.revertible_label_transformer = revertible_label_transformer

    def train_epoch_start(self) -> None:
        super().train_epoch_start()
        self.running["train_loader_iterator"].__next__ = self._wrap_iterator(
            self.running["train_loader_iterator"].__next__
        )

    def _wrap_iterator(self, func):
        def wrapper():
            data, target = func()
            return data, self.revertible_label_transformer(target)

        return wrapper

    def _wrap_metric(self, func):
        def wrapper(output, target):
            return func(output, self.revertible_label_transformer(target))

        return wrapper

    def add_metric(self, name, callback):
        if name in self.metrics or name in ["loss", "length"]:
            raise KeyError(f"Metrics ({name}) already added.")

        self.metrics[name] = self._wrap_metric(callback)

    def __str__(self) -> str:
        return "LabelFlippingWorker"


class ALittleIsEnoughAttack(DecentralizedByzantineWorker):
    """
    Adapted for the decentralized environment.

    Args:
        n (int): Total number of workers
        m (int): Number of Byzantine workers
    """

    def __init__(self, n, m, z=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Number of supporters
        if z is not None:
            self.z_max = z
        else:
            s = np.floor(n / 2 + 1) - m
            cdf_value = (n - m - s) / (n - m)
            self.z_max = norm.ppf(cdf_value)
        self.n_good = n - m

    def get_gradient(self):
        return 0

    def set_gradient(self, gradient):
        pass

    def apply_gradient(self):
        pass

    def pre_aggr(self, epoch, batch):
        self._initialize_target()

        tm = self.target.running["flattened_model"]
        models = [tm]
        for neighbor in self.target_good_neighbors:
            models.append(neighbor.running["flattened_model"])

        stacked_models = torch.stack(models, 1)
        mu = torch.mean(stacked_models, 1)
        std = torch.std(stacked_models, 1)

        self.running["flattened_model"] = mu - std * self.z_max


class IPMAttack(DecentralizedByzantineWorker):
    def __init__(self, epsilon, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.epsilon = epsilon

    def get_gradient(self):
        return 0

    def set_gradient(self, gradient):
        pass

    def apply_gradient(self):
        pass

    def pre_aggr(self, epoch, batch):
        self._initialize_target()

        tm = self.target.running["flattened_model"]
        models = [tm]
        for neighbor in self.target_good_neighbors:
            models.append(neighbor.running["flattened_model"])

        self.running["flattened_model"] = -self.epsilon * sum(models) / len(models)


def get_attackers(
    args, rank, trainer, model, opt, loss_func, loader, device, lr_scheduler
):
    if args.attack == "BF":
        return BitFlippingWorker(
            simulator=trainer,
            index=rank,
            data_loader=loader,
            model=model,
            loss_func=loss_func,
            device=device,
            optimizer=opt,
            lr_scheduler=lr_scheduler,
        )

    if args.attack == "LF":
        return LabelFlippingWorker(
            revertible_label_transformer=lambda label: 9 - label,
            simulator=trainer,
            index=rank,
            data_loader=loader,
            model=model,
            loss_func=loss_func,
            device=device,
            optimizer=opt,
            lr_scheduler=lr_scheduler,
        )

    if args.attack.startswith("ALIE"):
        if args.attack == "ALIE":
            z = None
        else:
            z = float(args.attack[4:])
        attacker = ALittleIsEnoughAttack(
            n=args.n,
            m=args.f,
            z=z,
            simulator=trainer,
            index=rank,
            data_loader=loader,
            model=model,
            loss_func=loss_func,
            device=device,
            optimizer=opt,
            lr_scheduler=lr_scheduler,
        )
        return attacker

    if args.attack == "IPM":
        attacker = IPMAttack(
            epsilon=0.1,
            simulator=trainer,
            index=rank,
            data_loader=loader,
            model=model,
            loss_func=loss_func,
            device=device,
            optimizer=opt,
            lr_scheduler=lr_scheduler,
        )
        return attacker

    if args.attack.startswith("dissensus"):
        epsilon = float(args.attack[len("dissensus") :])
        attacker = DissensusWorker(
            epsilon=epsilon,
            simulator=trainer,
            index=rank,
            data_loader=loader,
            model=model,
            loss_func=loss_func,
            device=device,
            optimizer=opt,
            lr_scheduler=lr_scheduler,
        )
        return attacker
    raise NotImplementedError(f"No such attack {args.attack}")
