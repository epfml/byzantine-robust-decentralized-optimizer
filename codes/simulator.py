import copy
import logging
import numpy as np
import time
import torch
from typing import Union, Callable, Any

from .worker import Worker, ByzantineWorker
from .server import TorchServer
from .utils import vectorize_model, unstack_vectorized_model


class _SimulatorBase(object):
    """Simulate distributed programs with low memory usage.

    Functionality:
    1. randomness control: numpy, torch, torch-cuda
    2. add workers

    This base class is used by both trainer and evaluator.
    """

    def __init__(self, metrics: dict, use_cuda: bool, debug: bool):
        """
        Args:
            metrics (dict): dict of metric names and their functions
            use_cuda (bool): Use cuda or not
            debug (bool):
        """
        self.metrics = metrics
        self.use_cuda = use_cuda
        self.debug = debug

        self.json_logger = logging.getLogger("stats")
        self.debug_logger = logging.getLogger("debug")


class Evaluator(_SimulatorBase):
    def __init__(
        self,
        model: torch.nn.Module,
        data_loader: torch.utils.data.DataLoader,
        loss_func: torch.nn.modules.loss._Loss,
        device: Union[torch.device, str],
        metrics: dict,
        use_cuda: bool,
        debug: bool,
        meta={"type": "validation"},
    ):
        super().__init__(metrics, use_cuda, debug)
        self.model = model
        self.data_loader = data_loader
        self.loss_func = loss_func
        self.device = device
        self.meta = meta

    def __str__(self):
        return f"Evaluator(type={self.meta['type']})"

    def evaluate(self, epoch):
        self.model.eval()
        r = {
            "_meta": self.meta,
            "E": epoch,
            "Length": 0,
            "Loss": 0,
        }
        for name in self.metrics:
            r[name] = 0

        with torch.no_grad():
            for _, (data, target) in enumerate(self.data_loader):
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                r["Loss"] += self.loss_func(output,
                                            target).item() * len(target)
                r["Length"] += len(target)

                for name, metric in self.metrics.items():
                    r[name] += metric(output, target) * len(target)

        for name in self.metrics:
            r[name] /= r["Length"]
        r["Loss"] /= r["Length"]

        # Output to file
        self.json_logger.info(r)
        self.debug_logger.info(
            f"\n=> {self.meta['type']} Eval Loss={r['Loss']:.4f} "
            + " ".join(name + "=" +
                       "{:>8.4f}".format(r[name]) for name in self.metrics)
            + "\n"
        )


class RandomStatesController(object):
    def __init__(self, use_cuda):
        self.use_cuda = use_cuda
        self.random_states = {}

    def cache(self) -> None:
        if self.use_cuda:
            self.random_states["torch_cuda"] = torch.cuda.get_rng_state()
        self.random_states["torch"] = torch.get_rng_state()
        self.random_states["numpy"] = np.random.get_state()

    def restore(self) -> None:
        if self.use_cuda:
            torch.cuda.set_rng_state(self.random_states["torch_cuda"])
        torch.set_rng_state(self.random_states["torch"])
        np.random.set_state(self.random_states["numpy"])


class ParallelTrainer(_SimulatorBase):
    """Synchronous and parallel training with specified aggregator."""

    def __init__(
        self,
        aggregator: Callable[[list], torch.Tensor],
        pre_batch_hooks: list,
        post_batch_hooks: list,
        max_batches_per_epoch: int,
        log_interval: int,
        metrics: dict,
        use_cuda: bool,
        debug: bool,
    ):
        """
        Args:
            aggregator (callable): A callable which takes a list of tensors and returns
                an aggregated tensor.
            max_batches_per_epoch (int): Set the maximum number of batches in an epoch.
                Usually used for debugging.
            log_interval (int): Control the frequency of logging training batches
            metrics (dict): dict of metric names and their functions
            use_cuda (bool): Use cuda or not
            debug (bool):
        """
        super().__init__(metrics, use_cuda, debug)

        self.aggregator = aggregator
        self.pre_batch_hooks = pre_batch_hooks or []
        self.post_batch_hooks = post_batch_hooks or []
        self.max_batches_per_epoch = max_batches_per_epoch
        self.log_interval = log_interval

        self.server = None
        self.workers = []

        self.random_states_controller = RandomStatesController(
            use_cuda=use_cuda)

    def train(self, epoch):
        self.debug_logger.info(f"Train epoch {epoch}")
        self.pcall(lambda worker: worker.train_epoch_start())

        progress = 0
        for batch_idx in range(self.max_batches_per_epoch):
            try:
                self._run_pre_batch_hooks(epoch, batch_idx)

                # Compute local gradients and log training information.
                batch_info = self.pget(lambda w: w.compute_gradient())
                progress += sum(res["length"] for res in batch_info)
                if batch_idx % self.log_interval == 0:
                    self.log_train(progress, batch_idx, epoch, batch_info)

                aggregated = self.aggregate(epoch, batch_idx)

                self.server.set_gradient(aggregated)
                self.server.apply_gradient()

                self._run_post_batch_hooks(epoch, batch_idx)
            except StopIteration:
                # TODO: ideally all workers should finish their job.
                break

    def aggregate(self, epoch, batch):
        self.pcall(lambda w: w.pre_aggr(epoch, batch))
        gradients = self.pget(lambda w: w.get_gradient())
        self.pcall(lambda w: w.post_aggr(epoch, batch))
        return self.aggregator(gradients)

    # ---------------------------------------------------------------------------- #
    #                                    Utility                                   #
    # ---------------------------------------------------------------------------- #

    def pcall(self, f: Callable[[Worker], None]) -> None:
        """Parallel call."""
        for w in self.workers:
            self.random_states_controller.cache()
            f(w)
            self.random_states_controller.restore()

    def pget(self, f: Callable[[Worker], Any]) -> list:
        """Parallel get."""
        results = []
        for w in self.workers:
            self.random_states_controller.cache()
            results.append(f(w))
            self.random_states_controller.restore()
        return results

    def add_worker(self, worker: Worker):
        self.debug_logger.info(f"=> Add worker {worker}")
        worker.add_metrics(self.metrics)
        self.workers.append(worker)
        assert worker.index == len(self.workers) - 1

    def add_server(self, server: TorchServer):
        self.debug_logger.info(f"=> Add server {server}")
        self.server = server

    def _run_pre_batch_hooks(self, epoch, batch_idx):
        [f(self, epoch, batch_idx) for f in self.pre_batch_hooks]

    def _run_post_batch_hooks(self, epoch, batch_idx):
        [f(self, epoch, batch_idx) for f in self.post_batch_hooks]

    # ---------------------------------------------------------------------------- #
    #                                Log information                               #
    # ---------------------------------------------------------------------------- #

    def __str__(self):
        return (
            "ParallelTrainer("
            f"aggregator={self.aggregator}, "
            f"max_batches_per_epoch={self.max_batches_per_epoch}, "
            f"log_interval={self.log_interval}, "
            f"metrics={list(self.metrics.keys())}"
            f"use_cuda={self.use_cuda}, "
            f"debug={self.debug}, "
            ")"
        )

    def log_train(self, progress, batch_idx, epoch, results):
        length = sum(res["length"] for res in results)

        r = {
            "_meta": {"type": "train"},
            "E": epoch,
            "B": batch_idx,
            "Length": length,
            "Loss": sum(res["loss"] * res["length"] for res in results) / length,
        }

        for metric_name in self.metrics:
            r[metric_name] = (
                sum(res["metrics"][metric_name] * res["length"]
                    for res in results)
                / length
            )

        # Output to console
        total = len(self.workers[0].data_loader.dataset)
        pct = 100 * progress / total
        self.debug_logger.info(
            f"[E{r['E']:2}B{r['B']:<3}| {progress:6}/{total} ({pct:3.0f}%) ] Loss: {r['Loss']:.4f} "
            + " ".join(name + "=" +
                       "{:>8.4f}".format(r[name]) for name in self.metrics)
        )

        # Output to file
        self.json_logger.info(r)


class AverageEvaluator(Evaluator):
    def __init__(
        self,
        models: list,
        data_loader: torch.utils.data.DataLoader,
        loss_func: torch.nn.modules.loss._Loss,
        device: Union[torch.device, str],
        metrics: dict,
        use_cuda: bool,
        debug: bool,
        meta={"type": "Average validation"},
    ):
        self.models = models
        model = copy.deepcopy(models[0])
        super().__init__(
            model, data_loader, loss_func, device, metrics, use_cuda, debug, meta
        )

    def __str__(self):
        return "AverageEvaluator"

    def _prepare_avg_model(self):
        state_dict = self.model.state_dict()
        state_dicts = [m.state_dict() for m in self.models]
        for key in state_dict:
            state_dict[key] = sum(sd[key]
                                  for sd in state_dicts) / len(state_dicts)
        self.model.load_state_dict(state_dict)
        self.model.eval()

    def evaluate(self, epoch):
        self._prepare_avg_model()

        r = {
            "_meta": self.meta,
            "E": epoch,
            "Length": 0,
            "Loss": 0,
        }
        for name in self.metrics:
            r[name] = 0

        with torch.no_grad():
            for _, (data, target) in enumerate(self.data_loader):
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                r["Loss"] += self.loss_func(output,
                                            target).item() * len(target)
                r["Length"] += len(target)

                for name, metric in self.metrics.items():
                    r[name] += metric(output, target) * len(target)

        for name in self.metrics:
            r[name] /= r["Length"]
        r["Loss"] /= r["Length"]

        # Output to file
        self.json_logger.info(r)
        self.debug_logger.info(
            f"\n=> Averaged model ({r['_meta']['type']}) | Eval Loss={r['Loss']:.4f} "
            + " ".join(name + "=" +
                       "{:>8.4f}".format(r[name]) for name in self.metrics)
            + "\n"
        )


class DecentralizedTrainer(_SimulatorBase):
    """Simulate decentralized programs with (low) memory usage.

    The `DecentralizedTrainer` is built on top of DistributedSimulatorBase. The key difference is that the `DecentralizedTrainer` cannot reuse the model.
    """

    def __init__(
        self,
        pre_batch_hooks: list,
        post_batch_hooks: list,
        max_batches_per_epoch: int,
        log_interval: int,
        metrics: dict,
        use_cuda: bool,
        debug: bool,
    ):
        super().__init__(metrics, use_cuda, debug)

        self.graph = None
        self.pre_batch_hooks = pre_batch_hooks or []
        self.post_batch_hooks = post_batch_hooks or []
        self.max_batches_per_epoch = max_batches_per_epoch
        self.log_interval = log_interval
        self.aggregators = []
        self.workers = []

        self.random_states_controller = RandomStatesController(
            use_cuda=use_cuda)
        self._time = 0
        self._counter = 0

    def train(self, epoch):
        self.debug_logger.info(f"Train epoch {epoch}")
        # Prepare model and data iterators
        self.decall(lambda worker: worker.train_epoch_start())

        progress = 0
        for batch_idx in range(self.max_batches_per_epoch):
            try:
                self._run_pre_batch_hooks(epoch, batch_idx)

                t0 = time.time()
                # Compute local gradients and log training information.
                results = self.deget(lambda w: w.compute_gradient())
                progress += sum(res["length"] for res in results)
                if batch_idx % self.log_interval == 0:
                    self.log_train(progress, batch_idx, epoch, results)

                # Update local model, vectorize it, and update it.
                self.decall(self._update_local_model)
                self.aggregate(epoch, batch_idx)
                self.decall(self._load_state_dict)
                self._time += time.time() - t0
                self._counter += 1

                if batch_idx % self.log_interval == 0:
                    print(
                        f"Avg time per batch = {self._time/self._counter:.2f} ({self._counter} batches)"
                    )

                self._run_post_batch_hooks(epoch, batch_idx)
            except StopIteration:
                # TODO: ideally all workers should finish their job.
                break

    def _update_local_model(self, worker: Worker):
        """
        - Good worker and Byz workers (label flipping, bitflipping, etc.)
            * Update local model.
        - Omniscient Byz workers
            * The get_gradient / set_gradient / apply gradient did nothing.
        """
        g = worker.get_gradient()
        worker.set_gradient(g)
        worker.apply_gradient()

        worker.running["flattened_model"] = vectorize_model(worker.model)
        worker.running["aggregated_model"] = None

    def _load_state_dict(self, worker):
        # Save `aggregated_model` to `state_dict`
        state_dict = worker.model.state_dict()
        unstack_vectorized_model(
            model=worker.running["aggregated_model"],
            state_dict=state_dict,
        )
        # Load `state_dict` to model
        worker.model.load_state_dict(state_dict)

    def aggregate(self, epoch, batch):
        """
        Apply gradient locally to local model and then aggregate the models.

        Here is how it works:
        - Each good worker compute gradients on model (x_t)
        - Each worker applies gradients to models and get (x_{t+1/2})
        - Each worker flatten x_{t+1/2} and save to `flattened_model`
            * Each Byzantine worker creates its model based on other good workers.
        - Each worker aggregates `flattened_model` of the neighbors and save to `aggregated_model`
        - Update model from `aggregated_model`
        """
        # Aggregate and save models
        self.decall(lambda w: w.pre_aggr(epoch, batch))

        def _aggregate_models(worker: Worker):
            agg = worker.running["aggregator"]

            inputs = []
            for w in worker.running["neighbor_workers"]:
                inputs.append(w.running["flattened_model"])

            worker.running["aggregated_model"] = agg(
                worker.running["flattened_model"],
                inputs,
            )

        self.decall(_aggregate_models)
        self.decall(lambda w: w.post_aggr(epoch, batch))

    # ---------------------------------------------------------------------------- #
    #                                Construct graph                               #
    # ---------------------------------------------------------------------------- #

    def get_worker(self, node):
        return self.workers[node.index]

    def add_worker(self, worker, aggregator: Callable[[list], torch.Tensor]):
        worker.add_metrics(self.metrics)
        worker.running["aggregator"] = aggregator
        self.workers.append(worker)
        self.aggregators.append(aggregator)
        self.debug_logger.info(f"=> Add worker {worker}")
        assert worker.index == len(self.workers) - 1

    def add_graph(self, graph):
        """Link each node in the graph with a worker.

        Args:
            graph (graph_utils.Graph): A communication constrained graph.
        """
        self.graph = graph
        assert len(self.graph.nodes) == len(self.workers)
        self.json_logger.info(
            {
                "_meta": {"type": "graph"},
                "spectral_gap": graph.spectral_gap,
            }
        )
        self.debug_logger.info(
            "\n=== Start adding graph ===\n" + str(graph) + "\n")

        # Link each worker with the node.
        for n, w in zip(graph.nodes, self.workers):
            # n._get_saved_grad
            w.running["node"] = n
            w.running["neighbor_workers"] = []
            for e in n.edges:
                theothernode = e.theother(n)
                theotherworker = self.workers[theothernode.index]
                w.running["neighbor_workers"].append(theotherworker)

    def get_good_neighbor_workers(self, node):
        good_neighbors = []
        for e in node.edges:
            neighbor_index = e.theother(node).index
            neighbor_worker = self.workers[neighbor_index]
            if not isinstance(neighbor_worker, ByzantineWorker):
                good_neighbors.append(neighbor_worker)
        return good_neighbors

    # ---------------------------------------------------------------------------- #
    #                               Utility functions                              #
    # ---------------------------------------------------------------------------- #

    def decall(self, f: Callable[[Worker], None]) -> None:
        for w in self.workers:
            self.random_states_controller.cache()
            f(w)
            self.random_states_controller.restore()

    def deget(self, f: Callable[[Worker], Any]) -> list:
        # Only used in the hooks for debug information or evaluation, not for training schemes
        results = []
        for w in self.workers:
            self.random_states_controller.cache()
            results.append(f(w))
            self.random_states_controller.restore()
        return results

    def _run_pre_batch_hooks(self, epoch, batch_idx):
        [f(self, epoch, batch_idx) for f in self.pre_batch_hooks]

    def _run_post_batch_hooks(self, epoch, batch_idx):
        [f(self, epoch, batch_idx) for f in self.post_batch_hooks]

    # ---------------------------------------------------------------------------- #
    #                                Log information                               #
    # ---------------------------------------------------------------------------- #
    def log_train(self, progress, batch_idx, epoch, results):
        # Filter the results from Byzantine workers.
        filtered = []
        for res, worker in zip(results, self.workers):
            if not isinstance(worker, ByzantineWorker):
                filtered.append(res)
        self._log_train(progress, batch_idx, epoch, filtered)

    def __str__(self):
        return "DecentralizedSimulator"

    def _log_train(self, progress, batch_idx, epoch, results):
        length = sum(res["length"] for res in results)

        r = {
            "_meta": {"type": "train"},
            "E": epoch,
            "B": batch_idx,
            "Length": length,
            "Loss": sum(res["loss"] * res["length"] for res in results) / length,
        }

        for metric_name in self.metrics:
            r[metric_name] = (
                sum(res["metrics"][metric_name] * res["length"]
                    for res in results)
                / length
            )

        # Output to console
        total = len(self.workers[0].data_loader.dataset)
        pct = 100 * progress / total
        self.debug_logger.info(
            f"[E{r['E']:2}B{r['B']:<3}| {progress:6}/{total} ({pct:3.0f}%) ] Loss: {r['Loss']:.4f} "
            + " ".join(name + "=" +
                       "{:>8.4f}".format(r[name]) for name in self.metrics)
        )

        # Output to file
        self.json_logger.info(r)


class DummyDecentralizedTrainer(DecentralizedTrainer):
    """Worker do not apply gradient any more. Only rely on aggregator for progress.

    Note that compute_gradient as usual so that data loader works as usual.
    """

    def _update_local_model(self, worker: Worker):
        """
        - Good worker and Byz workers (label flipping, bitflipping, etc.)
            * Update local model.
        - Omniscient Byz workers
            * The get_gradient / set_gradient / apply gradient did nothing.
        """
        # g = worker.get_gradient()
        # worker.set_gradient(g)
        # worker.apply_gradient()

        worker.running["flattened_model"] = vectorize_model(worker.model)
        worker.running["aggregated_model"] = None


class MultiRoundsDecentralizedTrainer(DecentralizedTrainer):
    def __init__(self, comm_rounds, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.comm_rounds = comm_rounds

    def aggregate(self, epoch, batch):
        """
        Apply gradient locally to local model and then aggregate the models.

        Here is how it works:
        - Each good worker compute gradients on model (x_t)
        - Each worker applies gradients to models and get (x_{t+1/2})
        - Each worker flatten x_{t+1/2} and save to `flattened_model`
            * Each Byzantine worker creates its model based on other good workers.
        - Each worker aggregates `flattened_model` of the neighbors and save to `aggregated_model`
        - Update model from `aggregated_model`
        """
        # Aggregate and save models
        self.decall(lambda w: w.pre_aggr(epoch, batch))

        def _aggregate_models(worker):
            agg = worker.running["aggregator"]

            inputs = []
            for w in worker.running["neighbor_workers"]:
                inputs.append(w.running["flattened_model"])

            worker.running["aggregated_model"] = agg(
                worker.running["flattened_model"],
                inputs,
            )

        def _update_flattened_model(worker):
            worker.running["flattened_model"] = worker.running[
                "aggregated_model"
            ].clone()

        for _ in range(self.comm_rounds):
            self.decall(_aggregate_models)
            self.decall(_update_flattened_model)

        self.decall(lambda w: w.post_aggr(epoch, batch))
