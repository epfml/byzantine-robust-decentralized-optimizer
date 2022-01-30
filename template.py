import argparse
import copy
import json
import numpy as np
import os
import sys
import torch
import torchvision.datasets as datasets
from torch.nn.modules.loss import CrossEntropyLoss
import torch.nn.functional as F

from codes.worker import SGDMWorker
from codes.aggregator import get_aggregator
from codes.graph_utils import get_graph
from codes.sampler import get_sampler_callback
from codes.simulator import DecentralizedTrainer, AverageEvaluator, Evaluator
from codes.utils import initialize_logger
from codes.tasks.async_loader import AsyncDataLoaderCoordinator
from codes.utils import top1_accuracy

from codes.tasks.cifar10 import cifar10
from codes.tasks.vgg import vgg11
from codes.tasks.mnist import Net, mnist
from codes.tasks.quadratics import LinearModel, get_distributed_quadratics_tasks
from codes.sampler import DistributedSampler


def define_optimizer(parser):
    parser.add_argument("--lr", type=float, default=0.01,
                        help="Initial learning rate.")
    parser.add_argument("--momentum", type=float,
                        default=0.9, help="Momentum.")


def define_dataset(parser):
    parser.add_argument(
        "--noniid",
        type=float,
        default=0,
        help="0 for iid and 1 for noniid",
    )

    parser.add_argument(
        "--longtail",
        type=float,
        default=0,
        help="0 for not-longtail and 1 for longtail",
    )


def define_aggregator(parser):
    parser.add_argument("--agg", type=str, default="avg")


def define_parser():
    parser = argparse.ArgumentParser()

    # Running environment related arguments
    parser.add_argument("--use-cuda", action="store_true", default=False)
    parser.add_argument("--debug", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--identifier", type=str, default="debug", help="")
    parser.add_argument("--analyze", action="store_true",
                        default=False, help="")

    # Common experiment setup
    parser.add_argument("-n", type=int, help="Number of workers")
    parser.add_argument("-f", type=int, help="Number of Byzantine workers.")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Number of workers")
    parser.add_argument("--graph", type=str, default=None)
    parser.add_argument("--attack", type=str, default="NA")

    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Train batch size of 32.",
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=128,
        help="Test batch size of 128.",
    )
    parser.add_argument(
        "--max-batch-size-per-epoch",
        type=int,
        default=999999999,
        help="Early stop of an epoch.",
    )

    define_optimizer(parser)
    define_aggregator(parser)
    define_dataset(parser)

    return parser


# ---------------------------------------------------------------------------- #
#                                     Hooks                                    #
# ---------------------------------------------------------------------------- #
def check_noniid_hooks(trainer, E, B):
    if E == 1 and B == 0:
        lg = trainer.debug_logger
        lg.info(f"\n=== Peeking data label distribution E{E}B{B} ===")
        for w in trainer.workers:
            lg.info(f"Worker {w.index} has targets: {w.running['target'][:5]}")
        lg.info("\n")


# ---------------------------------------------------------------------------- #
#                                 CIFAR10 TASK                                 #
# ---------------------------------------------------------------------------- #
class TaskDef(object):
    @staticmethod
    def model(device):
        raise NotImplementedError

    @staticmethod
    def metrics(device):
        raise NotImplementedError

    @staticmethod
    def loss_func(device):
        raise NotImplementedError

    @staticmethod
    def train_loader(args, data_dir, sampler, loader_kwargs):
        raise NotImplementedError

    @staticmethod
    def test_loader(args, data_dir, loader_kwargs):
        raise NotImplementedError


class CIFAR10Task(TaskDef):
    @staticmethod
    def model(device):
        return vgg11().to(device)

    @staticmethod
    def metrics():
        return {"top1": top1_accuracy}

    @staticmethod
    def loss_func(device):
        return CrossEntropyLoss().to(device)

    @staticmethod
    def train_loader(args, data_dir, sampler, loader_kwargs):
        return cifar10(
            data_dir=data_dir,
            train=True,
            download=True,
            batch_size=args.batch_size,
            sampler_callback=sampler,
            dataset_cls=datasets.CIFAR10,
            drop_last=True,  # Exclude the influence of non-full batch.
            **loader_kwargs,
        )

    @staticmethod
    def test_loader(args, data_dir, loader_kwargs):
        return cifar10(
            data_dir=data_dir,
            train=False,
            download=True,
            batch_size=args.test_batch_size,
            dataset_cls=datasets.CIFAR10,
            shuffle=False,
            **loader_kwargs,
        )
# ---------------------------------------------------------------------------- #
#                                RunnerTemplate                                #
# ---------------------------------------------------------------------------- #


class RunnerTemplate(object):

    DEFAULT_LINE_ARG = """--lr 0.1 --use-cuda --debug -n 4 -f 0 --epochs 150 --momentum 0.9 \
--batch-size 128 --max-batch-size-per-epoch 9999 --graph complete --noniid 0 --agg gossip_avg"""

    ROOT_DIR = os.path.dirname(os.path.abspath(__file__)) + "/"
    DATA_DIR = ROOT_DIR + "datasets/"

    # Pattern of the experiment output director
    EXP_PATTERN = "f{f}m{momentum}n{n}_noniid{noniid}_graph{graph}_agg{agg}"
    LOG_DIR_PATTERN = ROOT_DIR + \
        "outputs/{script}/{exp_id}/" + EXP_PATTERN + "/"

    def __init__(self,
                 parser_func,
                 trainer_fn,
                 sampler_fn,
                 lr_scheduler_fn,
                 task,
                 worker_fn,
                 evaluators_fn,
                 get_graph=get_graph,
                 get_aggregator=get_aggregator):
        parser = parser_func()
        self.args = self.parse_arguments(parser)
        self.check_arguments(self.args)
        self.setup(self.args)

        self.task = task
        self.sampler_fn = sampler_fn
        self.trainer_fn = trainer_fn
        self.lr_scheduler_fn = lr_scheduler_fn
        self.worker_fn = worker_fn
        self.evaluators_fn = evaluators_fn
        self.get_graph = get_graph
        self.get_aggregator = get_aggregator

    def run(self):
        if self.args.analyze:
            self.generate_analysis()
        else:
            self.train()

    def generate_analysis(self):
        raise NotImplementedError

    def train(self):
        args = self.args
        device = torch.device("cuda" if args.use_cuda else "cpu")
        kwargs = {"pin_memory": True}
        graph = self.get_graph(args)
        trainer = self.trainer_fn(args, self.task.metrics())
        model = self.task.model(device)
        loader_coordinator = AsyncDataLoaderCoordinator(device=device)

        trainer.debug_logger.info("\n=== Start adding workers ===")
        lr_schedulers = []
        for rank in range(args.n):
            sampler = self.sampler_fn(args, rank)
            train_loader = self.task.train_loader(
                args, data_dir=self.DATA_DIR, sampler=sampler, loader_kwargs=kwargs
            )
            m = copy.deepcopy(model).to(device)

            # NOTE: for the moment, we fix this to be SGD
            optimizer = torch.optim.SGD(m.parameters(), lr=args.lr)
            lr_scheduler = self.lr_scheduler_fn(optimizer)
            lr_schedulers.append(lr_scheduler)
            train_loader = loader_coordinator.add(train_loader)
            loss_func = self.task.loss_func(device)

            worker = self.worker_fn(args=args, trainer=trainer, rank=rank, model=m,
                                    opt=optimizer, loss_func=loss_func, m=args.momentum,
                                    loader=train_loader, device=device, lr_scheduler=lr_scheduler)

            trainer.add_worker(worker, self.get_aggregator(
                args, graph, rank, worker))

        trainer.add_graph(graph)
        test_loader = self.task.test_loader(args, self.DATA_DIR, kwargs)
        evaluators = self.evaluators_fn(
            args, self.task, trainer, test_loader, device)

        for epoch in range(1, args.epochs + 1):
            trainer.train(epoch)

            # Evaluation
            for evaluator in evaluators:
                evaluator.evaluate(epoch)

            # Update resampler and lr_schedulers
            if hasattr(trainer.workers[0], "sampler") and isinstance(
                trainer.workers[0].sampler, DistributedSampler
            ):
                trainer.decall(
                    lambda w: w.data_loader.sampler.set_epoch(epoch))
            for scheduler in lr_schedulers:
                scheduler.step()

    # ---------------------------------------------------------------------------- #
    #                                Parse arguments                               #
    # ---------------------------------------------------------------------------- #

    def parse_arguments(self, parser):
        if len(sys.argv) > 1:
            return parser.parse_args()
        return parser.parse_args(self.DEFAULT_LINE_ARG.split())

    def check_arguments(self, args):
        assert args.n > 0
        assert args.epochs >= 1

    # ---------------------------------------------------------------------------- #
    #                               Setup experiments                              #
    # ---------------------------------------------------------------------------- #
    def setup(self, args):
        self._setup_logs(args)

        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

    def _setup_logs(self, args):
        assert "script" not in args.__dict__
        assert "exp_id" not in args.__dict__
        log_dir = self.LOG_DIR_PATTERN.format(
            script=sys.argv[0][:-3],
            exp_id=args.identifier,
            # NOTE: Customize the hp
            **args.__dict__
        )

        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        self.log_dir = log_dir

        if not args.analyze:
            initialize_logger(log_dir)
            with open(os.path.join(log_dir, "args.json"), "w") as f:
                json.dump(args.__dict__, f)


# ---------------------------------------------------------------------------- #
#                            CIFAR10 Runner Example                            #
# ---------------------------------------------------------------------------- #

class CIFAR10_Template(RunnerTemplate):
    """
    The default setup with VGG 11 yields (88.38)% accuracy after 150 epochs.

    Setups:
    - n=4 + fully connected + iid + gossip avg
    - momentum=0.9 batchsize=128
    - epochs=150
    - no weight decay (maybe add one)
    """

    DEFAULT_LINE_ARG = """--lr 0.05 --use-cuda --debug -n 4 -f 0 --epochs 150 --momentum 0.9 \
--batch-size 128 --max-batch-size-per-epoch 9999 --graph complete --noniid 0 --agg gossip_avg \
--identifier vgg11
"""

    def __init__(self,
                 parser_func=define_parser,
                 trainer_fn=lambda args, metrics: DecentralizedTrainer(
                     pre_batch_hooks=[],
                     post_batch_hooks=[check_noniid_hooks],
                     max_batches_per_epoch=args.max_batch_size_per_epoch,
                     log_interval=args.log_interval,
                     metrics=metrics,
                     use_cuda=args.use_cuda,
                     debug=args.debug,
                 ),
                 sampler_fn=lambda args, rank: get_sampler_callback(
                     rank, args.n, noniid=args.noniid, longtail=args.longtail),
                 lr_scheduler_fn=lambda opt: torch.optim.lr_scheduler.MultiStepLR(
                     opt, milestones=list(range(30, 300, 30)), gamma=0.5),
                 task=CIFAR10Task,
                 worker_fn=lambda args, trainer, rank, model, opt, loss_func, m, loader, device, lr_scheduler: SGDMWorker(
                     momentum=m,
                     index=rank,
                     data_loader=loader,
                     model=model,
                     optimizer=opt,
                     loss_func=loss_func,
                     device=device,
                     lr_scheduler=lr_scheduler),
                 evaluators_fn=lambda args, task, trainer, test_loader, device: [
                     AverageEvaluator(
                         models=[w.model for w in trainer.workers],
                         data_loader=test_loader,
                         loss_func=task.loss_func(device),
                         device=device,
                         metrics=task.metrics(),
                         use_cuda=args.use_cuda,
                         debug=args.debug,
                     )
                 ],
                 get_graph=get_graph,
                 get_aggregator=get_aggregator):
        super().__init__(
            parser_func=parser_func,
            trainer_fn=trainer_fn,
            sampler_fn=sampler_fn,
            lr_scheduler_fn=lr_scheduler_fn,
            task=task,
            worker_fn=worker_fn,
            evaluators_fn=evaluators_fn,
            get_graph=get_graph,
            get_aggregator=get_aggregator
        )

# ---------------------------------------------------------------------------- #
#                             MNIST Runner example                             #
# ---------------------------------------------------------------------------- #


class MNISTTask(TaskDef):
    @staticmethod
    def model(device):
        return Net().to(device)

    @staticmethod
    def metrics():
        return {"top1": top1_accuracy}

    @staticmethod
    def loss_func(device):
        return F.nll_loss

    @staticmethod
    def train_loader(args, data_dir, sampler, loader_kwargs):
        return mnist(
            data_dir=data_dir,
            train=True,
            download=True,
            batch_size=args.batch_size,
            sampler_callback=sampler,
            dataset_cls=datasets.MNIST,
            drop_last=True,  # Exclude the influence of non-full batch.
            **loader_kwargs,
        )

    @staticmethod
    def test_loader(args, data_dir, loader_kwargs):
        return mnist(
            data_dir=data_dir,
            train=False,
            download=True,
            batch_size=args.test_batch_size,
            dataset_cls=datasets.MNIST,
            shuffle=False,
            **loader_kwargs,
        )


class MNISTTemplate(RunnerTemplate):
    """
    Accuracy 98.48%
    """

    DEFAULT_LINE_ARG = """--lr 0.01 --use-cuda --debug -n 8 -f 0 --epochs 30 --momentum 0.0 \
--batch-size 32 --max-batch-size-per-epoch 9999 --graph complete --noniid 0 --agg gossip_avg \
--identifier mnist"""

    def __init__(self,
                 parser_func=define_parser,
                 trainer_fn=lambda args, metrics: DecentralizedTrainer(
                     pre_batch_hooks=[],
                     post_batch_hooks=[check_noniid_hooks],
                     max_batches_per_epoch=args.max_batch_size_per_epoch,
                     log_interval=args.log_interval,
                     metrics=metrics,
                     use_cuda=args.use_cuda,
                     debug=args.debug,
                 ),
                 sampler_fn=lambda args, rank: get_sampler_callback(
                     rank, args.n, noniid=args.noniid, longtail=args.longtail),
                 lr_scheduler_fn=lambda opt: torch.optim.lr_scheduler.MultiStepLR(
                     opt, milestones=[], gamma=1.0),
                 task=MNISTTask,
                 worker_fn=lambda args, trainer, rank, model, opt, loss_func, m, loader, device, lr_scheduler: SGDMWorker(
                     momentum=m,
                     index=rank,
                     data_loader=loader,
                     model=model,
                     optimizer=opt,
                     loss_func=loss_func,
                     device=device,
                     lr_scheduler=lr_scheduler),
                 evaluators_fn=lambda args, task, trainer, test_loader, device: [
                     AverageEvaluator(
                         models=[w.model for w in trainer.workers],
                         data_loader=test_loader,
                         loss_func=task.loss_func(device),
                         device=device,
                         metrics=task.metrics(),
                         use_cuda=args.use_cuda,
                         debug=args.debug,
                     )
                 ],
                 get_graph=get_graph,
                 get_aggregator=get_aggregator):
        super().__init__(
            parser_func=parser_func,
            trainer_fn=trainer_fn,
            sampler_fn=sampler_fn,
            lr_scheduler_fn=lr_scheduler_fn,
            task=task,
            worker_fn=worker_fn,
            evaluators_fn=evaluators_fn,
            get_graph=get_graph,
            get_aggregator=get_aggregator
        )

# ---------------------------------------------------------------------------- #
#                              Quadratics Problem                              #
# ---------------------------------------------------------------------------- #


def define_parser_quadratics():
    parser = define_parser()

    # NOTE: customize per script
    parser.add_argument("--n-samples-per_worker", type=int, default=100)
    parser.add_argument("-d", type=int, default=10)
    parser.add_argument("-L", type=float, default=30.0)
    parser.add_argument("--mu", type=float, default=-1.0)
    parser.add_argument("--r0", type=float, default=10.0)
    parser.add_argument("--sigma", type=float, default=0.0)
    parser.add_argument("--zeta", type=float, default=1.0)

    parser.add_argument("--comm-rounds", type=int, default=1)
    return parser


class QuadraticsTask(TaskDef):
    def __init__(self, args, tasks, main_task):
        self.args = args
        self.tasks = tasks
        self.main_task = main_task

    def model(self, device):
        model = LinearModel(self.args.d)
        model.layer.weight.data /= model.layer.weight.data.norm() / self.args.r0
        return model

    @staticmethod
    def metrics():
        return {}

    @staticmethod
    def loss_func(device):
        return torch.nn.MSELoss(reduction="mean")

    def train_loader(self, args, data_dir, sampler, loader_kwargs):
        rank = sampler
        return self.tasks[rank].train_loader()

    def test_loader(self, args, data_dir, loader_kwargs):
        return self.main_task.test_loader()


class QuadraticsTemplate(RunnerTemplate):
    """
    Setups:
    """

    DEFAULT_LINE_ARG = """--debug -n 16 -f 0 --epochs 100 --momentum 0 --batch-size 100 -d 10 --n-samples-per_worker 200 \
-L 30.0 --mu -1.0 --r0 10.0 --sigma 0.0 --zeta 0.0 \
--graph torusC4C4 --agg gossip_avg --identifier quadratics"""

    EXP_PATTERN = "f{f}m{momentum}n{n}graph{graph}_agg{agg}_d{d}_L{L}_mu{mu}_r0{r0}_sigma{sigma}_zeta{zeta}"
    LOG_DIR_PATTERN = RunnerTemplate.ROOT_DIR + \
        "outputs/{script}/{exp_id}/" + EXP_PATTERN + "/"

    def __init__(self):
        def trainer_fn(args, metrics): return DecentralizedTrainer(
            pre_batch_hooks=[],
            post_batch_hooks=[check_noniid_hooks],
            max_batches_per_epoch=args.max_batch_size_per_epoch,
            log_interval=args.log_interval,
            metrics=metrics,
            use_cuda=args.use_cuda,
            debug=args.debug,
        )

        def sampler_fn(args, rank):
            return rank

        def lr_scheduler_fn(opt): return torch.optim.lr_scheduler.MultiStepLR(
            opt, milestones=[], gamma=1.0)

        def worker_fn(args, trainer, rank, model, opt, loss_func, m, loader, device): return SGDMWorker(
            momentum=m,
            index=rank,
            data_loader=loader,
            model=model,
            optimizer=opt,
            loss_func=loss_func,
            device=device)

        def evaluators_fn(args, task, trainer, test_loader, device): return [
            AverageEvaluator(
                models=[w.model for w in trainer.workers],
                data_loader=test_loader,
                loss_func=task.loss_func(device),
                device=device,
                metrics=task.metrics(),
                use_cuda=args.use_cuda,
                debug=args.debug,
            )
        ]

        #####################################################
        parser = define_parser_quadratics()
        args = self.parse_arguments(parser)
        self.args = args
        self.check_arguments(self.args)
        self.setup(self.args)
        args.lr = 1 / args.L * 0.5

        tasks, main_task = get_distributed_quadratics_tasks(
            m=args.n,
            n=args.n_samples_per_worker,
            d=args.d,
            b=args.batch_size,
            L=args.L,
            mu=args.mu,
            r0=args.r0,
            sigma=args.sigma,
            zeta=args.zeta,
            seed=args.seed,
        )

        self.task = QuadraticsTask(args, tasks, main_task)

        self.sampler_fn = sampler_fn
        self.trainer_fn = trainer_fn
        self.lr_scheduler_fn = lr_scheduler_fn
        self.worker_fn = worker_fn
        self.evaluators_fn = evaluators_fn


if __name__ == "__main__":
    runner = QuadraticsTemplate()
    runner.run()
