import copy
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import numpy as np

import codes.graph_utils as gu
from codes.simulator import ByzantineWorker, DummyDecentralizedTrainer
from codes.utils import filter_entries_from_json
from codes.attacks import get_attackers
from codes.sampler import DumbbellSampler, DecentralizedMixedSampler

from template import MNISTTemplate, MNISTTask
from template import (
    define_parser,
    DecentralizedTrainer,
    check_noniid_hooks,
    get_aggregator,
    get_sampler_callback,
    SGDMWorker,
    AverageEvaluator,
)

LOG_CONSENSUS_DISTANCE_INTERVAL = 10


# ---------------------------------------------------------------------------- #
#                                     Hooks                                    #
# ---------------------------------------------------------------------------- #


def log_global_consensus_distance(trainer, E, B):
    """Log the consensus distance among all good workers."""
    # TODO: Check
    if B % LOG_CONSENSUS_DISTANCE_INTERVAL == 0:
        lg = trainer.debug_logger
        jlg = trainer.json_logger
        lg.info(f"\n=== Log global consensus distance @ E{E}B{B} ===")

        mean, counter = 0, 0
        for w in trainer.workers:
            if not isinstance(w, ByzantineWorker):
                mean += w.running["aggregated_model"]
                counter += 1
        mean /= counter

        consensus_distance = 0
        for w in trainer.workers:
            if not isinstance(w, ByzantineWorker):
                consensus_distance += (
                    w.running["aggregated_model"] - mean).norm() ** 2
        consensus_distance /= counter
        lg.info(f"consensus_distance={consensus_distance:.3f}")

        jlg.info(
            {
                "_meta": {"type": "global_consensus_distance"},
                "E": E,
                "B": B,
                "gcd": consensus_distance.item(),
            }
        )

        lg.info("\n")


def log_clique_consensus_distance(trainer, E, B):
    """Log the consensus distance among all good workers."""
    # TODO: Check
    if B % LOG_CONSENSUS_DISTANCE_INTERVAL == 0:
        lg = trainer.debug_logger
        jlg = trainer.json_logger
        lg.info(f"\n=== Log clique consensus distance @ E{E}B{B} ===")

        counter = 0
        for w in trainer.workers:
            if not isinstance(w, ByzantineWorker):
                counter += 1
        clique_size = (counter - 1) // 2
        assert counter == clique_size * 2 + 1, (clique_size, counter)

        mean1, mean2, c = 0, 0, 0
        for w in trainer.workers:
            if not isinstance(w, ByzantineWorker):
                if c < clique_size:
                    mean1 += w.running["aggregated_model"]
                elif c < 2 * clique_size:
                    mean2 += w.running["aggregated_model"]
                c += 1
        mean1 /= clique_size
        mean2 /= clique_size

        cd1, cd2, c = 0, 0, 0
        for w in trainer.workers:
            if not isinstance(w, ByzantineWorker):
                if c < clique_size:
                    cd1 += (w.running["aggregated_model"] - mean1).norm() ** 2
                elif c < 2 * clique_size:
                    cd2 += (w.running["aggregated_model"] - mean1).norm() ** 2
                c += 1
        cd1 /= clique_size
        cd2 /= clique_size

        lg.info(f"clique1_consensus_distance={cd1:.3f}")
        lg.info(f"clique2_consensus_distance={cd2:.3f}")
        jlg.info(
            {
                "_meta": {"type": "clique_consensus_distance"},
                "E": E,
                "B": B,
                "clique1": cd1.item(),
                "clique2": cd2.item(),
            }
        )

        lg.info("\n")


def log_mixing_matrix(trainer, E, B):
    """Log the consensus distance among all good workers."""
    if E == 1 and B == 0:
        lg = trainer.debug_logger
        jlg = trainer.json_logger
        lg.info(f"\n=== Log mixing matrix @ E{E}B{B} ===")

        with np.printoptions(precision=3, suppress=True):
            lg.info(f"{trainer.graph.metropolis_weight}")

        lg.info("\n")


def sampler_fn(args, rank):
    assert args.n % 2 == 0
    assert args.f % 2 == 0
    num_replicas = args.n - args.f

    if args.noniid == 0:
        return lambda x: DecentralizedMixedSampler(
            noniid_percent=0,
            num_replicas=num_replicas,
            rank=rank % num_replicas,
            shuffle=True,
            dataset=x,
        )

    # Regular workers
    if rank < num_replicas:
        return lambda x: DumbbellSampler(dataset=x, noniid_percent=args.noniid,
                                         num_replicas=num_replicas, rank=rank, shuffle=True)

    # Byzantine workers
    if rank < num_replicas + args.f // 2:
        # Clique 1 data
        rank = (rank - num_replicas) % (num_replicas // 2)
    else:
        rank = (rank - num_replicas - args.f //
                2) % (num_replicas // 2) + num_replicas // 2
    return lambda x: DumbbellSampler(dataset=x, noniid_percent=args.noniid,
                                     num_replicas=num_replicas, rank=rank, shuffle=True)


def trainer_fn(args, metrics):
    if args.agg.startswith("mozi"):
        trainer_cls = DummyDecentralizedTrainer
    else:
        trainer_cls = DecentralizedTrainer

    return trainer_cls(
        pre_batch_hooks=[],
        post_batch_hooks=[
            check_noniid_hooks,
            # log_global_consensus_distance,
            # log_clique_consensus_distance,
            # log_mixing_matrix,
        ],
        max_batches_per_epoch=args.max_batch_size_per_epoch,
        log_interval=args.log_interval,
        metrics=metrics,
        use_cuda=args.use_cuda,
        debug=args.debug,
    )


def bucketing_wrapper(aggregator, s):
    """
    Key functionality.
    """
    print("Using bucketing wrapper.")

    def aggr(local_inputs, neighbor_inputs):
        inputs = [local_inputs] + neighbor_inputs
        indices = list(i % len(inputs) for i in range(len(inputs) * s))
        np.random.shuffle(indices)

        n = len(indices)
        T = int(np.ceil(n / s))

        reshuffled_inputs = []
        for t in range(T):
            indices_slice = indices[t * s: (t + 1) * s]
            g_bar = sum(inputs[i] for i in indices_slice) / len(indices_slice)
            reshuffled_inputs.append(g_bar)
        return aggregator(reshuffled_inputs[0], reshuffled_inputs[1:])

    return aggr


def bucketing_aggregator(args, graph, rank, worker):
    if args.agg.endswith("bucketing"):
        _args = copy.deepcopy(args)
        _args.agg = args.agg[:-9]
        agg = get_aggregator(_args, graph, rank, worker)
        return bucketing_wrapper(agg, s=2)
    return get_aggregator(args, graph, rank, worker)


class DumbbellImprovementRunner(MNISTTemplate):
    EXP_PATTERN = (
        "n{n}f{f}ATK{attack}_noniid{noniid}_agg{agg}_lr{lr:.3e}_m{momentum:.3e}_{graph}"
    )
    LOG_DIR_PATTERN = (
        MNISTTemplate.ROOT_DIR +
        "outputs/{script}/{exp_id}/" + EXP_PATTERN + "/"
    )

#     DEFAULT_LINE_ARG = """--lr 0.01 --use-cuda --debug -n 12 -f 2 --epochs 30 --momentum 0.9 \
# --batch-size 32 --max-batch-size-per-epoch 9999 --graph dumbbell5,1,0 --noniid 0 --agg gossip_avg \
# --identifier demo --attack BF"""
    DEFAULT_LINE_ARG = """--lr 0.01 --use-cuda --debug -n 10 -f 0 --epochs 30 --momentum 0.9 \
--batch-size 32 --max-batch-size-per-epoch 9999 --graph dumbbell5,0,0 --noniid 0 --agg gossip_avg \
--identifier demo --attack NA"""

    def __init__(
        self,
        parser_func=define_parser,
        trainer_fn=trainer_fn,
        sampler_fn=sampler_fn,
        lr_scheduler_fn=lambda opt: torch.optim.lr_scheduler.MultiStepLR(
            opt, milestones=[], gamma=1.0
        ),
        task=MNISTTask,
        worker_fn=lambda args, trainer, rank, model, opt, loss_func, m, loader, device, lr_scheduler: SGDMWorker(
            momentum=m,
            index=rank,
            data_loader=loader,
            model=model,
            optimizer=opt,
            loss_func=loss_func,
            device=device,
            lr_scheduler=lr_scheduler,
        )
        if rank < args.n - args.f
        else get_attackers(
            args, rank, trainer, model, opt, loss_func, loader, device, lr_scheduler
        ),
        evaluators_fn=lambda args, task, trainer, test_loader, device: [
            AverageEvaluator(
                # NOTE: as there is no Byzantine workers.
                models=[
                    w.model
                    for w in trainer.workers
                    if not isinstance(w, ByzantineWorker)
                ],
                data_loader=test_loader,
                loss_func=task.loss_func(device),
                device=device,
                metrics=task.metrics(),
                use_cuda=args.use_cuda,
                debug=args.debug,
                meta={"type": "Global Average Validation Accuracy"},
            ),
            # NOTE: evaluate the average accuracy inside clique 1
            AverageEvaluator(
                models=[trainer.workers[i].model for i in trainer.graph.clique1()],
                data_loader=test_loader,
                loss_func=task.loss_func(device),
                device=device,
                metrics=task.metrics(),
                use_cuda=args.use_cuda,
                debug=args.debug,
                meta={"type": "Clique1 Average Validation Accuracy"},
            ),
            # NOTE: evaluate the average accuracy inside clique 2
            AverageEvaluator(
                models=[trainer.workers[i].model for i in trainer.graph.clique2()],
                data_loader=test_loader,
                loss_func=task.loss_func(device),
                device=device,
                metrics=task.metrics(),
                use_cuda=args.use_cuda,
                debug=args.debug,
                meta={"type": "Clique2 Average Validation Accuracy"},
            ),
        ],
    ):
        super().__init__(
            parser_func=parser_func,
            trainer_fn=trainer_fn,
            sampler_fn=sampler_fn,
            lr_scheduler_fn=lr_scheduler_fn,
            task=task,
            worker_fn=worker_fn,
            evaluators_fn=evaluators_fn,
            get_graph=gu.get_graph,
            get_aggregator=bucketing_aggregator
        )

    def run(self):
        if self.args.analyze:
            if self.args.identifier == "exp":
                self.generate_analysis()
        else:
            self.train()

    def generate_analysis(self):
        out_dir = os.path.abspath(os.path.join(self.log_dir, os.pardir))
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        mapping = {
            "gossip_avg": "Gossip",
            "cp1": "CP",
            "scp1": "SCClip",
            "rfa8": "GM",
            "mozi0.4,0.5": "MOZI",
            "mozi0.99,0.5": "MOZI",
            "cm": "CM",
            "tm1": "TM",
            "krum1": "Krum",
            "gossip_avgbucketing": r"Gossip",
            "mozi0.99,0.5bucketing": r"MOZI",
            "tm1bucketing": r"TM",
            "rfa8bucketing": r"GM",
            "scp1bucketing": r"SCClip",
        }

        acc_results = []
        for agg in [
            "gossip_avg",
            "rfa8",
            "mozi0.99,0.5",
            "tm1",
            "scp1",
            "gossip_avgbucketing",
            "rfa8bucketing",
            "mozi0.99,0.5bucketing",
            "tm1bucketing",
            "scp1bucketing",
        ]:
            for r in [0, 1]:
                _log_dir = self.log_dir.replace(
                    "agg{}_".format(self.args.agg), "agg{}_".format(agg)
                )
                log_dir = _log_dir.replace(
                    "{}".format(self.args.graph),
                    "dumbbell10,0,{}".format(r),
                )
                path = log_dir + "stats"
                # Extract results for global accuracy
                try:
                    values = filter_entries_from_json(
                        path, kw="Global Average Validation Accuracy"
                    )
                    for v in values:
                        it = (v["E"] - 1) * \
                            self.args.max_batch_size_per_epoch
                        acc_results.append(
                            {
                                "Iterations": it,
                                "Accuracy (%)": v["top1"],
                                "Agg": mapping[agg],
                                "Bucketing": agg.endswith("bucketing"),
                                "RandomEdge": bool(r),
                                "Group": "global",
                            }
                        )
                except Exception as e:
                    raise NotImplementedError(f"agg={agg}")

                # Extract results for local accuracy
                try:
                    values = filter_entries_from_json(
                        path, kw="Clique1 Average Validation Accuracy"
                    )
                    for v in values:
                        it = (v["E"] - 1) * \
                            self.args.max_batch_size_per_epoch
                        acc_results.append(
                            {
                                "Iterations": it,
                                "Accuracy (%)": v["top1"],
                                "Agg": mapping[agg],
                                "Bucketing": agg.endswith("bucketing"),
                                "RandomEdge": bool(r),
                                "Group": "clique 1",
                            }
                        )
                except Exception as e:
                    raise NotImplementedError(f"agg={agg}")

                # Extract results for local accuracy
                try:
                    values = filter_entries_from_json(
                        path, kw="Clique2 Average Validation Accuracy"
                    )
                    for v in values:
                        it = (v["E"] - 1) * \
                            self.args.max_batch_size_per_epoch
                        acc_results.append(
                            {
                                "Iterations": it,
                                "Accuracy (%)": v["top1"],
                                "Agg": mapping[agg],
                                "Bucketing": agg.endswith("bucketing"),
                                "RandomEdge": bool(r),
                                "Group": "clique 2",
                            }
                        )
                except Exception as e:
                    raise NotImplementedError(f"agg={agg}")

        acc_df = pd.DataFrame(acc_results)
        acc_df.to_csv(out_dir + "/acc.csv", index=None)


if __name__ == "__main__":
    runner = DumbbellImprovementRunner()
    runner.run()
