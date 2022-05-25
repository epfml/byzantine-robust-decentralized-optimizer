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

# from template import CIFAR10_Template, MNISTTask
from template import CIFAR10_Template, CIFAR10Task
from template import (
    define_parser,
    DecentralizedTrainer,
    check_noniid_hooks,
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


class DumbbellCIFARRunner(CIFAR10_Template):
    EXP_PATTERN = (
        "n{n}f{f}ATK{attack}_noniid{noniid}_agg{agg}_lr{lr:.3e}_m{momentum:.3e}_{graph}"
    )
    LOG_DIR_PATTERN = (
        CIFAR10_Template.ROOT_DIR +
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
            opt, milestones=[80, 120], gamma=0.1),
        task=CIFAR10Task,
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
        )

    def run(self):
        if self.args.analyze:
            if self.args.identifier == "exp":
                self.generate_analysis_exp()
        else:
            self.train()

    def generate_analysis_exp(self):
        out_dir = os.path.abspath(os.path.join(self.log_dir, os.pardir))
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        mapping = {
            "gossip_avg": "Gossip",
            "cp1": "CP",
            "scp1": "SCClip",
            "scp10": "SCClip",
            "rfa8": "GM",
            "mozi0.4,0.5": "MOZI",
            "mozi1,0.5": "MOZI",
            "cm": "CM",
            "tm1": "TM",
            "krum1": "Krum"
        }

        acc_results = []
        for agg in [
            "gossip_avg",
            # "cp1",
            "rfa8",
            "mozi0.4,0.5",
            # "mozi1,0.5",
            # "cm",
            "tm1",
            "scp10",
            # "krum1"
        ]:
            _log_dir = self.log_dir.replace(
                "agg{}_".format(self.args.agg), "agg{}_".format(agg)
            )
            for noniid in [1.0, 0.0]:
                log_dir = _log_dir.replace(
                    "noniid{}_".format(self.args.noniid),
                    "noniid{}_".format(noniid),
                )
                path = log_dir + "stats"
                # Extract results for global accuracy
                try:
                    values = filter_entries_from_json(
                        path, kw="Global Average Validation Accuracy"
                    )
                    for v in values:
                        it = (v["E"] - 1) * self.args.max_batch_size_per_epoch
                        acc_results.append(
                            {
                                "Iterations": it,
                                "Accuracy (%)": v["top1"],
                                "Agg": mapping[agg],
                                "NonIID": bool(noniid),
                                "Group": "global",
                            }
                        )
                except Exception as e:
                    raise e
                    raise NotImplementedError(
                        f"agg={agg} noniid={noniid} {log_dir}\n{e}")

                # Extract results for local accuracy
                try:
                    values = filter_entries_from_json(
                        path, kw="Clique1 Average Validation Accuracy"
                    )
                    for v in values:
                        it = (v["E"] - 1) * self.args.max_batch_size_per_epoch
                        acc_results.append(
                            {
                                "Iterations": it,
                                "Accuracy (%)": v["top1"],
                                "Agg": mapping[agg],
                                "NonIID": bool(noniid),
                                "Group": "clique 1",
                            }
                        )
                except Exception as e:
                    raise NotImplementedError(f"agg={agg} noniid={noniid}")

                # Extract results for local accuracy
                try:
                    values = filter_entries_from_json(
                        path, kw="Clique2 Average Validation Accuracy"
                    )
                    for v in values:
                        it = (v["E"] - 1) * self.args.max_batch_size_per_epoch
                        acc_results.append(
                            {
                                "Iterations": it,
                                "Accuracy (%)": v["top1"],
                                "Agg": mapping[agg],
                                "NonIID": bool(noniid),
                                "Group": "clique 2",
                            }
                        )
                except Exception as e:
                    raise NotImplementedError(f"agg={agg} noniid={noniid}")

        acc_df = pd.DataFrame(acc_results)
        acc_df.to_csv(out_dir + "/CIFAR_acc.csv", index=None)

        # fig, ax = plt.subplots(1, 1, figsize=(4, 2))
        # ax.set_yscale('log')
        plt.rcParams["font.family"] = "Times New Roman"
        plt.rcParams["font.size"] = 1
        sns.set(rc={'figure.figsize': (6.75, 6.75 / 2)})
        plt.figure()
        g = sns.relplot(
            data=acc_df[acc_df["Group"] == "clique 1"],
            x="Iterations",
            y="Accuracy (%)",
            col="NonIID",
            hue="Agg",
            # style="Group",
            # ax=ax,
            kind="line",
            ci=None,
            # height=2.2,
            height=2.5,
            aspect=1,
            # palette=sns.color_palette("Set1", 4),
        )
        g.set(xlim=(0, 900))
        g.set(ylim=(45, 101))
        # g.get_xaxis().set_ticklabels([])

        # g.set(yscale="log")
        # # Put the legend out of the figure
        # g._legend(loc="center left", bbox_to_anchor=(1, 0.5))
        for i in range(2):
            g.axes[0][i].tick_params(axis='both', which='major', pad=-4)
        g.axes[0][0].set_ylabel('Accuracy (%)', labelpad=-2)
        g._legend.set_bbox_to_anchor([0.44, 0.5])

        print(out_dir)
        g.fig.savefig(out_dir + "/CIFAR_acc.pdf",
                      bbox_inches="tight", dpi=720)


if __name__ == "__main__":
    runner = DumbbellCIFARRunner()
    runner.run()
