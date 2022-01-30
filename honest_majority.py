import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import numpy as np

import codes.graph_utils as gu
from codes.simulator import ByzantineWorker
from codes.utils import filter_entries_from_json
from codes.attacks import get_attackers

from template import MNISTTemplate, MNISTTask
from template import (
    define_parser,
    DecentralizedTrainer,
    check_noniid_hooks,
    get_sampler_callback,
    SGDMWorker,
    AverageEvaluator,
)

LOG_CONSENSUS_DISTANCE_INTERVAL = 10


class MaliciousRing(gu.Graph):
    """
    There are totally 3g+1 nodes in the network.
    Node i=0, 1, ...., g-1 are good and they form a ring
    Node i+g, i+2g are the Byzantine neighbors of node i.
    In addition Byzantine node 3g is connected to node 0.
    """

    def __init__(self, g, honest_majority=False, normalize=False):
        edges = [(i, i + 1) for i in range(g - 1)] + [(g - 1, 0)]
        for i in range(g):
            edges.append((i, i + g))
            edges.append((i, i + 2 * g))

        if not honest_majority:
            edges.append((0, 3 * g))
            super().__init__(3 * g + 1, edges)

            if normalize:
                min_local_weight = np.diag(self.metropolis_weight)[:g].min()
                self.metropolis_weight += np.eye(3 * g + 1) * \
                    (1 - min_local_weight)
                self.metropolis_weight /= 1 + (1 - min_local_weight)
        else:
            super().__init__(3 * g, edges)

            if normalize:
                min_local_weight = np.diag(self.metropolis_weight)[:g].min()
                self.metropolis_weight += np.eye(3 * g) * \
                    (1 - min_local_weight)
                self.metropolis_weight /= 1 + (1 - min_local_weight)


def get_graph(args):
    if args.graph.startswith("mr"):
        # Pattern: twocliques2,1 for n=2 m=1
        g, normalize, honest_majority = args.graph[len("mr"):].split(",")
        g, normalize, honest_majority = int(g), bool(
            int(normalize)), bool(int(honest_majority))

        assert args.n == 3 * g + (0 if honest_majority else 1)
        return MaliciousRing(g, honest_majority=honest_majority, normalize=normalize)

    return gu.get_graph(args)


# ---------------------------------------------------------------------------- #
#                                     Hooks                                    #
# ---------------------------------------------------------------------------- #


def log_mixing_matrix(trainer, E, B):
    """Log the consensus distance among all good workers."""
    if E == 1 and B == 0:
        lg = trainer.debug_logger
        jlg = trainer.json_logger
        lg.info(f"\n=== Log mixing matrix @ E{E}B{B} ===")

        with np.printoptions(precision=3, suppress=True):
            lg.info(f"{trainer.graph.metropolis_weight}")

        lg.info("\n")


class HonestMajorityRunner(MNISTTemplate):
    EXP_PATTERN = (
        "n{n}f{f}ATK{attack}_noniid{noniid}_agg{agg}_lr{lr:.3e}_m{momentum:.3e}_{graph}"
    )
    LOG_DIR_PATTERN = (
        MNISTTemplate.ROOT_DIR +
        "outputs/{script}/{exp_id}/" + EXP_PATTERN + "/"
    )

    DEFAULT_LINE_ARG = """--lr 0.01 --use-cuda --debug -n 12 -f 1 --epochs 30 --momentum 0.0 \
--batch-size 32 --max-batch-size-per-epoch 9999 --graph tcb5,1 --noniid 0 --agg gossip_avg \
--identifier demo --attack BF"""

    def __init__(
        self,
        parser_func=define_parser,
        trainer_fn=lambda args, metrics: DecentralizedTrainer(
            pre_batch_hooks=[],
            post_batch_hooks=[
                check_noniid_hooks,
                log_mixing_matrix,
            ],
            max_batches_per_epoch=args.max_batch_size_per_epoch,
            log_interval=args.log_interval,
            metrics=metrics,
            use_cuda=args.use_cuda,
            debug=args.debug,
        ),
        sampler_fn=lambda args, rank: get_sampler_callback(
            rank, args.n, noniid=args.noniid, longtail=args.longtail
        ),
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
            )
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
            get_graph=get_graph,
        )

    def run(self):
        if self.args.analyze:
            if self.args.identifier == "exp":
                self.generate_analysis()
            else:
                raise NotImplementedError(self.args.identifier)
        else:
            self.train()

    # ---------------------------------------------------------------------------- #
    #                                 Plot for HonestMajorityRunner                                #
    # ---------------------------------------------------------------------------- #
    def generate_analysis(self):
        out_dir = os.path.abspath(os.path.join(self.log_dir, os.pardir))
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        def loop_files():
            for agg, agg_name in [
                # ("cp0.1", "CP"),
                ("scp0.1", "SCClip"),
                ("rfa8", "GM"),
                ("mozi0.4,0.5", "MOZI"),
                    ("tm2", "TM")]:
                for attack, atk_name in [
                    # ("BF", "BF"),
                    # ("LF", "LF"),
                    ("IPM", "IPM"),
                    ("ALIE10", "ALIE"),
                    ("dissensus1.5", "Dissensus"),
                ]:
                    for honest_majority in [False, True]:
                        if honest_majority:
                            n = 15
                            f = 10
                            graph = "mr5,1,1"
                        else:
                            n = 16
                            f = 11
                            graph = "mr5,1,0"
                        log_dir = self.LOG_DIR_PATTERN.format(
                            script=sys.argv[0][:-3],
                            exp_id=self.args.identifier,
                            n=n,
                            f=f,
                            graph=graph,
                            attack=attack,
                            agg=agg,
                            lr=1e-2,
                            noniid=0.0,
                            momentum=0.9,
                        )
                        path = log_dir + "stats"
                        yield agg, agg_name, attack, atk_name, path, honest_majority

        # Plot for accuracy
        acc_results = []
        for agg, agg_name, attack, atk_name, path, honest_majority in loop_files():
            # Add global accuracies
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
                            "Agg": agg_name,
                            "ATK": atk_name,
                            "Group": "global",
                            "H.M.E.": honest_majority
                        }
                    )
            except Exception as e:
                raise NotImplementedError(
                    f"agg={agg} attack={attack}"
                )
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
                            "Agg": agg_name,
                            "ATK": atk_name,
                            "Member": "clique 1",
                            "H.M.E.": honest_majority
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
                    it = (v["E"] - 1) * self.args.max_batch_size_per_epoch
                    acc_results.append(
                        {
                            "Iterations": it,
                            "Accuracy (%)": v["top1"],
                            "Agg": agg_name,
                            "ATK": atk_name,
                            "Group": "clique 2",
                            "H.M.E.": honest_majority
                        }
                    )
            except Exception as e:
                raise NotImplementedError(f"agg={agg}")

        acc_df = pd.DataFrame(acc_results)
        acc_df.to_csv(out_dir + "/acc.csv", index=None)

        plt.rcParams["font.family"] = "Times New Roman"
        plt.rcParams["font.size"] = 1
        plt.rcParams["legend.columnspacing"] = 0.5
        plt.rcParams["legend.handlelength"] = 1
        plt.rcParams["legend.borderaxespad"] = 0
        plt.rcParams["legend.labelspacing"] = 0.1

        sns.set(rc={'figure.figsize': (6.75, 6.75 / 2)})
        sns.set(font_scale=0.7)
        plt.figure()
        h = 1.8
        g = sns.relplot(
            data=acc_df[acc_df["Group"] == "global"],
            x="Iterations",
            y="Accuracy (%)",
            col="ATK",
            # row="H.M.E.",
            style="H.M.E.",
            hue="Agg",
            kind="line",
            ci=None,
            height=h,
            aspect=1.3/h,
        )
        g.set(xlim=(0, 900))
        g.set(ylim=(0, 100))

        a = g.legend

        for i in range(3):
            g.axes[0][i].tick_params(axis='both', which='major', pad=-4)
        g.axes[0][0].set_ylabel('Accuracy (%)', labelpad=-2)

        handles, labels = g.axes[0][0].get_legend_handles_labels()
        g.legend.remove()
        g.axes[0][0].legend(handles[:5], labels[:5], ncol=5, loc='lower center',
                            columnspacing=0.5, handlelength=1, borderaxespad=0, labelspacing=0.,
                            bbox_to_anchor=(0.4, 1.14, 2.5, 0.2), frameon=False)
        g.axes[0][2].legend(handles[5:], labels[5:], ncol=1, loc='upper right',
                            columnspacing=0.5, handlelength=1, borderaxespad=0, labelspacing=0.1,
                            bbox_to_anchor=(1., 0.6), frameon=False)

        g.fig.savefig(out_dir + "/acc.pdf", bbox_inches="tight", dpi=720)


if __name__ == "__main__":
    runner = HonestMajorityRunner()
    runner.run()
