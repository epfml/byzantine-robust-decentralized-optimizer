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


def get_graph(args):
    if args.graph.startswith("tcb"):
        # Pattern: twocliques2,1 for n=2 m=1
        m, b, delta = args.graph[len("tcb"):].split(",")
        m, b, delta = int(m), int(b), float(delta)
        assert args.n == 2 * m + 1 + b
        return gu.TwoCliquesWithByzantine(m, b, delta)

    return gu.get_graph(args)


# ---------------------------------------------------------------------------- #
#                                     Hooks                                    #
# ---------------------------------------------------------------------------- #


def log_global_consensus_distance(trainer, E, B):
    """Log the consensus distance among all good workers."""
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


class OptimizationDeltaRunner(MNISTTemplate):
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
                log_global_consensus_distance,
                log_clique_consensus_distance,
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
    #                                 Plot for OptimizationDeltaRunner                                #
    # ---------------------------------------------------------------------------- #

    def generate_analysis(self):
        out_dir = os.path.abspath(os.path.join(self.log_dir, os.pardir))
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        mapping_attack = {
            "LF": "LF",
            "ALIE10": "ALIE",
            "IPM": "IPM",
            "dissensus1.5": "Dissensus",
            "BF": "BF",
        }

        def loop_files():
            b = 1
            # for delta in [0, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 1]:
            for delta in [0, 0.25, 0.5, 0.75, 1]:
                # for attack in ["LF", "BF", "ALIE10", "IPM", "dissensus1.5"]:
                for attack in ["dissensus1.5"]:
                    log_dir = self.LOG_DIR_PATTERN.format(
                        script=sys.argv[0][:-3],
                        exp_id=self.args.identifier,
                        n=11 + b,
                        f=b,
                        attack=attack,
                        noniid=1.0,
                        agg="scp1",
                        lr=1e-3,
                        momentum=0.9,
                        graph=f"tcb5,1,{delta}",
                    )
                    path = log_dir + "stats"
                    yield b, delta, attack, path

        # Plot for accuracy
        acc_results = []
        for b, delta, attack, path in loop_files():
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
                            r"$\delta_{\max}$": str(delta * b / (b + 3)),
                            "ATK": mapping_attack[attack],
                            "b": b,
                            "Group": "All",
                        }
                    )
            except Exception as e:
                raise NotImplementedError(
                    f"attack={attack} b={b} delta={delta}")

            # Extract results for local accuracy
            for clique_id, clique_name in [(1, 'A'), (2, 'B')]:
                try:
                    values = filter_entries_from_json(
                        path, kw=f"Clique{clique_id} Average Validation Accuracy"
                    )
                    for v in values:
                        it = (v["E"] - 1) * self.args.max_batch_size_per_epoch
                        acc_results.append(
                            {
                                "Iterations": it,
                                "Accuracy (%)": v["top1"],
                                r"$\delta_{\max}$": str(delta * b / (b + 3)),
                                "ATK": mapping_attack[attack],
                                "b": b,
                                "Group": f"Clique {clique_name}",
                            }
                        )
                except Exception as e:
                    raise NotImplementedError(
                        f"clique_id={clique_id} attack={attack} b={b} delta={delta}"
                    )

        acc_df = pd.DataFrame(acc_results)
        acc_df.to_csv(out_dir + "/acc.csv", index=None)

        acc_df[r"$\delta_{\max}$ "] = acc_df[r"$\delta_{\max}$"].apply(
            lambda x: float(x))

        plt.rcParams["font.family"] = "Times New Roman"
        sns.set(rc={'figure.figsize': (6, 6.75 / 3)})
        sns.set(font_scale=1)

        fig, axes = plt.subplots(nrows=1, ncols=2, sharey=True)
        g = sns.lineplot(
            data=acc_df,
            x="Iterations",
            y="Accuracy (%)",
            hue=r"$\delta_{\max}$",
            style="Group",
            ax=axes[0],
        )
        g.set(xlim=(0, 1500))
        g.set(xlim=(0, 1500))

        g.set_xticks([0, 500, 1000])
        g.set_xticklabels([0, 500, 1000])

        axes[0].legend(loc="lower left",  ncol=6, columnspacing=0.5, handlelength=1,
                       borderaxespad=0, labelspacing=0.1, bbox_to_anchor=(1, 1.02, 1, 0.2))

        handles, labels = axes[0].get_legend_handles_labels()
        g.legend().remove()
        axes[0].legend(handles[:6], labels[:6], ncol=6, loc='lower center',
                       columnspacing=0.5, handlelength=1, borderaxespad=0, labelspacing=0.,
                       bbox_to_anchor=(0.4, 1.02, 1, 0.2), frameon=False)

        last_iterate = acc_df[acc_df['Iterations'] == 1470]
        g = sns.lineplot(
            data=last_iterate,
            x=r"$\delta_{\max}$ ",
            y="Accuracy (%)",
            style="Group",
            ax=axes[1],
            hue='ATK',
            palette=['black']
        )
        g.set(xlim=(0, 0.25))
        axes[1].legend(handles[6:], labels[6:], ncol=1, loc='upper right',
                       columnspacing=0.5, handlelength=1, borderaxespad=0, labelspacing=0.1,
                       bbox_to_anchor=(1., 1), frameon=False)

        for i in range(2):
            axes[i].tick_params(axis='both', which='major', pad=-4)
        axes[0].set_ylabel('Accuracy (%)', labelpad=0)
        axes[1].set_ylabel('', labelpad=-1)

        fig.subplots_adjust(wspace=0.093)

        fig.savefig(out_dir + "/acc.pdf",
                    bbox_inches="tight", dpi=720)


if __name__ == "__main__":
    runner = OptimizationDeltaRunner()
    runner.run()
