# Byzantine-robust decentralized learning via self-centered clipping

In this paper, we study the challenging task of Byzantine-robust decentralized training on arbitrary communication graphs. Unlike federated learning where workers communicate through a server, workers in the decentralized environment can only talk to their neighbors, making it harder to reach consensus. We identify a novel *dissensus* attack in which few malicious nodes can take advantage of information bottlenecks in the topology to poison the collaboration. To address these issues, we propose a Self-Centered Clipping (*SSClip*) algorithm for Byzantine-robust consensus and optimization, which is the first to provably converge to a $O(\delta_{\max}\zeta^2/\gamma^2)$ neighborhood of the stationary point for non-convex objectives under standard assumptions. Finally, we demonstrate the encouraging empirical performance of *SSClip* under a large number of attacks.


# Table of contents

- [Structure of code](#Code-organization)
- [Reproduction](#Reproduction)
- [License](#license)
- [Reference](#Reference)

# Code organization

The structure of the repository is as follows:
- `codes/`
  - Source code.
- `outputs/`
  - Store the output of the launcher scripts.
- `consensus.ipynb`: Study the error of aggregators to the average consensus under dissensus attack.
    - This notebook generates Fig. 3 in the main text and Fig. 8 in the appendix.
- `dumbbell.py`: Study how topology + heterogeneity influence on the aggregators.
- `dumbbell_improvement.py`: Study how to help aggregators to address topology + heterogeneity influence.
- `dumbbell.ipynb`: Plot the results of `dumbbell.py` and `dumbbell_improvement.py`.
    - Generate Fig. 4 in the main text.
- `optimization_delta.py`: Fix `p`, `zeta^2` and varying delta of dissensus attack for SCClip aggregator.
    - Generate Fig. 5 in the main text.
- `honest_majority.py`: Study the influence of honest majority in the text.
    - Generate Fig. 6 in the main text.

- `dumbbell_CIFAR.py` and `run_plot_CIFAR.ipynb`:
    - Generate Fig. 11 in the appendix.

# Reproduction

To reproduce the results in the paper, do the following steps

1. Add `codes/` to environment variable `PYTHONPATH`
2. Install the dependencies: `pip install -r requirements.txt`
3. Run `bash run.sh` and select option 2 to 9 to generate the code.
4. The output will be saved to the corresponding folders under `outputs`

Note that if the GPU memory is small (e.g. less than 16 GB), then running the previous commands may raise insufficient exception. In this case, one can decrease the level parallelism in the script by changing the order of loops and reduce the number of parallel processes. 


# License

This repo is covered under [The MIT License](LICENSE).


# Reference

TODO