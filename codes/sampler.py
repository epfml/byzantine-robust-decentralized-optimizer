import math
import torch
from torch.utils.data.sampler import Sampler
import torch.distributed as dist


class DistributedSampler(Sampler):
    """Sampler that restricts data loading to a subset of the dataset.

    It is especially useful in conjunction with
    :class:`torch.nn.parallel.DistributedDataParallel`. In such case, each
    process can pass a DistributedSampler instance as a DataLoader sampler,
    and load a subset of the original dataset that is exclusive to it.

    .. note::
        Dataset is assumed to be of constant size.

    Arguments:
        dataset: Dataset used for sampling.
        num_replicas (optional): Number of processes participating in
            distributed training.
        rank (optional): Rank of the current process within num_replicas.
        shuffle (optional): If true (default), sampler will shuffle the indices
    """

    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True):
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError(
                    "Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError(
                    "Requires distributed package to be available")
            rank = dist.get_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.num_samples = int(
            math.ceil(len(self.dataset) * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas
        self.shuffle = shuffle

    def __iter__(self):
        # deterministically shuffle based on epoch
        g = torch.Generator()
        g.manual_seed(self.epoch)
        if self.shuffle:
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))

        # add extra samples to make it evenly divisible
        indices += indices[: (self.total_size - len(indices))]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank: self.total_size: self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __str__(self):
        return "DistributedSampler(num_replicas={num_replicas},rank={rank},shuffle={shuffle})".format(
            num_replicas=self.num_replicas, rank=self.rank, shuffle=self.shuffle
        )


class DecentralizedNonIIDSampler(DistributedSampler):
    def __iter__(self):
        nlabels = len(self.dataset.classes)

        # deterministically shuffle based on epoch
        g = torch.Generator()
        g.manual_seed(0)

        indices = []
        for i in range(nlabels):
            indices_i = torch.nonzero(self.dataset.targets == i)

            indices_i = indices_i.flatten().tolist()
            indices += indices_i

        # add extra samples to make it evenly divisible
        indices += indices[: (self.total_size - len(indices))]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[
            self.rank * self.num_samples: (self.rank + 1) * self.num_samples
        ]
        assert len(indices) == self.num_samples

        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.epoch)
            idx_idx = torch.randperm(len(indices), generator=g).tolist()
            indices = [indices[i] for i in idx_idx]

        return iter(indices)

    def __str__(self):
        return "DecentralizedNonIIDSampler(num_replicas={num_replicas},rank={rank},shuffle={shuffle})".format(
            num_replicas=self.num_replicas, rank=self.rank, shuffle=self.shuffle
        )


class DecentralizedMixedSampler(DistributedSampler):
    def __init__(self, noniid_percent, *args, **kwargs):
        super(DecentralizedMixedSampler, self).__init__(*args, **kwargs)
        self.noniid_percent = noniid_percent

    def __iter__(self):
        nlabels = len(self.dataset.classes)

        # deterministically shuffle based on epoch
        g = torch.Generator()
        g.manual_seed(0)

        all_indices = torch.randperm(len(self.dataset), generator=g).tolist()

        iid_count = int((1 - self.noniid_percent) * len(all_indices))
        iid_count = iid_count - (iid_count % self.num_replicas)
        iid_indices, noniid_indices = all_indices[:
                                                  iid_count], all_indices[iid_count:]

        indices = []
        for i in range(nlabels):
            indices_i = torch.nonzero(self.dataset.targets == i)
            indices_i = indices_i.flatten().tolist()
            # Find those in the noniid parts
            indices_i = set(indices_i).intersection(set(noniid_indices))
            indices += indices_i

        # add extra samples to make it evenly divisible
        indices += indices[: (self.total_size - iid_count - len(indices))]
        assert len(indices) + iid_count == self.total_size

        # subsample
        num_noniid_samples_per_node = self.num_samples - iid_count // self.num_replicas
        indices = indices[
            self.rank
            * num_noniid_samples_per_node: (self.rank + 1)
            * num_noniid_samples_per_node
        ]
        # Add iid part
        indices += iid_indices[self.rank: iid_count: self.num_replicas]
        assert len(indices) == self.num_samples, (len(
            indices), self.num_samples)

        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.epoch)
            idx_idx = torch.randperm(len(indices), generator=g).tolist()
            indices = [indices[i] for i in idx_idx]

        return iter(indices)


class NONIIDLTSampler(DistributedSampler):
    """NONIID + Long-Tail sampler.

    alpha: alpha controls the noniidness.
        - alpha = 0 refers to completely noniid
        - alpha = 1 refers to iid.

    beta: beta controls the long-tailness.
        - Class i takes beta ** i percent of data.
    """

    def __init__(self, alpha, beta, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.alpha = alpha
        self.beta = beta
        assert beta >= 0 and beta <= 1
        assert alpha >= 0

    def __iter__(self):
        # The dataset are not shuffled across nodes.
        g = torch.Generator()
        g.manual_seed(0)

        if self.shuffle:
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))

        nlabels = len(self.dataset.classes)
        indices = []
        for i in range(nlabels):
            label_indices = torch.nonzero(self.dataset.targets == i)
            label_indices = label_indices.flatten().tolist()
            label_selected = int(len(label_indices) * self.beta ** i)
            # discard the rest of label_indices[label_selected:]
            indices += label_indices[:label_selected]

        # Adjust after removing data points.
        self.num_samples = int(
            math.ceil(len(indices) * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas

        # add extra samples to make it evenly divisible
        indices += indices[: (self.total_size - len(indices))]
        assert len(indices) == self.total_size

        #
        if self.alpha:
            # IID
            indices = indices[self.rank: self.total_size: self.num_replicas]
        else:
            # NONIID
            indices = indices[
                self.rank * self.num_samples: (self.rank + 1) * self.num_samples
            ]
        assert len(indices) == self.num_samples

        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.rank ** 3 + self.epoch)
            idx_idx = torch.randperm(len(indices), generator=g).tolist()
            indices = [indices[i] for i in idx_idx]

        return iter(indices)

    def __str__(self):
        return "NONIIDLTSampler"


class DumbbellSampler(DistributedSampler):

    def __init__(self, dataset, noniid_percent, num_replicas=None, rank=None, shuffle=True):
        assert num_replicas % 2 == 0
        assert noniid_percent >= 0 and noniid_percent <= 1
        self.noniid_percent = noniid_percent
        self.clique_index = int(rank >= num_replicas // 2)
        super().__init__(dataset, num_replicas, rank, shuffle)

    def __iter__(self):
        nlabels = len(self.dataset.classes)
        clique1_labels = list(range(nlabels // 2))
        clique2_labels = list(range(nlabels // 2, nlabels))

        # deterministically shuffle based on epoch
        g = torch.Generator()
        g.manual_seed(0)

        clique1_indices = []
        for i in clique1_labels:
            indices_i = torch.nonzero(self.dataset.targets == i)
            indices_i = indices_i.flatten().tolist()
            clique1_indices += indices_i
        permuted = torch.randperm(len(clique1_indices), generator=g).tolist()
        clique1_indices = [clique1_indices[i] for i in permuted]

        clique2_indices = []
        for i in clique2_labels:
            indices_i = torch.nonzero(self.dataset.targets == i)
            indices_i = indices_i.flatten().tolist()
            clique2_indices += indices_i
        permuted = torch.randperm(len(clique2_indices), generator=g).tolist()
        clique2_indices = [clique2_indices[i] for i in permuted]

        min_size = min(len(clique1_indices), len(clique2_indices))
        self.num_samples = min_size // (self.num_replicas // 2)
        self.total_size = self.num_samples * self.num_replicas

        min_size = self.num_samples * (self.num_replicas // 2)
        clique1_indices = clique1_indices[:min_size]
        clique2_indices = clique2_indices[:min_size]

        # Randomly exchange vectors.
        exchanged = int((1 - self.noniid_percent) * min_size / 2)
        clique1_indices[:exchanged], clique2_indices[:
                                                     exchanged] = clique2_indices[:exchanged], clique1_indices[:exchanged]

        permuted = torch.randperm(len(clique1_indices), generator=g).tolist()
        clique1_indices = [clique1_indices[i] for i in permuted]
        permuted = torch.randperm(len(clique2_indices), generator=g).tolist()
        clique2_indices = [clique2_indices[i] for i in permuted]

        # TODO: Update stats
        if self.clique_index == 0:
            indices = clique1_indices[self.rank::self.num_replicas//2]
        elif self.clique_index == 1:
            indices = clique2_indices[self.rank -
                                      self.num_replicas//2::self.num_replicas//2]
        else:
            raise NotImplementedError

        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.epoch)
            idx_idx = torch.randperm(len(indices), generator=g).tolist()
            indices = [indices[i] for i in idx_idx]

        return iter(indices)


def get_sampler_callback(rank, num_replicas, noniid=0, longtail=0):
    """
    noniid: noniid controls the noniidness.
        - noniid = 1 refers to completely noniid
        - noniid = 0 refers to iid.

    longtail: longtail controls the long-tailness.
        - Class i takes (1-longtail) ** i percent of data.
    """

    assert noniid >= 0 and noniid <= 1, f"`noniid` in [0, 1], get {noniid}"
    assert longtail >= 0 and longtail <= 1, f"`longtail` in [0, 1], get {longtail}"

    if longtail > 0:
        return lambda x: NONIIDLTSampler(
            alpha=1 - noniid,
            beta=1 - longtail,
            num_replicas=num_replicas,
            rank=rank,
            shuffle=True,
            dataset=x,
        )

    if noniid == 0:
        # Byzantine workers
        return lambda x: DistributedSampler(
            num_replicas=num_replicas,
            rank=rank,
            shuffle=True,
            dataset=x,
        )

    if noniid > 0:
        return lambda x: DecentralizedMixedSampler(
            noniid_percent=noniid,
            num_replicas=num_replicas,
            rank=rank,
            shuffle=True,
            dataset=x,
        )

    raise NotImplementedError("")
