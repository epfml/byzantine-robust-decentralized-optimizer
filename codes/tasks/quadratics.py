import contextlib
import copy
import math
import numpy as np
import torch


class LinearModel(torch.nn.Module):
    def __init__(self, d):
        super(LinearModel, self).__init__()
        self.layer = torch.nn.Linear(d, 1, bias=False)

    def forward(self, x):
        return self.layer(x)


def generate_A_with_L_mu(n, d, L, mu=-1):
    """
    Generate a data matrix A for
        f(x) = \frac{1}{n} || A x - b ||_2^2
    with L-smoothnes and mu-convexity.

    The L-smoothness is the largest eigenvalue of the hessian of f
        hessian(f)(x) = \frac{2}{n} A^T A
    """
    assert mu <= L

    # Generate unitary matrix U and V
    dummy_matrix = torch.randn(n, d)
    U, _, V = torch.linalg.svd(dummy_matrix)

    # Construct matrix S such that S.T @ S has largest elements L
    # and smallest elements mu.
    smallest = math.sqrt(abs(mu) * n / 2)
    largest = math.sqrt(L * n / 2)
    diag = torch.linspace(start=smallest, end=largest, steps=min(n, d))
    S = torch.zeros(n, d)
    S[list(range(d)), list(range(d))] = diag

    # Reconstruct A
    return U @ S @ V.T


def generate_synthetic_dataset(n, d, L, mu, sigma):
    """
    The optimum model is zeros.
    """
    A = generate_A_with_L_mu(n, d, L, mu)
    x_opt = torch.zeros(d)

    def _add_noise_to_target():
        # Generate a random noise and compute its variance
        b = torch.randn(n)
        _2Atb = 2 * A.T @ torch.diag(b)
        # The variance over `n` and sum over `d`.
        b_sigma = _2Atb.var(axis=1).sum().sqrt()
        # Rescale the noise to the specific sigma
        return sigma / b_sigma * b

    b = _add_noise_to_target()
    return A, b, x_opt


def generate_synthetic_distributed_dataset(m, n, d, L, mu, sigma, zeta):
    """
    Create datasets for `m` workers, each having `n` samples.

    Note the L and mu is for each worker not for all workers.
    """
    # The `A` here
    A = torch.cat([generate_A_with_L_mu(n, d, L, mu) for _ in range(m)])

    # Create noise as heterogeneity
    b = []
    xi_stars = []
    for i in range(m):
        Ai = A[i * n: (i + 1) * n, :]
        xi_star = torch.randn(d) + i
        bi = Ai @ xi_star
        b.append(bi)
        xi_stars.append(xi_star)
    b = torch.cat(b)

    x_opt = torch.Tensor(np.linalg.solve(A.T @ A, A.T @ b))
    zeta2_ = 0
    for i in range(m):
        Ai = A[i * n: (i + 1) * n, :]
        xi_star = xi_stars[i]
        zeta_i2 = (2 / n * Ai.T @ Ai @ (x_opt - xi_star)).norm() ** 2
        zeta2_ += zeta_i2
    zeta2_ /= m

    scale = zeta / zeta2_.sqrt()
    x_opt = scale * x_opt
    xi_stars = [xi_star * scale for xi_star in xi_stars]
    b = scale * b

    # Adding sigma2 noise
    sigma_noises = []
    for i in range(m):
        xi = torch.randn(n)
        Ai = A[i * n: (i + 1) * n, :]
        _2Atb = 2 * Ai.T @ torch.diag(xi)

        # The variance over `n` and sum over `d`.
        b_sigma = _2Atb.var(axis=1).sum().sqrt()
        # Rescale the noise to the specific sigma
        sigma_noise = sigma / b_sigma * xi
        sigma_noises.append(sigma_noise)

    sigma_noises = torch.cat(sigma_noises)
    b += sigma_noises

    return (
        [A[i * n: (i + 1) * n, :] for i in range(m)],
        [b[i * n: (i + 1) * n] for i in range(m)],
        x_opt,
    )


class Quadratics(torch.utils.data.Dataset):
    def __init__(self, n_samples, n_features, L, mu, sigma=0, seed=0):
        self.n_samples = n_samples
        self.n_features = n_features
        self.sigma = sigma

        with fork_with_seed(seed=seed):
            self._A, self._b, self._x_opt = generate_synthetic_dataset(
                n=self.n_samples,
                d=self.n_features,
                L=L,
                mu=mu,
                sigma=sigma,
            )

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return self.A[idx, :], self.b[idx]

    @property
    def A(self):
        return self._A

    @property
    def b(self):
        return self._b

    @property
    def x_opt(self):
        return self._x_opt


@contextlib.contextmanager
def fork_with_seed(seed):
    if seed is None:
        yield
    else:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            yield


class QuadraticTask(object):
    def __init__(
        self,
        n_samples,
        n_features,
        batch_size=None,
        L=10,
        mu=1,
        r0=1,
        sigma=0,
        seed=0,
    ):
        self.r0 = r0
        self._model = self._initialize_model(n_features, r0)
        self._dataset = Quadratics(n_samples, n_features, L, mu, sigma, seed)
        self.n_samples = n_samples
        self.batch_size = batch_size or n_samples

    def _initialize_model(self, d, r0):
        model = LinearModel(d)
        model.layer.weight.data /= model.layer.weight.data.norm() / r0
        return model

    def loss_func(self):
        return torch.nn.MSELoss(reduction="mean")

    @property
    def model(self):
        return self._model

    def model_class(self):
        return LinearModel

    def metrics(self):
        return {}

    def train_loader(self):
        return torch.utils.data.DataLoader(
            self._dataset,
            batch_size=self.batch_size,
            shuffle=True,
        )

    def test_loader(self):
        raise NotImplementedError()


class DistributedQuadratics(torch.utils.data.Dataset):
    def __init__(self, A, b):
        self._A = A
        self._b = b
        self.n = A.shape[0]

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return self.A[idx, :], self.b[idx]

    @property
    def A(self):
        return self._A

    @property
    def b(self):
        return self._b


class DistributedQuadraticsTask(object):
    def __init__(self, A, b, batch_size, model):
        self.A = A
        self.b = b
        self.batch_size = batch_size
        self._model = model
        self.n_samples = A.shape[0]
        self._dataset = DistributedQuadratics(A, b)

    def loss_func(self):
        return torch.nn.MSELoss(reduction="mean")

    @property
    def model(self):
        return self._model

    def model_class(self):
        return LinearModel

    def metrics(self):
        return {}

    def train_loader(self):
        return torch.utils.data.DataLoader(
            self._dataset,
            batch_size=self.batch_size,
            shuffle=True,
        )

    def test_loader(self):
        return torch.utils.data.DataLoader(
            self._dataset,
            batch_size=self.batch_size,
            shuffle=True,
        )


def get_distributed_quadratics_tasks(m, n, d, b, L, mu, r0, sigma, zeta, seed):
    with fork_with_seed(seed=seed):
        As, bs, _ = generate_synthetic_distributed_dataset(
            m=m, n=n, d=d, L=L, mu=mu, sigma=sigma, zeta=zeta
        )
        model = LinearModel(d)
        model.layer.weight.data /= model.layer.weight.data.norm() / r0

    tasks = []
    for i in range(m):
        worker_task = DistributedQuadraticsTask(
            A=As[i],
            b=bs[i],
            batch_size=b,
            model=copy.deepcopy(model),
        )
        tasks.append(worker_task)

    main_task = DistributedQuadraticsTask(
        A=torch.cat(As), b=torch.cat(bs), batch_size=m * n, model=None
    )
    return tasks, main_task
