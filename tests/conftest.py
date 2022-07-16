import pytest
import torch

from matplotlib import pyplot as plt

from gmr import mixtures
from gmr.mixtures import GM
from gmr.mixtures import BatchGM


def pytest_addoption(parser):
    parser.addoption("--plot", action="store_true", default=False, help="test with plot")
    parser.addoption("--gpu", action="store_true", default=False, help="test with plot")


@pytest.fixture(scope='session')
def _gm(use_gpu):
    if use_gpu:
        mixtures.options.device = 'cuda'

    gm = GM.sample_gm(
        n=3,
        d=2,
        pi_alpha=5 * torch.ones(3),
        mu_rng=[0., 3.],
        var_df=5,
        var_scale=1 / 5 * torch.eye(2),
    )

    return gm


@pytest.fixture(scope='session')
def _batch_gm(use_gpu):
    mixtures.options.device = 'cuda'

    batch_gm = BatchGM.sample_batch_gm(
        n=3,
        b=[5],
        d=2,
        pi_alpha=5 * torch.ones(3),
        mu_rng=[0., 3.],
        var_df=5,
        var_scale=1 / 5 * torch.eye(2),
    )

    return batch_gm


@pytest.fixture(scope='session')
def _dbatch_gm(use_gpu):
    if use_gpu:
        mixtures.options.device = 'cuda'

    double_batch_gm = BatchGM.sample_batch_gm(
        n=3,
        b=[2, 5],
        d=2,
        pi_alpha=5 * torch.ones(3),
        mu_rng=[0., 3.],
        var_df=5,
        var_scale=1 / 5 * torch.eye(2),
    )

    return double_batch_gm


@pytest.fixture(scope="session")
def plot(pytestconfig):
    return pytestconfig.getoption("plot")


@pytest.fixture(scope="session")
def use_gpu(pytestconfig):
    return pytestconfig.getoption("gpu")


x, y = torch.meshgrid(torch.linspace(-1., 4., 100), torch.linspace(-1., 4., 100))
xy = torch.stack([x, y], dim=-1)


class Helper:

    @staticmethod
    def sampled_gm_plot(gm: GM):
        print(gm)
        m_p = gm.prob(xy.to(gm.mu)).cpu()
        plt.contourf(x, y, m_p)
        plt.plot(gm.mu[:, 0].cpu(), gm.mu[:, 1].cpu(), 'x')
        plt.show()


@pytest.fixture(scope='session')
def helper():
    return Helper
