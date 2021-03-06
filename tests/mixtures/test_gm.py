import pytest
import torch
from matplotlib import pyplot as plt
from copy import deepcopy

from gmr.mixtures import calc_ise, kl_gm_comp, GM


def test_gm_mul(plot):
    gm0 = GM(pi=[0.3, 0.7], mu=torch.tensor([1., 2.]).view(2, 1), var=torch.tensor([0.2, 0.1]).view(2, 1, 1))
    gm1 = GM(pi=[0.5, 0.5], mu=torch.tensor([1.2, 2.7]).view(2, 1), var=torch.tensor([0.1, 0.3]).view(2, 1, 1))
    gm_prod = gm0 * gm1

    if plot:
        t = torch.linspace(-0.5, 4.5, steps=1000)[..., None]
        p0 = gm0.prob(t.to(gm_prod.mu)).cpu()
        p1 = gm1.prob(t.to(gm_prod.mu)).cpu()
        p_prod = gm_prod.prob(t.to(gm_prod.mu)).cpu()

        plt.plot(t, p0)
        plt.plot(t, p1)
        plt.plot(t, p_prod)
        plt.legend(['gm0', 'gm1', 'product'])
        plt.show()


def test_calc_ise(_gm):
    assert calc_ise(_gm, _gm) < 1e-9


@pytest.mark.parametrize('reduce', [True, False])
def test_gm_prob(plot, _gm, reduce):
    x, y = torch.meshgrid(torch.linspace(-1., 4., 100), torch.linspace(-1., 4., 100))
    xy = torch.stack([x, y], dim=-1).to(_gm.mu)
    p = _gm.prob(xy, reduce=reduce)


def test_merge_gm(plot, _gm, helper):
    cp_gm = deepcopy(_gm)
    cp_gm.merge([[0, 1]])

    if plot:
        helper.sampled_gm_plot(cp_gm)


def test_kl_gm_comp(_gm):
    assert torch.all(torch.diag(kl_gm_comp(_gm, _gm)) < 1e-6)
