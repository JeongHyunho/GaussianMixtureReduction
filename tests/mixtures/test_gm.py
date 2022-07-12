import numpy as np
from matplotlib import pyplot as plt

from mixtures.gm import sample_gm, gm_prob, calc_ise, merge_gm, kl_gm_comp

gm = sample_gm(
        n=3,
        d=2,
        pi_alpha=5*np.ones(3),
        mu_rng=[0., 3.],
        var_df=5,
        var_scale=1/5 * np.eye(2),
    )


def test_gm_prob():
    t0, t1 = np.meshgrid(np.linspace(-1., 4., 100), np.linspace(-1., 4., 100))
    p = gm_prob(np.stack([t0, t1], axis=-1), gm)

    print(gm)
    plt.contourf(t0, t1, p)
    plt.plot(gm.mu[:, 0], gm.mu[:, 1], 'x')
    plt.show()


def test_calc_ise():
    assert calc_ise(gm, gm) < 1e-9


def test_merge_gm():
    t0, t1 = np.meshgrid(np.linspace(-1., 4., 100), np.linspace(-1., 4., 100))
    p = gm_prob(np.stack([t0, t1], axis=-1), gm)

    print(gm)
    plt.contourf(t0, t1, p)
    plt.plot(gm.mu[:, 0], gm.mu[:, 1], 'x')
    plt.show()

    m_gm = merge_gm(gm, [[0, 1]])
    p = gm_prob(np.stack([t0, t1], axis=-1), m_gm)
    print(m_gm)
    plt.contourf(t0, t1, p)
    plt.plot(m_gm.mu[:, 0], m_gm.mu[:, 1], 'x')
    plt.show()


def test_kl_gm_comp():
    assert np.all(np.diag(kl_gm_comp(gm, gm)) < 1e-9)
