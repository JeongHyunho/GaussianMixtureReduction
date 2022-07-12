import numpy as np
from matplotlib import pyplot as plt

from mixtures.gm import sample_gm, gm_prob
from algo.min_ise import fit_min_ise

gm = sample_gm(
        n=3,
        d=2,
        pi_alpha=5*np.ones(3),
        mu_rng=[0., 3.],
        var_df=5,
        var_scale=1/5 * np.eye(2),
    )

t0, t1 = np.meshgrid(np.linspace(-1., 4., 100), np.linspace(-1., 4., 100))
p = gm_prob(np.stack([t0, t1], axis=-1), gm)

print(gm)
plt.contourf(t0, t1, p)
plt.plot(gm.mu[:, 0], gm.mu[:, 1], 'x')
plt.show()


def test_min_ise():
    m_gm = fit_min_ise(gm, L=2)

    m_p = gm_prob(np.stack([t0, t1], axis=-1), m_gm)
    print(m_gm)
    plt.contourf(t0, t1, m_p)
    plt.plot(m_gm.mu[:, 0], m_gm.mu[:, 1], 'x')
    plt.show()
