import numpy as np
from matplotlib import pyplot as plt

from mixtures.gm import gm_prob, GM
from algo.min_ise import fit_min_ise

gm = GM.sample_gm(
        n=3,
        d=2,
        pi_alpha=5*np.ones(3),
        mu_rng=[0., 3.],
        var_df=5,
        var_scale=1/5 * np.eye(2),
    )


def test_min_ise():
    m_gm = fit_min_ise(gm, L=2)
