from gmr.algo.cowa import fit_cowa
from gmr.algo import fit_west


def test_west(plot, sampled_gm, helper):
    m_gm = fit_west(sampled_gm, L=2)

    if plot:
        helper.sampled_gm_plot(m_gm)


def test_cowa(plot, sampled_gm, helper):
    m_gm = fit_cowa(sampled_gm, L=2)

    if plot:
        helper.sampled_gm_plot(m_gm)
