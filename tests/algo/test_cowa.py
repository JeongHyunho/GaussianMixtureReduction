from gmr.algo.cowa import fit_cowa
from gmr.algo import fit_west


def test_west(plot, _gm, helper):
    m_gm = fit_west(_gm, L=2)

    if plot:
        helper.sampled_gm_plot(m_gm)


def test_cowa(plot, _gm, helper):
    m_gm = fit_cowa(_gm, L=2)

    if plot:
        helper.sampled_gm_plot(m_gm)
