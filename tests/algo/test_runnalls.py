from gmr.algo import fit_runnalls
from gmr.algo import fit_gmrc


def test_runnalls(plot, _gm, helper):
    m_gm = fit_runnalls(_gm, L=2)

    if plot:
        helper.sampled_gm_plot(m_gm)


def test_gmrc(plot, _gm, helper):
    m_gm = fit_gmrc(_gm, L=2)

    if plot:
        helper.sampled_gm_plot(m_gm)
