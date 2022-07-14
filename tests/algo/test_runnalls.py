from gmr.algo import fit_runnalls
from gmr.algo import fit_gmrc


def test_runnalls(plot, sampled_gm, helper):
    m_gm = fit_runnalls(sampled_gm, L=2)

    if plot:
        helper.sampled_gm_plot(m_gm)


def test_gmrc(plot, sampled_gm, helper):
    m_gm = fit_gmrc(sampled_gm, L=2)

    if plot:
        helper.sampled_gm_plot(m_gm)
