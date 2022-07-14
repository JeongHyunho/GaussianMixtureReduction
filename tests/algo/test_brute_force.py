from gmr.algo import fit_brute_force


def test_brute_force(plot, _gm, helper):
    m_gm = fit_brute_force(_gm, L=2)

    if plot:
        helper.sampled_gm_plot(m_gm)
