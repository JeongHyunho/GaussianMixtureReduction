from algo.brute_force import fit_brute_force


def test_brute_force(plot, sampled_gm, helper):
    m_gm = fit_brute_force(sampled_gm, L=2)

    if plot:
        helper.sampled_gm_plot(m_gm)
