import numpy as np

from cowa import fit_cowa
from gm import sample_gm, calc_ise
from gmrc import fit_gmrc
from runnalls import fit_runnalls
from west import fit_west


def main():

    num_eval = 1000
    ise = {'runnalls': [], 'west': [], 'gmrc': [], 'cowa': []}

    for seed in range(num_eval):
        gm = sample_gm(
            n=10,
            pi_alpha=np.ones(10),
            mu_rng=[0., 3.],
            var_df=3,
            var_scale=1. / 50,
            seed=seed,
        )

        runnalls_gm = fit_runnalls(gm, L=5)
        west_gm = fit_west(gm, L=5)
        gmrc_gm = fit_gmrc(gm, L=5)
        cowa_gm = fit_cowa(gm, L=5)

        ise['runnalls'].append(calc_ise(gm, runnalls_gm))
        ise['west'].append(calc_ise(gm, west_gm))
        ise['gmrc'].append(calc_ise(gm, gmrc_gm))
        ise['cowa'].append(calc_ise(gm, cowa_gm))

    print(f"ISE Runnalls: {np.mean(ise['runnalls']):.5f} ± {np.std(ise['runnalls']):.5f}")
    print(f"ISE West: {np.mean(ise['west']):.5f} ± {np.std(ise['west']):.5f}")
    print(f"ISE GMRC: {np.mean(ise['gmrc']):.5f} ± {np.std(ise['gmrc']):.5f}")
    print(f"ISE COWA: {np.mean(ise['cowa']):.5f} ± {np.std(ise['cowa']):.5f}")


if __name__ == '__main__':
    main()
