import sys

import numpy as np
import multiprocessing as mp

from brute_force import fit_brute_force
from cowa import fit_cowa
from gm import sample_gm, calc_ise
from gmrc import fit_gmrc
from min_ise import fit_min_ise
from runnalls import fit_runnalls
from west import fit_west


N_PROCESSES = 8
INC_BRUTE = False


def sample_eval(seed):
    gm = sample_gm(
        n=10,
        d=4,
        pi_alpha=np.ones(10),
        mu_rng=[0., 3.],
        var_df=5,
        var_scale=1./50*np.eye(4),
        seed=seed,
    )

    runnalls_gm = fit_runnalls(gm, L=5)
    west_gm = fit_west(gm, L=5)
    gmrc_gm = fit_gmrc(gm, L=5)
    cowa_gm = fit_cowa(gm, L=5)
    min_ise_gm = fit_min_ise(gm, L=5)
    brute_gm = fit_brute_force(gm, L=5) if INC_BRUTE else gm

    ise_runnalls = calc_ise(gm, runnalls_gm)
    ise_west = calc_ise(gm, west_gm)
    ise_gmrc = calc_ise(gm, gmrc_gm)
    ise_cowa = calc_ise(gm, cowa_gm)
    ise_min_ise = calc_ise(gm, min_ise_gm)
    ise_brute = calc_ise(gm, brute_gm)

    return ise_runnalls, ise_west, ise_gmrc, ise_cowa, ise_min_ise, ise_brute


def main():
    num_eval = 1000
    result = []

    pool = mp.Pool(processes=N_PROCESSES)
    for i, r in enumerate(pool.imap_unordered(sample_eval, range(num_eval))):
        result.append(r)
        sys.stdout.write(f'\r{i/num_eval*100:.2f} % ...')
        sys.stdout.flush()
    print()
    pool.close()
    pool.join()

    re_array = np.array(result)
    print(f"ISE Runnalls: {np.mean(re_array[:, 0]):.5f} ± {np.std(re_array[:, 0]):.5f}")
    print(f"ISE West: {np.mean(re_array[:, 1]):.5f} ± {np.std(re_array[:, 1]):.5f}")
    print(f"ISE GMRC: {np.mean(re_array[:, 2]):.5f} ± {np.std(re_array[:, 2]):.5f}")
    print(f"ISE COWA: {np.mean(re_array[:, 3]):.5f} ± {np.std(re_array[:, 3]):.5f}")
    print(f"ISE MIN-ISE: {np.mean(re_array[:, 4]):.5f} ± {np.std(re_array[:, 4]):.5f}")

    if INC_BRUTE:
        print(f"ISE BRUTE: {np.mean(re_array[:, 5]):.5f} ± {np.std(re_array[:, 5]):.5f}")


if __name__ == '__main__':
    main()
