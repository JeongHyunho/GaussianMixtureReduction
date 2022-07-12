# Gaussian Mixture Reduction

Algorithms in ["A Look At Gaussian Mixture Reduction Algorithms"](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=5977695) for reducing gaussian mixtures are implemented.

I also added MIN-ISE which every mergin round selects the pair minimally increasing integrated squared error.

1) Runnalls algorithm
2) West algorithm
3) GMRC (without refinement)
4) COWA (constrained optimized weight optimization)
5) MIN-ISE (merge pairs that minimally increase ISE)
6) Brute-force (best among all possible merge sets)

Also, I tried to reproduce results in the paper and got similar one. (see [eval.py](./eval.py), it takes a couple of hours)

|          | ISE Mean  | ISE Std  |
|----------|-----------|----------|
| Runnalls | 0.08211   | 0.08062  |
| West     | 0.10452   | 0.09380  |
| GMRC     | 0.08208   | 0.08062  |
| COWA     | 0.10172   | 0.09124  |
| MIN-ISE  | 0.04605   | 0.02832  |
| Brute    | 0.04597   | 0.02827  |

But, I could not reproduce Fig.1 in the paper. (see [demo.py](./demo.py))

<img alt="Reducec mixtures" height="400" src="./images/demo.png" width="500"/>

Future works list:
- [ ] Component means and variance refinement via gradient descent
- [x] Brute-force initialization to show optimality gap
- [ ] Batch operation for reducing mixture
- [ ] Evaluating the product of mixtures reduction
