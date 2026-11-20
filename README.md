# Improving SciPy Root-Finding Reliability

CSCI 3656 (Numerical Computation) group project.

**Problem.** [SciPy issue #6379](https://github.com/scipy/scipy/issues/6379):
`scipy.optimize.newton` can return `converged=True` at a point that is **not a root**.
It stops when the *step* `|x_{k+1} - x_k|` is small and never checks the *residual* `|f(x)|`.

```text
$ python original_problem.py
SciPy version: 1.18.1
Returned root: 0.0009999814755135925
Function value: 0.9999990000380485      <- f(x) is ~1, not 0
Converged: True                         <- but SciPy says it converged
```

`f(x) = x^4 - x^2 + 1 = (x^2 - 1/2)^2 + 3/4 >= 3/4` has **no real roots**, so the right answer is
"no root found".

**Our solution.** `reliable_root()` runs SciPy's solver unchanged, **verifies** the answer, and if
verification fails, **falls back to Brent's method** on a sign-change bracket. It never invents a root.

## Results (1616 runs: 8 equations x 101 initial guesses x secant/Newton)

| | `scipy.optimize.newton` | `reliable_root` (ours) |
|---|---:|---:|
| Valid root found | 954 | **1214** |
| False convergence (claims success at a non-root) | 6 | **0** |
| No root claimed | 656 | 402 |
| Extra cost when SciPy was already right | – | ~2-3 function evals |

![totals](figures/totals.png)

We also found a second kind of false convergence: for `x*exp(-x)` from `x0 = 1`, the secant step
overshoots to `x ≈ 10002`. There `exp(-x)` underflows, so `f(x) == 0.0` exactly and SciPy reports
success. A residual check alone can't catch this. Our underflow guard does.

## Repository layout

| File | Owner | What it is |
|---|---|---|
| `original_problem.py` | everyone | Reproduces issue #6379 |
| `improved_root.py` | Person 1, Algorithm | `reliable_root()`: verify, bracket, Brent fallback, report |
| `benchmark.py` | Person 2, Testing | Runs both solvers on all test cases; writes `results/*.csv` and `figures/*.png` |
| `demo.ipynb` | Person 3, Integration | Walkthrough: reproduce, explain, fix, benchmark, tutorial |

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python original_problem.py   # reproduce the bug
python improved_root.py      # our tool on the bug + a recovery example
python benchmark.py          # regenerate results/ and figures/
jupyter notebook demo.ipynb  # full walkthrough
```

## Tutorial

```python
from improved_root import reliable_root

rep = reliable_root(f, x0)                          # secant, like newton() without fprime
rep = reliable_root(f, x0, fprime=df)               # Newton
rep = reliable_root(f, x0, bracket=(a, b))          # guaranteed fallback if f(a)*f(b) < 0
rep = reliable_root(f, x0, ftol=1e-10, xtol=1e-10)  # stricter verification

print(rep)            # human-readable report
if rep.reliable:
    x = rep.root
else:
    print(rep.message)
```

`rep.status` is one of:

| status | meaning |
|---|---|
| `reliable` | SciPy's answer passed verification |
| `recovered` | SciPy's answer failed; Brent's method found a verified root |
| `no_root_found` | SciPy's answer failed and no sign-change bracket exists near `x0` |
| `unreliable` | Brent converged to a sign change that fails verification (a pole, or a very steep root) |

Other fields: `scipy_root`, `scipy_converged`, `false_convergence`, `residual`, `step_estimate`,
`bracket`, `nfev`, `scipy_nfev`, `notes`.

## How it works

1. **Run SciPy:** `newton(f, x0, fprime, full_output=True)`, unchanged.
2. **Verify** the candidate `x`:
   - residual: `|f(x)| <= ftol` (default `1e-6`)
   - Newton-step error estimate: `|f(x)/f'(x)| <= xtol * max(1, |x|)`. This estimates the distance
     to a simple root. `f'` comes from a central difference when not supplied.
   - underflow guard: an exact `f(x) == 0` is rejected if `f(x ± h)` are also exactly 0.
3. **Bracket:** use the user's `bracket` if `f(a)*f(b) < 0`, otherwise search outward from `x0`
   (`x0 ± step·1.6^k`) for a sign change.
4. **Recover:** `scipy.optimize.brentq` on the bracket. It is guaranteed to converge for continuous
   `f`. Its answer is verified too, so poles get rejected.

## Limitations

- `ftol` depends on the problem. `1e-6` is a demonstration default, not a universal guarantee. For
  `cbrt(x)`, `|f| = 1e-5` even at `x = 1e-15`, so the strict residual test rejects a real root and the
  tool reports `unreliable` (with an explanation). Loosening `ftol` recovers it.
- Even-multiplicity roots (e.g. `(x-1)^2`) have no sign change, so Brent's method can't find them.
- The bracket search is local and can miss distant roots. Pass `bracket=` when you know one.
- Proving there is *no* root nearby is the most expensive case (~150 function evals vs. 50 for SciPy).

## References

- SciPy issue #6379: https://github.com/scipy/scipy/issues/6379
- `scipy.optimize.newton` docs: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.newton.html
- `scipy.optimize.brentq` docs: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.brentq.html
- R. P. Brent, *Algorithms for Minimization without Derivatives*, 1973, ch. 4.
