"""
benchmark.py -- compare scipy.optimize.newton against improved_root.reliable_root.

For every test equation and every initial guess x0 on a grid, both solvers run
and each answer is checked against the *known* true roots. Results are written
to results/benchmark_results.csv and graphs to figures/.

Outcome categories
------------------
SciPy newton:
    valid             converged=True and the answer is a true root
    false convergence converged=True but the answer is NOT a root   <- issue #6379
    reported failure  converged=False (SciPy admits failure)

reliable_root:
    valid             status 'reliable' and the answer is a true root
    recovered         status 'recovered' (Brent fallback) and a true root
    honest failure    no root claimed (status 'no_root_found' / 'unreliable')
    false claim       claimed reliable but the answer is NOT a root

Run:  python benchmark.py
"""

from __future__ import annotations

import csv
import time
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import newton

from improved_root import RECOVERED, RELIABLE, reliable_root

HERE = Path(__file__).parent
FIG_DIR = HERE / "figures"
RES_DIR = HERE / "results"

FTOL = 1e-6
X0_GRID = np.linspace(-5, 5, 101)

# name, f, f', true real roots (empty = no real root), difficulty note
PROBLEMS = [
    ("x^2 - 2", lambda x: x**2 - 2, lambda x: 2 * x,
     [-np.sqrt(2), np.sqrt(2)], "easy"),
    ("cos(x) - x", lambda x: np.cos(x) - x, lambda x: -np.sin(x) - 1,
     [0.7390851332151607], "easy"),
    ("(x - 1)^2", lambda x: (x - 1) ** 2, lambda x: 2 * (x - 1),
     [1.0], "double root: slow, no sign change"),
    ("x^3 - 2x + 2", lambda x: x**3 - 2 * x + 2, lambda x: 3 * x**2 - 2,
     [-1.7692923542386314], "Newton cycles 0 <-> 1"),
    ("arctan(x)", np.arctan, lambda x: 1 / (1 + x**2),
     [0.0], "Newton diverges for |x0| > 1.39"),
    ("x exp(-x)", lambda x: x * np.exp(-x), lambda x: (1 - x) * np.exp(-x),
     [0.0], "flat tail for x > 1"),
    ("cbrt(x)", np.cbrt, lambda x: (1 / 3) * np.abs(x) ** (-2 / 3) if x != 0 else np.inf,
     [0.0], "Newton diverges from any x0 != 0"),
    ("x^4 - x^2 + 1  (#6379)", lambda x: x**4 - x**2 + 1, lambda x: 4 * x**3 - 2 * x,
     [], "no real roots"),
]

# Colors (validated reference palette: status steps + categorical slots 1-2)
C_VALID = "#0ca30c"
C_RECOVERED = "#2a78d6"
C_FAILURE = "#a8a7a0"
C_FALSE = "#d03b3b"
C_SCIPY = "#2a78d6"
C_OURS = "#eb6834"
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e4e3df"
SURFACE = "#fcfcfb"


def is_true_root(x, roots, tol=1e-5):
    if x is None or not np.isfinite(x):
        return False
    return any(abs(x - r) <= tol * max(1.0, abs(r)) for r in roots)


class Counter:
    def __init__(self, f):
        self.f, self.n = f, 0

    def __call__(self, x):
        self.n += 1
        return self.f(x)


def run_scipy(f, fp, x0):
    fc = Counter(f)
    t = time.perf_counter()
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore")
        try:
            x, info = newton(fc, x0, fprime=fp, full_output=True, disp=False)
            conv, iters = bool(info.converged), info.iterations
        except (RuntimeError, OverflowError, ZeroDivisionError, ValueError):
            x, conv, iters = np.nan, False, 0
    return float(x), conv, iters, fc.n, time.perf_counter() - t


def run_ours(f, fp, x0):
    t = time.perf_counter()
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore")
        rep = reliable_root(f, x0, fprime=fp, ftol=FTOL)
    return rep, time.perf_counter() - t


def classify_scipy(x, conv, f, roots):
    if not conv:
        return "reported failure"
    with np.errstate(all="ignore"):
        ok = np.isfinite(x) and abs(f(x)) <= FTOL and is_true_root(x, roots)
    return "valid" if ok else "false convergence"


def classify_ours(rep, roots):
    if rep.reliable:
        if not is_true_root(rep.root, roots):
            return "false claim"
        return "recovered" if rep.status == RECOVERED else "valid"
    return "honest failure"


def run_benchmark():
    rows = []
    for mode in ("secant", "newton"):
        for name, f, fp, roots, note in PROBLEMS:
            deriv = fp if mode == "newton" else None
            for x0 in X0_GRID:
                x0 = float(x0)
                xs, conv, it, nf, ts = run_scipy(f, deriv, x0)
                rep, to = run_ours(f, deriv, x0)
                rows.append(dict(
                    mode=mode, problem=name, note=note, x0=x0,
                    scipy_root=xs, scipy_converged=conv, scipy_iterations=it,
                    scipy_nfev=nf, scipy_time_s=ts,
                    scipy_outcome=classify_scipy(xs, conv, f, roots),
                    ours_root=rep.root, ours_status=rep.status,
                    ours_nfev=rep.nfev, ours_time_s=to,
                    ours_outcome=classify_ours(rep, roots),
                ))
    return rows


def write_csv(rows):
    RES_DIR.mkdir(exist_ok=True)
    with open(RES_DIR / "benchmark_results.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def summarize(rows):
    """Per (mode, problem) percentages and mean costs."""
    out = []
    for mode in ("secant", "newton"):
        for name, *_ in PROBLEMS:
            sub = [r for r in rows if r["mode"] == mode and r["problem"] == name]
            n = len(sub)
            pct = lambda key, val: 100 * sum(r[key] == val for r in sub) / n
            out.append(dict(
                mode=mode, problem=name, n=n,
                scipy_valid=pct("scipy_outcome", "valid"),
                scipy_false=pct("scipy_outcome", "false convergence"),
                scipy_fail=pct("scipy_outcome", "reported failure"),
                ours_valid=pct("ours_outcome", "valid"),
                ours_recovered=pct("ours_outcome", "recovered"),
                ours_honest=pct("ours_outcome", "honest failure"),
                ours_false=pct("ours_outcome", "false claim"),
                scipy_nfev=np.mean([r["scipy_nfev"] for r in sub]),
                ours_nfev=np.mean([r["ours_nfev"] for r in sub]),
                scipy_ms=1e3 * np.mean([r["scipy_time_s"] for r in sub]),
                ours_ms=1e3 * np.mean([r["ours_time_s"] for r in sub]),
            ))
    with open(RES_DIR / "summary.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0].keys()))
        w.writeheader()
        for s in out:
            w.writerow({k: (round(v, 3) if isinstance(v, float) else v) for k, v in s.items()})
    return out


# ---------------------------------------------------------------- plotting
def _style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=9, length=0)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def plot_outcomes(summary, mode):
    sub = [s for s in summary if s["mode"] == mode]
    names = [s["problem"] for s in sub][::-1]
    sub = sub[::-1]
    y = np.arange(len(sub))
    h = 0.36
    fig, ax = plt.subplots(figsize=(10, 6.2), facecolor=SURFACE)
    _style(ax)

    def stack(offset, parts):
        left = np.zeros(len(sub))
        for vals, color, label in parts:
            vals = np.array(vals)
            ax.barh(y + offset, vals, h, left=left, color=color, label=label,
                    edgecolor=SURFACE, linewidth=1.5)
            left += vals

    stack(+h / 2 + 0.02, [
        ([s["scipy_valid"] for s in sub], C_VALID, "Valid root"),
        ([s["scipy_false"] for s in sub], C_FALSE, "False convergence (claims success, not a root)"),
        ([s["scipy_fail"] for s in sub], C_FAILURE, "Reported failure / no root claimed"),
    ])
    stack(-h / 2 - 0.02, [
        ([s["ours_valid"] for s in sub], C_VALID, None),
        ([s["ours_recovered"] for s in sub], C_RECOVERED, "Recovered by Brent fallback"),
        ([s["ours_false"] for s in sub], C_FALSE, None),
        ([s["ours_honest"] for s in sub], C_FAILURE, None),
    ])
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=10, color=INK)
    for yi in y:
        ax.text(-1, yi + h / 2 + 0.02, "SciPy", ha="right", va="center", fontsize=8, color=INK2)
        ax.text(-1, yi - h / 2 - 0.02, "Ours", ha="right", va="center", fontsize=8, color=INK2)
    ax.tick_params(axis="y", pad=40)
    ax.set_xlim(0, 100)
    ax.set_xlabel(f"% of {len(X0_GRID)} initial guesses in [-5, 5]", color=INK2)
    label = "secant (no derivative, the issue's setting)" if mode == "secant" else "Newton (with f')"
    ax.set_title(f"Outcomes by test equation: {label}", loc="left", fontsize=13,
                 color=INK, pad=52)
    handles, labels = ax.get_legend_handles_labels()
    order = [0, 3, 1, 2]
    ax.legend([handles[i] for i in order], [labels[i] for i in order], ncol=2,
              frameon=False, fontsize=9, loc="lower left", bbox_to_anchor=(0, 1.0),
              labelcolor=INK)
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"outcomes_{mode}.png", dpi=160, facecolor=SURFACE)
    plt.close(fig)


def plot_totals(summary):
    """Headline: false convergences and valid roots, summed over everything."""
    n_total = sum(s["n"] for s in summary)
    tot = lambda k: sum(s[k] * s["n"] / 100 for s in summary)
    cats = ["Valid root", "False convergence", "No root claimed"]
    scipy_v = [tot("scipy_valid"), tot("scipy_false"), tot("scipy_fail")]
    ours_v = [tot("ours_valid") + tot("ours_recovered"), tot("ours_false"), tot("ours_honest")]
    x = np.arange(len(cats))
    w = 0.36
    fig, ax = plt.subplots(figsize=(8, 4.6), facecolor=SURFACE)
    _style(ax)
    ax.grid(axis="x", visible=False)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    b1 = ax.bar(x - w / 2 - 0.01, scipy_v, w, color=C_SCIPY, label="scipy.optimize.newton")
    b2 = ax.bar(x + w / 2 + 0.01, ours_v, w, color=C_OURS, label="reliable_root (ours)")
    for bars in (b1, b2):
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + n_total * 0.01,
                    f"{int(round(b.get_height()))}", ha="center", fontsize=10, color=INK)
    ax.set_xticks(x)
    ax.set_xticklabels(cats, fontsize=11, color=INK)
    ax.set_ylabel("runs", color=INK2)
    ax.set_title(f"All {n_total} runs (8 equations x 101 guesses x 2 modes)",
                 loc="left", fontsize=13, color=INK)
    ax.legend(frameon=False, fontsize=10, labelcolor=INK)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "totals.png", dpi=160, facecolor=SURFACE)
    plt.close(fig)


def plot_cost(summary):
    sub = [s for s in summary if s["mode"] == "secant"][::-1]
    y = np.arange(len(sub))
    h = 0.36
    fig, ax = plt.subplots(figsize=(9, 5.2), facecolor=SURFACE)
    _style(ax)
    ax.barh(y + h / 2 + 0.02, [s["scipy_nfev"] for s in sub], h, color=C_SCIPY,
            label="scipy.optimize.newton")
    ax.barh(y - h / 2 - 0.02, [s["ours_nfev"] for s in sub], h, color=C_OURS,
            label="reliable_root (ours)")
    ax.set_yticks(y)
    ax.set_yticklabels([s["problem"] for s in sub], fontsize=10, color=INK)
    ax.set_xlabel("mean function evaluations per solve (secant mode)", color=INK2)
    ax.set_title("Cost: verification is cheap; recovery costs a bracket search",
                 loc="left", fontsize=13, color=INK, pad=24)
    ax.legend(frameon=False, fontsize=9, ncol=2, loc="lower left",
              bbox_to_anchor=(0, 1.0), labelcolor=INK)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "cost_nfev.png", dpi=160, facecolor=SURFACE)
    plt.close(fig)


def plot_issue():
    """The function from issue #6379 and where SciPy 'converged'."""
    f = lambda x: x**4 - x**2 + 1
    xs = np.linspace(-1.4, 1.4, 400)
    x_sp, info = newton(f, 0.001, full_output=True, disp=False)
    fig, ax = plt.subplots(figsize=(8, 4.4), facecolor=SURFACE)
    _style(ax)
    ax.grid(axis="x", visible=False)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.axhline(0, color=INK2, linewidth=1)
    ax.plot(xs, f(xs), color=C_SCIPY, linewidth=2)
    ax.plot([x_sp], [f(x_sp)], "o", markersize=10, color=C_FALSE,
            markeredgecolor=SURFACE, markeredgewidth=2, zorder=5)
    ax.annotate(f"SciPy 'root': x = {x_sp:.6f}\nf(x) = {f(x_sp):.6f}\nconverged = {info.converged}",
                xy=(x_sp, f(x_sp)), xytext=(0.25, 1.9), fontsize=10, color=INK,
                arrowprops=dict(arrowstyle="->", color=INK2))
    ax.text(-1.35, 0.12, "f(x) = 0  (never reached: min f = 0.75)", fontsize=9, color=INK2)
    ax.set_ylim(-0.2, 3)
    ax.set_xlabel("x", color=INK2)
    ax.set_title("Issue #6379: f(x) = x^4 - x^2 + 1 has no real root",
                 loc="left", fontsize=13, color=INK)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "issue_6379.png", dpi=160, facecolor=SURFACE)
    plt.close(fig)


def print_table(summary):
    hdr = (f"{'mode':7} {'problem':24} {'SciPy valid':>11} {'SciPy FALSE':>11} "
           f"{'ours valid':>10} {'ours FALSE':>10} {'nfev SciPy':>10} {'nfev ours':>9}")
    print(hdr)
    print("-" * len(hdr))
    for s in summary:
        print(f"{s['mode']:7} {s['problem']:24} {s['scipy_valid']:10.1f}% {s['scipy_false']:10.1f}% "
              f"{s['ours_valid'] + s['ours_recovered']:9.1f}% {s['ours_false']:9.1f}% "
              f"{s['scipy_nfev']:10.1f} {s['ours_nfev']:9.1f}")


def main():
    FIG_DIR.mkdir(exist_ok=True)
    rows = run_benchmark()
    write_csv(rows)
    summary = summarize(rows)
    print_table(summary)
    plot_outcomes(summary, "secant")
    plot_outcomes(summary, "newton")
    plot_totals(summary)
    plot_cost(summary)
    plot_issue()
    print(f"\nWrote {RES_DIR/'benchmark_results.csv'}, {RES_DIR/'summary.csv'} and figures/*.png")


if __name__ == "__main__":
    main()
