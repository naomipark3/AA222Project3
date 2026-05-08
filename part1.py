r"""
project3_part1.py
Part 1, Tasks 1-2 of AA222 Project 3 (Probabilistic Optimization). Solves
the portfolio layout problem: place n stocks in the 2D (beta, HML) grid
to maximize the minimum pairwise L2 distance, subject to the four mandate
constraints from Table 1 of the handout (active-share inner boundary,
tracking-error outer boundary, high-beta growth exclusion, defensive
deep-value exclusion). Solves for n in [2, 10] and writes the (n, p*)
results to a header-less CSV for the Gradescope autograder.

approach used:
- Epigraph reformulation: introduce a slack variable t and maximize t
  subject to ||s_i - s_j||^2 >= t^2 for all i<j (the squared form keeps
  things smooth near zero) plus all four feasibility constraints. The
  resulting problem has 2n+1 variables and 4n + n(n-1)/2 inequality
  constraints, all quadratic.
- SLSQP with analytical Jacobians: handles inequalities natively (no
  AL outer loop, no penalty schedule). Jacobians are cheap closed-form
  expressions and they stabilize SLSQP near active sets where finite
  differences would otherwise get noisy.
- Multistart: a mix of (a) pure rejection-sampled random layouts and
  (b) structured "ring" layouts (n points evenly on a random-radius
  circle around (5, 5) with random rotation), since the optimal layouts
  for several n turn out to lie on or near the outer boundary.
- For each n we keep the best feasible result by the *recomputed* min
  pairwise distance from the final coordinates (not the solver's t,
  which can sit slightly below the true min if a constraint is inactive).
"""

import csv
import numpy as np
from scipy.optimize import minimize


#problem geometry (Table 1 of the handout)
BENCHMARK = np.array([5.0, 5.0])
R_INNER = 1.5
R_OUTER = 4.0

HIGHBETA_CENTER = np.array([7.5, 2.5])
HIGHBETA_R = 1.2

DEFVALUE_CENTER = np.array([2.5, 7.5])
DEFVALUE_R = 1.2

#solver knobs
N_RESTARTS = 80
SLSQP_OPTS = {"maxiter": 300, "ftol": 1e-10, "disp": False}
FEAS_TOL = 1e-6  #slack allowed when verifying solver output is feasible


#feasibility helpers

def is_feasible_point(x, y, tol=0.0):
    """
    Check the four mandate constraints at a single (x, y) point.
    @param x, y: scalar coordinates.
    @param tol: nonnegative relaxation in distance units. tol = 0 is a
        strict check; tol > 0 lets points sit up to `tol` inside an
        exclusion before being rejected (useful for accepting solver
        output that's feasible up to floating point).
    @return: bool, True iff (x, y) satisfies all four constraints.
    """
    dx, dy = x - BENCHMARK[0], y - BENCHMARK[1]
    d_bench = np.sqrt(dx*dx + dy*dy)
    if d_bench < R_INNER - tol: return False
    if d_bench > R_OUTER + tol: return False

    dx, dy = x - HIGHBETA_CENTER[0], y - HIGHBETA_CENTER[1]
    if np.sqrt(dx*dx + dy*dy) < HIGHBETA_R - tol: return False

    dx, dy = x - DEFVALUE_CENTER[0], y - DEFVALUE_CENTER[1]
    if np.sqrt(dx*dx + dy*dy) < DEFVALUE_R - tol: return False

    return True


def sample_feasible_point(rng):
    """
    Rejection-sample a single (x, y) inside the feasible annulus. Uses
    area-uniform polar sampling (r drawn from sqrt of a uniform on
    [R_INNER^2, R_OUTER^2]) so points spread evenly in 2D rather than
    bunching near the inner boundary.
    @return: (x, y) tuple of floats.
    """
    while True:
        r = np.sqrt(rng.uniform(R_INNER**2, R_OUTER**2))
        theta = rng.uniform(0.0, 2*np.pi)
        x = BENCHMARK[0] + r*np.cos(theta)
        y = BENCHMARK[1] + r*np.sin(theta)
        if is_feasible_point(x, y):
            return x, y


def random_initial_layout(n, rng):
    """n independently rejection-sampled feasible points, shape (n, 2)."""
    return np.array([sample_feasible_point(rng) for _ in range(n)])


def structured_initial_layout(n, rng):
    """
    n points placed evenly on a circle of random radius around (5, 5)
    with a random rotation offset. Any point that lands inside an
    exclusion zone is replaced by an independently sampled feasible
    point. Useful because the optimum at several n is roughly an
    n-gon inscribed in the outer boundary.
    """
    rho = rng.uniform(2.0, 3.9)
    offset = rng.uniform(0.0, 2*np.pi)
    pts = np.empty((n, 2))
    for k in range(n):
        theta = 2*np.pi*k/n + offset
        x = BENCHMARK[0] + rho*np.cos(theta)
        y = BENCHMARK[1] + rho*np.sin(theta)
        if is_feasible_point(x, y):
            pts[k] = (x, y)
        else:
            pts[k] = sample_feasible_point(rng)
    return pts

#objective and constraints

def objective(z):
    """Negative slack: SLSQP minimizes, we want max t."""
    return -z[-1]

def objective_jac(z):
    g = np.zeros_like(z)
    g[-1] = -1.0
    return g

def build_constraint_funcs(n):
    """
    Build vector-valued (fun, jac) for the full inequality constraint
    block c(z) >= 0. Order of constraints (length 4n + n(n-1)/2):
        [0   : n  ] outside inner boundary  (one per stock)
        [n   : 2n ] inside outer boundary   (one per stock)
        [2n  : 3n ] outside high-beta zone  (one per stock)
        [3n  : 4n ] outside defensive zone  (one per stock)
        [4n  : ...] pairwise distance^2 - t^2  (n(n-1)/2 entries)
    Vectorizing over a single SLSQP constraint dict avoids a large
    per-constraint Python overhead and keeps the Jacobian as one matrix.
    """
    n_pair = n*(n-1)//2
    pair_idx = [(i, j) for i in range(n) for j in range(i+1, n)]

    def fun(z):
        pts = z[:-1].reshape(n, 2)
        t = z[-1]
        c = np.empty(4*n + n_pair)
        d_bench = pts - BENCHMARK
        d_hb    = pts - HIGHBETA_CENTER
        d_dv    = pts - DEFVALUE_CENTER
        d2_bench = (d_bench*d_bench).sum(axis=1)
        c[0*n:1*n] = d2_bench - R_INNER**2
        c[1*n:2*n] = R_OUTER**2 - d2_bench
        c[2*n:3*n] = (d_hb*d_hb).sum(axis=1) - HIGHBETA_R**2
        c[3*n:4*n] = (d_dv*d_dv).sum(axis=1) - DEFVALUE_R**2
        for k, (i, j) in enumerate(pair_idx):
            d = pts[i] - pts[j]
            c[4*n + k] = d @ d - t*t
        return c

    def jac(z):
        pts = z[:-1].reshape(n, 2)
        t = z[-1]
        m = 4*n + n_pair
        J = np.zeros((m, 2*n + 1))
        #disk constraints (rows 0..4n-1)
        for i in range(n):
            J[0*n + i, 2*i:2*i+2] =  2*(pts[i] - BENCHMARK)
            J[1*n + i, 2*i:2*i+2] = -2*(pts[i] - BENCHMARK)
            J[2*n + i, 2*i:2*i+2] =  2*(pts[i] - HIGHBETA_CENTER)
            J[3*n + i, 2*i:2*i+2] =  2*(pts[i] - DEFVALUE_CENTER)
        #pairwise rows (rows 4n..)
        for k, (i, j) in enumerate(pair_idx):
            dx = pts[i, 0] - pts[j, 0]
            dy = pts[i, 1] - pts[j, 1]
            row = 4*n + k
            J[row, 2*i  ] =  2*dx
            J[row, 2*i+1] =  2*dy
            J[row, 2*j  ] = -2*dx
            J[row, 2*j+1] = -2*dy
            J[row, -1]    = -2*t
        return J

    return fun, jac


#single-n solver
def min_pairwise(pts):
    """Actual minimum pairwise L2 distance over a layout, for verification."""
    n = len(pts)
    best = np.inf
    for i in range(n):
        for j in range(i+1, n):
            d = np.linalg.norm(pts[i] - pts[j])
            if d < best:
                best = d
    return best


def all_feasible(pts, tol=FEAS_TOL):
    return all(is_feasible_point(pts[i, 0], pts[i, 1], tol=tol)
               for i in range(len(pts)))


def solve_n(n, rng, n_restarts=N_RESTARTS):
    """
    Multistart SLSQP for a fixed portfolio size n. Returns (best_pts,
    best_p) where best_p is the recomputed minimum pairwise distance
    from the best feasible final layout across all restarts.
    @param n: number of stocks (>= 2).
    @param rng: numpy Generator for reproducibility.
    @param n_restarts: number of independent SLSQP starts.
    """
    cfun, cjac = build_constraint_funcs(n)
    cons = [{"type": "ineq", "fun": cfun, "jac": cjac}]
    #x, y in [1, 9] (the bounding box of the outer disk); t in [0, 2 R]
    bounds = [(1.0, 9.0)] * (2*n) + [(0.0, 2*R_OUTER)]

    best_p = -np.inf
    best_pts = None

    for k in range(n_restarts):
        #every third start is structured (ring); the rest are pure random
        if k % 3 == 0:
            pts0 = structured_initial_layout(n, rng)
        else:
            pts0 = random_initial_layout(n, rng)
        #seed t just under the current min so the slack constraints
        #start out feasible
        t0 = 0.99 * min_pairwise(pts0)
        z0 = np.concatenate([pts0.flatten(), [t0]])

        res = minimize(
            objective, z0, jac=objective_jac, method="SLSQP",
            bounds=bounds, constraints=cons, options=SLSQP_OPTS,
        )
        if not np.isfinite(res.fun):
            continue

        pts = res.x[:-1].reshape(n, 2)
        if not all_feasible(pts):
            continue

        p = min_pairwise(pts)
        if p > best_p:
            best_p = p
            best_pts = pts

    return best_pts, best_p


#main

def main():
    rng = np.random.default_rng(seed=42)

    rows = []
    layouts = {}
    print("Solutions computed for each configuration (n)")
    for n in range(2, 11):
        pts, p = solve_n(n, rng, n_restarts=N_RESTARTS)
        print(f"n={n:2d}  p* = {p:.6f}")
        rows.append((n, p))
        layouts[str(n)] = pts

    #autograder CSV (no header, two columns: n, p*)
    with open("portfolio_layouts.csv", "w", newline="") as f:
        w = csv.writer(f)
        for n, p in rows:
            w.writerow([n, f"{p:.6f}"])

    #sidecar with the actual coordinates, useful for plotting tasks (3-5) later
    np.savez("portfolio_layouts.npz", **layouts)

    print("wrote portfolio_layouts.csv and portfolio_layouts.npz")


if __name__ == "__main__":
    main()