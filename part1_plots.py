r"""
project3_part1_plots.py
Part 1, Tasks 3-5 of AA222 Project 3. Loads the optimized layouts and
the (n, p*) CSV produced by project3_part1.py and generates
scatter plots of the optimal layouts for n in {3, 5, 7}
with the four mandate boundaries drawn so feasibility is visible (task 3),
the minimum pairwise distance p* vs portfolio size n (task 4), and
the theoretical diversification efficiency eta = 1/(1 + 1/p*)
vs minimum separation distance p* (task 5)
NOTE: this script does NOT plot the Gaussian Process. GP fitting and
plotting belongs to Part 2 
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

#problem geometry (kept in sync with project3_part1.py)
BENCHMARK = (5.0, 5.0)
R_INNER = 1.5
R_OUTER = 4.0
HIGHBETA_CENTER = (7.5, 2.5)
HIGHBETA_R = 1.2
DEFVALUE_CENTER = (2.5, 7.5)
DEFVALUE_R = 1.2

#feasible region drawing
def draw_feasible_region(ax):
    """
    Draw the four mandate boundaries from Table 1 of the handout on
    ax. Color choices roughly mirror Figure 1 of the handout: green
    outer disk, red inner disk, orange high-beta zone, purple
    defensive deep-value zone.
    @param ax: matplotlib Axes to draw onto.
    @return: None (mutates `ax`).
    """
    outer = Circle(BENCHMARK, R_OUTER, facecolor="green", alpha=0.16,
                   edgecolor="darkgreen", linewidth=1.4,
                   label=f"Outer boundary (R = {R_OUTER})")
    inner = Circle(BENCHMARK, R_INNER, facecolor="red", alpha=0.32,
                   edgecolor="darkred", linewidth=1.4,
                   label=f"Inner boundary (r = {R_INNER})")
    hb = Circle(HIGHBETA_CENTER, HIGHBETA_R, facecolor="orange", alpha=0.42,
                edgecolor="darkorange", linewidth=1.4,
                label=r"High-$\beta$ growth zone")
    dv = Circle(DEFVALUE_CENTER, DEFVALUE_R, facecolor="purple", alpha=0.28,
                edgecolor="indigo", linewidth=1.4,
                label="Defensive deep-value zone")
    for patch in (outer, inner, hb, dv):
        ax.add_patch(patch)


#task 3: per-n layout scatter
def plot_layout(pts, n, p_star, filename):
    """
    One scatter plot of the optimal layout for portfolio size n with
    all four constraint regions drawn. Saves to {filename}.
    @param pts: (n, 2) array of optimized stock coordinates.
    @param n: portfolio size (used for titling).
    @param p_star: minimum pairwise distance achieved (for the title).
    @param filename: output path for the figure.
    """
    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    draw_feasible_region(ax)
    ax.scatter(pts[:, 0], pts[:, 1], s=80, color="black", zorder=5,
               edgecolor="white", linewidth=1.2,
               label=f"n = {n} stocks")
    #label each stock with its index for traceability against the npz
    for i, (x, y) in enumerate(pts):
        ax.annotate(str(i + 1), (x, y), xytext=(6, 6),
                    textcoords="offset points", fontsize=9, color="black")
    ax.set_xlim(0.5, 9.5)
    ax.set_ylim(0.5, 9.5)
    ax.set_aspect("equal")
    ax.set_xlabel("Market Beta (normalized)")
    ax.set_ylabel("Value HML (normalized)")
    ax.set_title(f"Optimal portfolio layout for n = {n}, $p^*$ = {p_star:.4f}")
    ax.legend(
    loc="upper left",
    bbox_to_anchor=(1.02, 1),
    fontsize=8,
    framealpha=0.9,
    borderaxespad=0.
    )
    ax.grid(True, alpha=0.3)
    fig.savefig(filename, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close(fig)


#task 4: p* vs n
def plot_p_vs_n(n_values, p_values, filename):
    """
    Line plot of the optimized minimum pairwise distance against
    portfolio size, for n in [2, 10].
    @param n_values: 1D int array of portfolio sizes.
    @param p_values: 1D float array of corresponding p* values.
    @param filename: output path.
    """
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.plot(n_values, p_values, marker="o", color="navy", linewidth=1.6)
    ax.set_xlabel("Number of stocks (n)")
    ax.set_ylabel("Min. pairwise distance (p*)")
    ax.set_title("Optimal min. pairwise distance vs. portfolio size")
    ax.set_xticks(n_values)
    ax.grid(True, alpha=0.3)
    fig.savefig(filename, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close(fig)

#task 5: theoretical efficiency vs separation
def plot_efficiency_vs_p(n_values, p_values, filename):
    """
    Theoretical diversification efficiency eta = 1/(1 + 1/p*) plotted
    on the y-axis against the minimum separation distance p* on the
    x-axis. The 9 (p*, eta) markers are overlaid on the continuous
    eta(p) curve to make the underlying functional form visible (the
    markers will lie exactly on the curve since eta is a deterministic
    function of p).
    @param n_values: 1D int array of portfolio sizes (for marker labels).
    @param p_values: 1D float array of corresponding p* values.
    @param filename: output path.
    """
    eta_pts = 1.0 / (1.0 + 1.0 / p_values) #following the eta equation given in the handout

    #continuous theoretical curve over the range of observed p*
    p_grid = np.linspace(p_values.min() * 0.9, p_values.max() * 1.05, 400)
    eta_grid = 1.0 / (1.0 + 1.0 / p_grid)

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.plot(p_grid, eta_grid, color="gray", linewidth=1.2, alpha=0.7,
            label="Efficiency")
    ax.scatter(p_values, eta_pts, s=55, color="navy", zorder=5,
               label="Optimized layouts (n = [2,10])")
    for n, p, e in zip(n_values, p_values, eta_pts):
        ax.annotate(f"n={n}", (p, e), xytext=(5, -10),
                    textcoords="offset points", fontsize=8)
    ax.set_xlabel("Min. pairwise distance (p*)")
    ax.set_ylabel(r"Theoretical efficiency ($\eta$)")
    ax.set_title("Theoretical diversification efficiency vs. separation")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.savefig(filename, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close(fig)

#main
def main():
    csv_data = np.loadtxt("portfolio_layouts.csv", delimiter=",")
    n_values = csv_data[:, 0].astype(int)
    p_values = csv_data[:, 1]

    layouts = np.load("portfolio_layouts.npz") #load npz file populated in part1.py

    #layouts for n in {3, 5, 7} (task 3)
    for n in (3, 5, 7):
        pts = layouts[str(n)]
        p_star = float(p_values[list(n_values).index(n)])
        plot_layout(pts, n, p_star, f"layout_n{n}.png")
        print(f"saved layout_n{n}.png")

    #p* vs n (task 4)
    plot_p_vs_n(n_values, p_values, "p_star_vs_n.png")
    print("saved p_star_vs_n.png")

    #theoretical efficiency vs p* (task 5)
    plot_efficiency_vs_p(n_values, p_values, "eta_vs_p.png")
    print("saved eta_vs_p.png")

if __name__ == "__main__":
    main()