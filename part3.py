r"""
part3.py
Part 3, Task 1 of AA222 Project 3. Uses the GP surrogate from Part 2
to compute expected yearly portfolio profit for n \in [2, 10] together
with a 95% confidence band, and produces a figure analogous to Figure
18.7 from the textbook showing P(n) +/- 1.96 \sigma.
profit model from the handout:
P(n) = n * \eta_n * capital_stock * \alpha_stock − n * c_stock
where \eta_n is evaluated at p*(n), the optimized minimum pairwise
separation distance from Part 1. Because P(n) is affine in \eta for a
fixed n, the GP's Gaussian uncertainty on \eta maps cleanly to Gaussian
uncertainty on P with P_std(n) = n * capital_stock * \alpha_stock · \eta_std(n)
and the 95% confidence band is P_mean +/- 1.96 * P_std.

The numerical results needed for task 3 of part 3 (P_mean and P_std for each n)
are saved to a .csv file
"""
import numpy as np
import matplotlib.pyplot as plt
from part2 import fit_gp, predict_gp

#parameters (from the handout)
CAPITAL_PER_STOCK = 25_000_000.0  #$25,000,000 allocated per position
ALPHA_PER_STOCK = 0.04            #4% expected annualized excess return
COST_PER_STOCK = 550_000.0        #$550k annual cost per position

#profit mapping
def profit_mean_and_std(n_values, eta_mean, eta_std):
    r"""
    Map GP predictions on η at each n to (P_mean, P_std) via the
    handout's profit formula. Profit is affine in η for fixed n, so
    P_std = (n · capital · \alpha) · \eta_std.
    @param n_values: 1D int array of portfolio sizes.
    @param eta_mean: 1D float array of GP posterior means of \eta at p*(n).
    @param eta_std: 1D float array of GP posterior stds of \eta at p*(n).
    @return: (P_mean, P_std), each shape (len(n_values),), in dollars.
    """
    revenue_coef = n_values * CAPITAL_PER_STOCK * ALPHA_PER_STOCK
    total_cost = n_values * COST_PER_STOCK
    P_mean = revenue_coef * eta_mean - total_cost
    P_std = revenue_coef * eta_std
    return P_mean, P_std

#plotting
def plot_profit_vs_n(n_values, P_mean, P_std, filename="profit_vs_n.png"):
    """
    Figure 18.7 style plot in profit space, i.e. GP-mean profit at each
    integer n with a shaded 95% confidence band. Profits are
    reported in millions of USD for readability.
    """
    P_lower = P_mean - 1.96 * P_std
    P_upper = P_mean + 1.96 * P_std
    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    ax.fill_between(n_values, P_lower / 1e6, P_upper / 1e6,
                    color="lightblue", alpha=0.6,
                    label="95% confidence region")
    ax.plot(n_values, P_mean / 1e6, marker="o", color="tab:blue",
            linewidth=2.0, markersize=6,
            label="Expected profit (GP mean)")
    ax.axhline(0.0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Number of stocks (n)")
    ax.set_ylabel("Expected yearly profit (P(n) in USD, millions)")
    ax.set_xticks(n_values)
    ax.set_title("Expected portfolio profit vs. portfolio size")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.savefig(filename, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close(fig)
    print(f"saved {filename}")

def main():
    #load p* values from Part 1
    csv_data = np.loadtxt("portfolio_layouts.csv", delimiter=",")
    n_values = csv_data[:, 0].astype(int)
    p_values = csv_data[:, 1]

    #fit the Part 2 GP and predict η at each p*(n)
    gp = fit_gp()
    eta_mean, eta_std = predict_gp(gp, p_values)

    #map to profit space
    P_mean, P_std = profit_mean_and_std(n_values, eta_mean, eta_std)

    #plot
    plot_profit_vs_n(n_values, P_mean, P_std, "profit_vs_n.png")

    #save numbers for Task 2
    data = np.column_stack([n_values, p_values, eta_mean, eta_std, P_mean, P_std])
    np.savetxt("profit_predictions.csv", data, delimiter=",",
               header="n,p_star,eta_mean,eta_std,P_mean,P_std",
               comments="",
               fmt=["%d", "%.6f", "%.6f", "%.6f", "%.4f", "%.4f"])

if __name__ == "__main__":
    main()