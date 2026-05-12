r"""
project3_part3_theoretical.py
Part 3, Task 2(a). Computes total expected yearly profit using the
THEORETICAL diversification efficiency η(p) = 1/(1 + 1/p), i.e.,
ignoring the GP and the simulation data entirely. Loads the optimized
p*(n) values from Part 1 and applies the handout's profit formula at
each n ∈ [2, 10] to answer:
"How many stocks would you recommend if you ignored uncertainty
 and used only the theoretical efficiency?"
"""

import numpy as np


#economic parameters (from the handout)
CAPITAL_PER_STOCK = 25_000_000.0  #$25M per position
ALPHA_PER_STOCK = 0.04            #4% expected annualized excess return
COST_PER_STOCK = 550_000.0        #$550k annual cost per position


def main():
    #load (n, p*) from Part 1
    csv = np.loadtxt("portfolio_layouts.csv", delimiter=",")
    n_values = csv[:, 0].astype(int)
    p_values = csv[:, 1]

    #theoretical efficiency at each p*
    eta_theory = 1.0 / (1.0 + 1.0 / p_values)

    #profit: P(n) = n · η · cap · α − n · cost
    P_theory = (n_values * eta_theory * CAPITAL_PER_STOCK * ALPHA_PER_STOCK
                - n_values * COST_PER_STOCK)

    #print
    print(f"{'n':>3} {'p*':>7} {'eta_theory':>11} {'P_theory ($M)':>14}")
    for n, p, eta, P in zip(n_values, p_values, eta_theory, P_theory):
        print(f"{n:3d} {p:7.4f} {eta:11.4f} {P/1e6:14.4f}")

    best = np.argmax(P_theory)
    print(f"\nargmax: n = {n_values[best]}, P = ${P_theory[best]/1e6:.3f}M")


if __name__ == "__main__":
    main()