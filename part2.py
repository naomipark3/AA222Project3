r"""
part2.py
Fits a GP surrogate to the noisy simulation data of measured 
diversification efficiency vs. minimum separation distance.
The GP is anchored to financial theory through its prior mean 
so that predictions in data-poor regions revert to the theoretical
efficiency curve.
setup details (per the handout):
- squared exponential (RBF) kernel with length scale = 4.0 (fixed)
- prior mean function m(p) = 1/(1 + 1/p) (the theoretical efficiency)
- observation noise std = 0.02, so noise variance \sigma² = 4e-4
- no hyperparameter optimization (sklearn's optimizer=None)

custom prior mean trick: sklearn's GaussianProcessRegressor assumes a
zero-mean GP, so we SUBTRACT m(p) from the training targets, fit a
zero-mean GP to the residuals, then add m(p*) back at predict time.
The predictive covariance is unchanged. Produces the GP-fit plot in
the style of Figure 18.7 of the textbook: predicted mean, 95% confidence 
band (±1.96 \sigma), simulation data, and theoretical efficiency curve on 
the same axes over p \in [1, 10].

The fit_gp / predict_gp functions are written here so Part 3 can import
them and call the same fitted GP for the profit calculations.
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF

#simulation data (table from the handout, Part 2)
P_TRAIN = np.array([2.5, 2.5, 4.5, 4.5, 6.5, 6.5, 8.0, 8.0])
ETA_TRAIN = np.array([0.59, 0.62, 0.82, 0.85, 0.86, 0.88, 0.90, 0.91])

#defined in the problem statement
LENGTH_SCALE = 4.0
NOISE_STD = 0.02

#prior mean and GP fit
def theoretical_efficiency(p):
    r"""
    Theoretical diversification efficiency \eta(p) = 1/(1 + 1/p). Used
    both as the GP prior mean and as the reference curve in the plot.
    @param p: scalar or array-like separation distance(s).
    @return: ndarray (or scalar) of \eta values.
    """
    p = np.asarray(p, dtype=float)
    return 1.0 / (1.0 + 1.0 / p)

def fit_gp(p_train=P_TRAIN, eta_train=ETA_TRAIN):
    r"""
    Fit a sklearn GaussianProcessRegressor to (p, \eta) data with a fixed
    RBF kernel and a non-zero prior mean implemented via residuals.
    Hyperparameters are LOCKED (length_scale_bounds='fixed' on the
    kernel, optimizer=None on the regressor) so that nothing about the
    GP is tuned to the data beyond conditioning on the residuals.
    @param p_train: training separations, shape (N,).
    @param eta_train: training efficiencies, shape (N,).
    @return: fitted GaussianProcessRegressor (operating on residuals,
    always pair with predict_gp to get back to η space).
    """
    residuals = eta_train - theoretical_efficiency(p_train)
    kernel = RBF(length_scale=LENGTH_SCALE, length_scale_bounds="fixed")
    gp = GaussianProcessRegressor(
        kernel=kernel,
        alpha=NOISE_STD**2,
        optimizer=None,
        normalize_y=False,
    )
    gp.fit(p_train.reshape(-1, 1), residuals)
    return gp

def predict_gp(gp, p_query):
    """
    Posterior mean and std of the GP at query points, with the prior
    mean function added back to the residual posterior mean. Std is
    unchanged by the mean shift.
    @param gp: fitted GP regressor (from fit_gp).
    @param p_query: scalar or array of separation distances to query.
    @return: (mean, std), each shape (len(p_query),).
    """
    p_query = np.atleast_1d(p_query).astype(float)
    mean_resid, std = gp.predict(p_query.reshape(-1, 1), return_std=True)
    mean = mean_resid + theoretical_efficiency(p_query)
    return mean, std

#plotting
def plot_gp_fit(filename="gp_fit.png"):
    """
    Figure 18.7 style plot: 95% confidence band, theoretical \eta curve,
    GP predicted mean function, and simulation data points used to fit the gp,
    on a common axis over p \in [1, 10]
    """
    gp = fit_gp()
    p_grid = np.linspace(1.0, 10.0, 400)
    mean, std = predict_gp(gp, p_grid)
    upper = mean + 1.96 * std
    lower = mean - 1.96 * std
    eta_theory = theoretical_efficiency(p_grid)

    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    ax.fill_between(p_grid, lower, upper, color="lightblue", alpha=0.6,
                    label="95% confidence region")
    ax.plot(p_grid, eta_theory, color="black", linewidth=1.3,
            label=r"Theoretical efficiency curve $\eta(p) = 1/(1+1/p)$")
    ax.plot(p_grid, mean, color="tab:blue", linewidth=2.0,
            label="The predicted GP mean function")
    ax.scatter(P_TRAIN, ETA_TRAIN, color="black", s=28, zorder=5,
               label="Simulation data")
    ax.set_xlabel(r"Separation distance (p)") #x-axis here is a generic continuous separation distance, not optimal p*
    ax.set_ylabel(r"Diversification efficiency ($\eta$)")
    ax.set_xlim(1.0, 10.0)
    ax.set_title(r"GP fit to simulation data (squared exponential kernel, length scale = 4.0)")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.savefig(filename, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close(fig)
    print(f"saved {filename}")


if __name__ == "__main__":
    plot_gp_fit("gp_fit.png")