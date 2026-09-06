"""GP forecasts in log-price space. All fitting uses only supplied history."""
from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy.special import logsumexp
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, Matern, WhiteKernel

ASSETS = ['A', 'B', 'C']

@dataclass
class Forecast:
    mean: np.ndarray
    covariance: np.ndarray
    day: np.ndarray
    model: object
    mean_function: str

    def price_interval(self, mass=.9):
        z = norm.ppf((1 + mass) / 2)
        sd = np.sqrt(np.maximum(np.diag(self.covariance), 0))
        return np.exp(self.mean), np.exp(self.mean-z*sd), np.exp(self.mean+z*sd)


def forecast(history, asset='A', horizon=20, kernel='matern', length=25.,
             amplitude=.15, noise=.015, mean='flat', optimize=False):
    """Fixed hyperparameters by default: these specify the GP prior.

    WhiteKernel contributes independent observation noise to future predictions.
    Optional optimization is empirical Bayes, not integration over hyperparameters.
    Mean coefficients are plug-in estimates; their uncertainty is not integrated.
    """
    if horizon < 1 or length <= 0 or amplitude <= 0 or noise <= 0:
        raise ValueError('Positive horizon and kernel parameters required')
    t = np.asarray(history.day, float)
    y = np.log(np.asarray(history[asset], float))
    future = np.arange(t[-1]+1, t[-1]+horizon+1)
    if mean == 'flat':
        # A constant prior mean anchored at the last observed log price.
        slope, intercept = 0., y[-1]
    elif mean == 'linear':
        slope, intercept = np.polyfit(t, y, 1)
    else:
        raise ValueError('mean must be flat or linear')
    baseline = lambda x: intercept + slope*x
    bounds = (2., 200.) if optimize else 'fixed'
    if kernel == 'rbf':
        signal = RBF(length, bounds)
    elif kernel == 'matern':
        signal = Matern(length, bounds, nu=1.5)
    elif kernel == 'rough':
        signal = Matern(length, bounds, nu=.5)
    else:
        raise ValueError('kernel must be rbf, matern, or rough')
    k = ConstantKernel(amplitude**2, 'fixed') * signal + WhiteKernel(noise**2, 'fixed')
    gp = GaussianProcessRegressor(kernel=k, alpha=1e-9, normalize_y=False,
                                  optimizer='fmin_l_bfgs_b' if optimize else None)
    gp.fit(t[:, None], y-baseline(t))
    mu, cov = gp.predict(future[:, None], return_cov=True)
    return Forecast(mu+baseline(future), cov, future, gp, mean)


def prior_paths(days, kernel='matern', length=25., amplitude=.15, seed=7):
    """Noise-free functions from a zero-mean log-price residual prior."""
    k = {'rbf': RBF(length), 'matern': Matern(length, nu=1.5),
         'rough': Matern(length, nu=.5)}[kernel]
    covariance = amplitude**2 * k(np.asarray(days)[:, None])
    return np.random.default_rng(seed).multivariate_normal(np.zeros(len(days)), covariance+1e-10*np.eye(len(days)), 5).T


def mixture_forecast(history, asset='A', lengths=(8., 25., 80.), prior=(1/3, 1/3, 1/3), **kwargs):
    """Discrete Bayesian averaging over length scales, conditional on other choices.

    Returns components and posterior probabilities. Mixture draws preserve the
    non-Gaussian predictive mixture rather than replacing it with one Gaussian.
    """
    prior = np.asarray(prior, float)
    if len(prior) != len(lengths) or np.any(prior <= 0) or not np.isclose(prior.sum(), 1):
        raise ValueError('Positive prior probabilities must sum to one')
    models = [forecast(history, asset, length=l, **kwargs) for l in lengths]
    log_weights = np.log(prior) + np.array([f.model.log_marginal_likelihood_value_ for f in models])
    weights = np.exp(log_weights-logsumexp(log_weights))
    return models, weights


def validation(history, asset='A', horizon=20, origins=(80, 100, 120, 140), **kwargs):
    """Non-overlapping endpoint checks. Origins are training row counts.

    Four windows are a small diagnostic, not evidence of reliable calibration.
    """
    rows = []
    for n in origins:
        if n+horizon > len(history):
            continue
        train = history.iloc[:n]
        actual = float(history[asset].iloc[n+horizon-1])
        f = forecast(train, asset, horizon, **kwargs)
        mu, variance = f.mean[-1], f.covariance[-1, -1]
        returns = np.diff(np.log(train[asset]))
        rw_mu = np.log(train[asset].iloc[-1])
        rw_var = horizon*np.var(returns, ddof=1)
        for name, m, v in [('GP', mu, variance), ('Random walk', rw_mu, rw_var)]:
            sd = np.sqrt(max(v, 1e-12))
            low, high = np.exp(m + np.array([-1, 1])*norm.ppf(.95)*sd)
            rows.append(dict(origin=n, model=name, actual=actual, median=np.exp(m),
                             absolute_error=abs(actual-np.exp(m)), covered=low <= actual <= high,
                             width=high-low, log_score=norm.logpdf(np.log(actual), m, sd)-np.log(actual)))
    return pd.DataFrame(rows)


def portfolio_scenarios(history, forecasts, weights, draws=10000, seed=42, correlated=True):
    """Approximate endpoint coupling, NOT a fitted multi-output GP.

    Uses shrunk historical daily-return correlation as a proxy for correlation
    of endpoint forecast errors. This assumption is deliberately inspectable.
    """
    weights = np.asarray(weights, float)
    if weights.shape != (4,) or not np.isfinite(weights).all() or np.any(weights < 0) or not np.isclose(weights.sum(), 1):
        raise ValueError('Supply A, B, C, cash weights >= 0 summing to 1')
    corr = np.corrcoef(np.diff(np.log(history[ASSETS].to_numpy()), axis=0), rowvar=False)
    corr = .8*corr + .2*np.eye(3) if correlated else np.eye(3)
    z = np.random.default_rng(seed).multivariate_normal(np.zeros(3), corr, draws)
    mu = np.array([f.mean[-1] for f in forecasts])
    sd = np.sqrt([f.covariance[-1, -1] for f in forecasts])
    ratios = np.exp(mu+z*sd) / history[ASSETS].iloc[-1].to_numpy()
    return 10000*(ratios@weights[:3]+weights[3]), corr
