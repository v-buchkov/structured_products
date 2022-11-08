import numpy as np
import itertools
import random


class DistributionParams:

    def __init__(self, mean_return_list, volatility_list, correlation_matrix):
        self.means = mean_return_list
        self.stds = volatility_list
        self.var_covar = self.calculate_var_covar_matrix(correlation_matrix)
        self.n_assets = len(mean_return_list)

    def worst_of_mean_return(self, time_till_maturity, number_of_worst_of):
        simulated_final_returns = geometric_brownian_motion_simulate_paths(mean_returns=self.means,
                                                                           var_covar=self.var_covar,
                                                                           time_till_maturity=time_till_maturity)
        return np.mean([path.sort()[number_of_worst_of - 1] for path in simulated_final_returns])

    # calculate variance-covariance matrix
    # pass list of standard deviations and correlation matrix
    def calculate_var_covar_matrix(self, corr_matrix):
        std_list = self.stds

        # calculate variance-covariance matrix
        var_covar = []
        for j in range(0, self.n_assets):
            var_covar.append([std_list[i] * std_list[k] * corr_matrix[i][k] for k in range(0, self.n_assets)])

        return var_covar


def optimize_by_sharpe_ratio(mean_returns, var_covar):

    n_assets = len(mean_returns)
    # initiate Sharpe ratio set
    sharpe_ratio_set = {}

    for iteration in itertools.product(range(1, 100), repeat=n_assets):

        # get list of potential weights in the portfolio
        weights = np.array([w / 100 for w in iteration])
        # calculate weighted mean return
        portfolio_mean = np.dot(weights, mean_returns).item()
        # get variance via weighted variance-covariance matrix
        portfolio_variance = sum([var_covar[i][j] * weights[i] * weights[j]
                                  for i, j in itertools.product(range(0, n_assets), repeat=2)]).item()
        # record dictionary: key - Sharpe Ratio, value - list of weights
        sharpe_ratio_set[portfolio_mean / np.sqrt(portfolio_variance)] = weights

    # get the best attainable Sharpe Ratio
    best_sharpe_ratio = max(sharpe_ratio_set.keys())

    # return [weights in the best SR portfolio, the best SR]
    return [sharpe_ratio_set[best_sharpe_ratio], best_sharpe_ratio]


def geometric_brownian_motion_simulate_paths(mean_returns, var_covar, time_till_maturity, number_of_paths=20000):

    n_objects = len(mean_returns)

    time = np.linspace(0, time_till_maturity / 252, time_till_maturity)
    d_time = time[1] - time[0]
    cholesky = np.linalg.cholesky(var_covar)

    drift = np.exp(np.array([(mean_returns[j] - 0.5 * var_covar[j][j])
                             for j in range(n_objects)]) * d_time) ** time_till_maturity

    diffusion = np.exp([np.matmul(cholesky, np.random.normal(0, 1, size=n_objects)) * np.sqrt(d_time)
                        for m in range(number_of_paths)])

    return drift * np.array([return_vs_start for return_vs_start in np.vstack(diffusion)])


def simulate_issuer_default(default_p, path_length):
    """
    Function randomly generates the order numbers for some array of instruments that will be defaulted under default_p

    E.g., it provides [3, 33, 121] for 500 instruments => 3rd, 33rd and 121st will be defaulted in this simulation
    """
    return [random.randrange(0, int(path_length * default_p)) for note in range(int(path_length * default_p))]
