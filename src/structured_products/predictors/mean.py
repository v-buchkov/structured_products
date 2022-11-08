"""
The file provides functions to predict underlying asset's mean over the specified period
"""
import numpy as np
import scipy.optimize
import scipy.stats


class ARMA:
    """
    ARMA model, which requires list of returns as the only input

    It is possible to enter any (even different) lags for estimation - this Class supports any ARMA(p,q)

    Formula: return_predicted(t) = const_0 + const_1 * return(t - 1) + ... + const_p * dummy_p * return(t - p) +
                                   + const_[p+1] * error(t - 1) + ... + const_[p+q] * error(t - q)
    """
    def __init__(self, return_lag_order: int, error_lag_order: int, returns: list, has_intercept=True):

        # Transform logical operation into indicator
        if has_intercept:
            self.has_intercept = 1
        else:
            self.has_intercept = 0

        self.return_lag_order = return_lag_order
        self.error_lag_order = error_lag_order
        self.returns = returns
        self.coefficients = self.optimize_loglikelihood()

    # Function calculates the long-term mean for the estimated model
    # E(return[x]) is set to be equal to long_term_mean
    # E(error[x]) is set to be equal to zero
    def long_term_mean(self):
        if self.has_intercept == 1:
            return self.coefficients[0] / (1 - sum(self.coefficients[1:(self.return_lag_order + 1)]))
        else:
            return 0

    # Function generates the list of variances for a given set of constants
    def forecasted_returns_list(self, constants):

        # If need to estimate the intercept, the first in the list of constants is set to be the intercept =>
        # other are shock coefficients (in accord to the lag order set)
        if self.has_intercept == 1:
            intercept = constants[0]
            return_lag_constants = constants[1:(self.error_lag_order + 1)]
            error_lag_constants = constants[(self.error_lag_order + 1):]
        # Otherwise only shock coefficients are in the list of constants
        else:
            intercept = 0
            return_lag_constants = constants[0:self.error_lag_order]
            error_lag_constants = constants[self.error_lag_order:]

        # Initializing an empty list
        length = len(self.returns)
        forecasted_returns_list = np.zeros(length)

        # If do not have enough past returns for the lag order, use long-term variance
        for i in range(0, self.return_lag_order):
            forecasted_returns_list[i] = intercept / (1 - sum(return_lag_constants))

        # Otherwise, append values of predicted returns
        for i in range(self.return_lag_order, length):

            # Append all shocks that arise from past returns
            returns_shock_sum = 0
            for j in range(1, self.return_lag_order + 1):
                if i >= j:
                    returns_shock_sum += return_lag_constants[j - 1] * self.returns[i - j]

            # Append all shocks that arise from past errors (white noise)
            error_shock_sum = 0
            for j in range(1, self.error_lag_order + 1):
                if i >= j:
                    error_shock_sum += error_lag_constants[j - 1] * (self.returns[i - j] -
                                                                     forecasted_returns_list[i - j])

            # Calculate final value of predicted returns
            forecasted_returns_list[i] = intercept + returns_shock_sum + error_shock_sum

        return forecasted_returns_list

    # Function that returns the value of maximum likelihood after optimizing for the coefficients
    def maximum_likelihood_value(self):
        return np.exp(-self.inverse_loglikelihood(constants=self.coefficients))

    # Function that produces value of inverse loglikelihood for the given set of constants
    # (inverse, as scipy supports only function minimization => will minimize inverse to get max of log_likelihood)
    def inverse_loglikelihood(self, constants):

        forecasted = self.forecasted_returns_list(constants)

        errors = (np.array(forecasted) - np.array(self.returns))**2

        loglikelihood = 1/2 * sum(errors)

        return loglikelihood

    # Function that returns te optimal constants for the model - final output of the estimation
    def optimize_loglikelihood(self):

        # n = # interсepts (1 or 0) + # error constants + # variance constants
        number_of_constants = self.has_intercept + self.return_lag_order + self.error_lag_order

        # Initialize list of n constants
        constants = np.array([0.001] * number_of_constants)

        # Get optimal constants as minimization of inverse_loglikelihood function
        optimal_constants = scipy.optimize.minimize(self.inverse_loglikelihood, constants,
                                                    bounds=[(None, None)] * number_of_constants)

        return optimal_constants.x


class StrategyPerformance:
    """
    Class that contains different measures of performance of some strategy versus the benchmark
    """
    def __init__(self, strategy_daily_returns: list, market_daily_returns: list):
        self.strategy = strategy_daily_returns
        self.market = market_daily_returns

    def beta(self):
        return np.cov(self.strategy, self.market)[1][1] / np.var(self.market)

    def alpha(self):
        return 252 * np.mean([self.strategy[i] - self.beta() * self.market[i] for i in range(0, len(self.strategy))])


# class SpecificationARMA:
#     """
#     Class defines several operations to determine the correct application of ARMA model:
#          * checks the stationarity of the timeseries (Augmented Dickey-Fuller test)
#          * checks, if overall autoregressive patter is present in the timeseries (Ljung-Box tests)
#          * returns the optimal lags for Autoregressive & Moving Average parts (Box-Jenkins approach)
#     """
#     def __init__(self, timeseries: list, max_lag_to_consider=10):
#         self.max_lag = max_lag_to_consider
#
#         self.timeseries = timeseries
#         self.n = len(timeseries)
#
#         # Calculate mean and variance of the considered timeseries
#         timeseries_mean = np.mean(timeseries)
#         timeseries_variance = np.var(timeseries)
#         autoregressive_function = []
#         lagged_returns_dataframe = {}
#         returns_dataframe = {}
#         for s in range(1, self.max_lag):
#             y_lagged = timeseries[s:]
#             list_to_operate = timeseries[s:(self.n - self.max_lag + s)]
#             lagged_returns_dataframe['Order {}'.format(s)] = list_to_operate
#             returns_dataframe['Order {}'.format(s)] = [list_to_operate[i] - list_to_operate[i - 1]
#                                                        for i in range(1, len(list_to_operate))]
#             y_t = timeseries[:(self.n - s)]
#             autoregressive_function.append([s, round(sum([(y_t[j] - timeseries_mean) * (y_lagged[j] - timeseries_mean)
#                                                           for j in range(len(y_t))])
#                                                      / self.n, 5) / timeseries_variance])
#         self.autoregressive_function = autoregressive_function
#         self.lagged_returns_dataframe = lagged_returns_dataframe
#         self.returns_dataframe = returns_dataframe
#
#     def stationary(self, alpha_significance):
#         """
#         Uses Augmented Dickey-Fuller test. Returns True (stationary) or False (non-stationary)
#         """
#         orders_names = ['Order {}'.format(s) for s in range(1, self.max_lag)]
#         df_dataframe = pd.DataFrame(self.returns_dataframe, columns=['Y', 'Fixed Order 1'] + orders_names)
#         y_df = df_dataframe['Y']
#         x_df = df_dataframe[orders_names + ['Fixed Order 1']]
#         x_df = sm.add_constant(x_df)
#         estimation_df = sm.OLS(y_df, x_df).fit()
#         p_value = estimation_df.pvalues['Order 1']
#
#         if p_value <= alpha_significance:
#             return True
#         else:
#             return False
#
#     def overall_autoregressive_pattern(self, alpha_significance):
#         """
#         Uses Ljung-Box test. Returns
#         True (autoregressive patter is present) or False (autoregressive patter is not present)
#         """
#         q_likeness_value = self.n * (self.n + 2) * sum([order[1] ** 2 /
#                                                         (self.n - order[0]) for order in self.autoregressive_function])
#         chi_square_p_value = 1 - scipy.stats.chi2.cdf(x=q_likeness_value, df=len(self.autoregressive_function))
#
#         if chi_square_p_value <= alpha_significance:
#             return True
#         else:
#             return False
#
#     # Box-Jenkins
#     def optimal_lag_orders(self, alpha_significance):
#         """
#         Uses Box-Jenkins approach. Returns [optimal AR lag, optimal MA lag]
#         """
#         # Initialize the orders for columns in the dataframe of lags
#         orders_names = ['Order {}'.format(s) for s in range(1, self.max_lag)]
#
#         """
#         AR part
#         """
#         # H0: AR(0) => PACF / Std ~ N(0,1)
#         ar_part_z_stats = [[order[0], order[1] *
#                             np.sqrt(self.n / (1 + 2 * sum([o[1] for o in self.autoregressive_function[:order[0]]])))]
#                            for order in self.autoregressive_function]
#         # Get p-values from z-statistics
#         ar_part_p_values = [1 - scipy.stats.norm.cdf(z) for z in ar_part_z_stats]
#
#         """
#         MA part
#         """
#         # H0: MA(0) => PACF is generated by OLS
#         self.lagged_returns_dataframe['Y'] = self.timeseries[:(self.n - self.max_lag)]
#         self.returns_dataframe['Fixed Order 1'] = self.lagged_returns_dataframe['Order 1'][1:]
#         self.returns_dataframe['Y'] = [self.timeseries[:(self.n - self.max_lag)][i] -
#                                        self.timeseries[:(self.n - self.max_lag)][i - 1] for i in
#                                        range(1, len(self.timeseries[:(self.n - self.max_lag)]))]
#         bj_dataframe = pd.DataFrame(self.lagged_returns_dataframe, columns=['Y'] + orders_names)
#
#         # Estimate OLS
#         y = bj_dataframe['Y']
#         x = bj_dataframe[orders_names]
#         x = sm.add_constant(x)
#         estimation = sm.OLS(y, x).fit()
#         # Get p-values
#         ma_part_p_values = estimation.pvalues
#
#         # Try to determine optimal autoregressive lag
#         ar_optimal_lag = 0
#         for i in range(1, len(ar_part_p_values)):
#             # Lag is found, if the p-value is less that alpha level
#             # Exclude lags, if the 1st condition is not consecutive (cannot claim that optimal lag = 3, if
#             # only 1 and 3 satisfy the condition
#             if ar_part_p_values[i - 1] <= alpha_significance < ar_part_p_values[i]:
#                 ar_optimal_lag = i
#                 break
#
#         # Try to determine optimal moving average lag
#         ma_optimal_lag = 0
#         for i in range(1, len(ma_part_p_values)):
#             # Lag is found, if the p-value is less that alpha level
#             # Exclude lags, if the 1st condition is not consecutive (cannot claim that optimal lag = 3, if
#             # only 1 and 3 satisfy the condition
#             if ar_part_p_values[i - 1] <= alpha_significance < ar_part_p_values[i]:
#                 ma_optimal_lag = i
#                 break
#
#         return [ar_optimal_lag, ma_optimal_lag]
