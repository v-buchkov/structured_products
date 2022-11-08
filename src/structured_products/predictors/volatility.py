"""
The file provides functions to predict underlying asset's volatility over the specified period
"""
import numpy as np
import scipy.optimize
import scipy.stats


class Garch:
    """
    GARCH model, which requires list of returns &
    list of predicts for returns (which can be based on any predict function)

    It allows to enter both predicted returns under model return(t) = mu + error(t) (classical setup) and any other

    It is possible to enter any (even different) lags for estimation - this Class supports any GARCH(p,q)

    Formula: variance_predicted(t) =
             const_0 +
             + const_1 * error(t - 1)**2 + const_2 * error(t - 2)**2 + ... + const_p * error(t - p)**2 +
             + const_[p+1] * variance_predicted(t - 1) + ... + const_[p+q] * variance_predicted(t - q)
    """
    def __init__(self, error_lag_order: int, variance_lag_order: int, returns: list, returns_predicted: list,
                 has_intercept=True):

        # Transform logical operation into indicator
        if has_intercept:
            self.has_intercept = 1
        else:
            self.has_intercept = 0

        self.error_lag_order = error_lag_order
        self.variance_lag_order = variance_lag_order
        self.returns = returns
        self.returns_predicted = returns_predicted
        self.coefficients = self.optimize_loglikelihood()

    # Function returns the long-term variance under the specified model:
    # E[variance_predicted(t - x)] is set to be equal to E[variance_predicted(t)] for all x (due to stationarity)
    # E[error**2] is set to be equal to E[variance_predicted(t)] too by variance formula
    def long_term_variance(self):
        if self.has_intercept == 1:
            return self.coefficients[0] / (1 - sum(self.coefficients[1:]))
        else:
            return 0

    # Function generates the list of variances for a given set of constants
    def variance_list(self, constants):

        # If need to estimate the omega, the first in the list of constants is set to be omega =>
        # other are shock coefficients (in accord to the lag order set)
        if self.has_intercept == 1:
            omega = constants[0]
            error_lag_constants = constants[1:(self.error_lag_order + 1)]
            variance_lag_constants = constants[(self.error_lag_order + 1):]
        # Otherwise only shock coefficients are in the list of constants
        else:
            omega = 0
            error_lag_constants = constants[0:self.error_lag_order]
            variance_lag_constants = constants[self.error_lag_order:]

        # Initializing an empty list
        length = len(self.returns)
        variance_list = np.zeros(length)

        # Filling the array with variances by the values in accord with the model
        # If (i < error_lag or i < variance_lag) then uses the long-term variance [described above]
        for i in range(0, max(self.error_lag_order, self.variance_lag_order)):
            variance_list[i] = omega / (1 - sum(error_lag_constants) - sum(variance_lag_constants))

        # Otherwise the full model is used for estimation
        for i in range(max(self.error_lag_order, self.variance_lag_order), length):

            # Sum up shocks until maximum lag that we use for error term
            error_shock_sum = 0
            for j in range(1, self.error_lag_order + 1):
                if i >= j:
                    error_shock_sum += error_lag_constants[j - 1] * \
                                       (self.returns[i - j] - self.returns_predicted[i - j]) ** 2

            # Sum up shocks until maximum lag that we use for previous realized variances
            variance_shock_sum = 0
            for j in range(1, self.variance_lag_order + 1):
                if i >= j:
                    variance_shock_sum += variance_lag_constants[j - 1] * variance_list[i - j]

            # The variance needed is sum of shocks plus the constant
            variance_list[i] = omega + error_shock_sum + variance_shock_sum

        return variance_list

    # Function that returns the value of maximum likelihood after optimizing for the coefficients
    def maximum_likelihood_value(self):
        return np.exp(-self.inverse_loglikelihood(constants=self.coefficients))

    # Function that produces value of inverse loglikelihood for the given set of constants
    # (inverse, as scipy supports only function minimization => will minimize inverse to get max of log_likelihood)
    def inverse_loglikelihood(self, constants):

        variance = self.variance_list(constants)
        returns = self.returns
        sample_size = len(variance)

        loglikelihood = 1 / (2*(sample_size - 1)) * (sum([np.log(var) for var in variance]) +
                                                     sum([(returns[i] - self.returns_predicted[i])**2 / variance[i]
                                                          for i in range(len(variance))]))

        return loglikelihood

    # Function that returns te optimal constants for the model - final output of the estimation
    def optimize_loglikelihood(self):

        # n = # interсepts (1 or 0) + # error constants + # variance constants
        number_of_constants = self.has_intercept + self.error_lag_order + self.variance_lag_order

        # Initialize list of n constants
        constants = np.array([0.001] * number_of_constants)

        # Get optimal constants as minimization of inverse_loglikelihood function
        optimal_constants = scipy.optimize.minimize(self.inverse_loglikelihood, constants,
                                                    bounds=[(0.0001, None)] * number_of_constants)

        return optimal_constants.x


class GJRGarch:
    """
    GJR-GARCH model, which requires list of returns &
    list of predicts for returns (which can be based on any predict function)

    It allows to enter both predicted returns under model return(t) = mu + error(t) (classical setup) and any other

    It is possible to enter any (even different) lags for estimation - this Class supports any GJR-GARCH(p,q)

    Formula: variance_predicted(t) =
             const_0 +
             + (const_1 + dummy_const_1 * dummy_1) * error(t - 1)**2 + ... +
             + (const_p + dummy_const_p * dummy_p) * error(t - p)**2 +
             + const_[p+1] * variance_predicted(t - 1) + ... + const_[p+q] * variance_predicted(t - q)

             for all dummy_p, such that [if return(t) < 0, dummy_p = 1, otherwise dummy_p = 0]
    """
    def __init__(self, error_lag_order: int, variance_lag_order: int, returns: list, returns_predicted: list,
                 has_intercept=True):

        # Transform logical operation into indicator
        if has_intercept:
            self.has_intercept = 1
        else:
            self.has_intercept = 0

        self.error_lag_order = error_lag_order
        self.variance_lag_order = variance_lag_order
        self.returns = returns
        self.returns_predicted = returns_predicted
        self.coefficients = self.optimize_loglikelihood()

    # Function returns the long-term variance under the specified model:
    # E[variance_predicted(t - x)] is set to be equal to E[variance_predicted(t)] for all x (due to stationarity)
    # E[error**2] is set to be equal to E[variance_predicted(t)] too by variance formula
    # Probability of dummy = 1 is assumed to be under c.d.f. of [return(t) ~ N(E(return(t), 1)]
    def long_term_variance(self):
        if self.has_intercept == 1:
            probability_of_dummy_equal_to_one = scipy.stats.norm.cdf(-np.mean(self.returns_predicted))
            return self.coefficients[0] / (1 - sum(self.coefficients[1:(self.error_lag_order + 1)]) -
                                           probability_of_dummy_equal_to_one *
                                           sum(self.coefficients[(self.error_lag_order + 1):
                                                                 (2 * self.error_lag_order + 1)]) -
                                           sum(self.coefficients[(2 * self.error_lag_order + 1):]))
        else:
            return 0

    # Function generates the list of variances for a given set of constants
    def variance_list(self, constants):

        # If need to estimate the omega, the first in the list of constants is set to be omega =>
        # other are shock coefficients (in accord to the lag order set)
        if self.has_intercept == 1:
            omega = constants[0]
            error_lag_constants = constants[1:(self.error_lag_order + 1)]
            dummy_constants = constants[(self.error_lag_order + 1):(2 * self.error_lag_order + 1)]
            variance_lag_constants = constants[(2 * self.error_lag_order + 1):]
        # Otherwise only shock coefficients are in the list of constants
        else:
            omega = 0
            error_lag_constants = constants[0:self.error_lag_order]
            dummy_constants = constants[self.error_lag_order:(2 * self.error_lag_order)]
            variance_lag_constants = constants[(2 * self.error_lag_order):]

        # Initializing an empty list
        length = len(self.returns)
        variance_list = np.zeros(length)

        # Filling the array with variances by the values in accord with the model
        # If (i < error_lag or i < variance_lag) then uses the long-term variance [described above]
        for i in range(0, max(self.error_lag_order, self.variance_lag_order)):
            probability_of_dummy_equal_to_one = scipy.stats.norm.cdf(-np.mean(self.returns_predicted))
            variance_list[i] = omega / (1 - sum(error_lag_constants) -
                                        probability_of_dummy_equal_to_one * sum(dummy_constants) -
                                        sum(variance_lag_constants))

        # Otherwise the full model is used for estimation
        for i in range(max(self.error_lag_order, self.variance_lag_order), length):

            # Sum up shocks until maximum lag that we use for error term
            error_shock_sum = 0
            asymmetry_shock_sum = 0
            for j in range(1, self.error_lag_order + 1):
                return_lagged = self.returns[i - j]
                error_lagged = (return_lagged - self.returns_predicted[i - j]) ** 2
                if i >= j:
                    error_shock_sum += error_lag_constants[j - 1] * error_lagged
                    if return_lagged < 0:
                        asymmetry_shock_sum += dummy_constants[j - 1] * error_lagged

            # Sum up shocks until maximum lag that we use for previous realized variances
            variance_shock_sum = 0
            for j in range(1, self.variance_lag_order + 1):
                if i >= j:
                    variance_shock_sum += variance_lag_constants[j - 1] * variance_list[i - j]

            # The variance needed is sum of shocks plus the constant
            variance_list[i] = omega + error_shock_sum + asymmetry_shock_sum + variance_shock_sum

        return variance_list

    # Function that returns the value of maximum likelihood after optimizing for the coefficients
    def maximum_likelihood_value(self):
        return np.exp(-self.inverse_loglikelihood(constants=self.coefficients))

    # Function that produces value of inverse loglikelihood for the given set of constants
    # (inverse, as scipy supports only function minimization => will minimize inverse to get max of log_likelihood)
    def inverse_loglikelihood(self, constants):

        variance = self.variance_list(constants)
        returns = self.returns
        sample_size = len(variance)

        loglikelihood = 1 / (2*(sample_size - 1)) * (sum([np.log(var) for var in variance]) +
                                                     sum([(returns[i] - self.returns_predicted[i])**2 / variance[i]
                                                          for i in range(len(variance))]))

        return loglikelihood

    # Function that returns te optimal constants for the model - final output of the estimation
    def optimize_loglikelihood(self):

        # n = # interсepts (1 or 0) + 2 times # error constants (for error lag & dummy variables) + # variance constants
        number_of_constants = self.has_intercept + 2 * self.error_lag_order + self.variance_lag_order

        # Initialize list of n constants
        constants = np.array([0.001] * number_of_constants)

        # Get optimal constants as minimization of inverse_loglikelihood function
        optimal_constants = scipy.optimize.minimize(self.inverse_loglikelihood, constants,
                                                    bounds=[(0.0001, None)] * number_of_constants)

        return optimal_constants.x
