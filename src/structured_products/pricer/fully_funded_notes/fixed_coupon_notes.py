"""
The file provides blueprints to create fixed coupon structured notes objects
----------------------------
The convention for all fractions is decimals
(e.g., if the price is 95% from notional, the ClassObject.price() will return 0.95)
"""
import scipy.stats
import numpy as np
from src.structured_products.pricer.options.vanilla_options import EuropeanCall, EuropeanPut
from src.structured_products.pricer.options.barrier_options import UpInBarrierCall, DownInBarrierPut

DAYS_IN_YEAR_CONVENTION = 365


class ReverseConvertible:
    """
    RC = sold Vanilla EuropeanPut + money market investment
    """
    def __init__(self, time_till_maturity: float, coupon_frequency: float, underlying_mean: float,
                 underlying_volatility: float, current_spot: float, initial_spot_fixing: float, risk_free_rate: float,
                 strike_decimal):
        self.term = time_till_maturity
        self.mean = underlying_mean
        self.sigma = underlying_volatility
        self.spot = current_spot
        self.initial_spot = initial_spot_fixing
        self.rf = risk_free_rate
        self.cpn_freq = coupon_frequency
        if strike_decimal is not None:
            self.strike_price = strike_decimal * self.initial_spot

    # Function allows to calculate the required strike_percentage by defining it as a threshold that is allowed to be
    # breached only in loss_probability * 100% cases => VaR with "alpha" = loss_probability by definition
    def set_strike_level_by_value_at_risk(self, loss_probability):
        self.strike_price = np.exp(self.mean * self.term -
                                   scipy.stats.norm.ppf(1 - loss_probability) * self.sigma *
                                   np.sqrt(self.term)) * self.initial_spot

    # Function calculates premium of the option (in price space) by simple Black-Scholes-Merton
    def option_premium(self):
        return EuropeanPut(self.term, self.sigma, self.strike_price / self.initial_spot, self.spot, self.initial_spot,
                           self.mean).option_premium()

    # Function produces the fair price of the structured product as decimal fraction of the notional invested
    def price(self):
        return 1 / (1 + self.rf) ** self.term - self.option_premium() / self.initial_spot

    # Get bid as the fair decimal fraction price minus the commission
    def bid(self, initial_commission, spread_from_mid_price, days_from_last_coupon):
        return self.price() - spread_from_mid_price - days_from_last_coupon / DAYS_IN_YEAR_CONVENTION * \
               self.coupon(initial_commission)

    # Get offer as the fair decimal fraction price plus the commission
    def offer(self, initial_commission, spread_from_mid_price, days_from_last_coupon):
        return self.price() + spread_from_mid_price + days_from_last_coupon / DAYS_IN_YEAR_CONVENTION * \
               self.coupon(initial_commission)

    # Calculate the annualized coupon rate in decimals
    def coupon(self, commission):
        return (self.price() - commission) / sum([self.cpn_freq / (1 + self.rf)**(j * self.cpn_freq)
                                                  for j in range(1, int(self.term / self.cpn_freq) + 1)])

    # The function provides decimal fraction of the invested capital that the client receives at maturity
    # (without the coupons received)
    def execute(self):
        initial_fixing = self.initial_spot
        final_fixing = self.spot

        # If at maturity price falls below the strike_price => loses the difference between strike and final_fixing
        if final_fixing < self.strike_price:
            return 1 + (final_fixing - self.strike_price) / initial_fixing
        # Otherwise, 100% of the notional is returned
        else:
            return 1

    # Final result => return (+ or -) as decimal fraction, including the coupons received and excluding the commissions
    def final_result(self, commission_paid):
        return (1 + self.option_premium() / self.spot - commission_paid) * \
               (1 + self.rf)**self.term + self.execute() - 1


class BarrierReverseConvertible:
    """
    BRC = sold Down&In Barrier EuropeanPut + money market investment
    """
    def __init__(self, time_till_maturity: float, coupon_frequency: float, underlying_mean: float,
                 underlying_volatility: float, current_spot: float, initial_spot_fixing: float, risk_free_rate: float,
                 barrier_decimal):
        self.term = time_till_maturity
        self.mean = underlying_mean
        self.sigma = underlying_volatility
        self.spot = current_spot
        self.initial_spot = initial_spot_fixing
        self.rf = risk_free_rate
        self.cpn_freq = coupon_frequency
        if barrier_decimal is not None:
            self.barrier_price = barrier_decimal * self.initial_spot

    # Function allows to calculate the required barrier_percentage by defining it as a threshold that is allowed to be
    # breached only in loss_probability * 100% cases => VaR with "alpha" = loss_probability by definition
    def set_barrier_level_by_value_at_risk(self, loss_probability):
        self.barrier_price = np.exp(self.mean * self.term -
                                    scipy.stats.norm.ppf(1 - loss_probability) * self.sigma *
                                    np.sqrt(self.term)) * self.initial_spot

    # Function calculates premium of the option (in price space) by simple Black-Scholes-Merton
    def option_premium(self):
        return DownInBarrierPut(self.term, self.sigma, 1, self.barrier_price / self.initial_spot, self.spot,
                                self.initial_spot, self.mean).option_premium()

    # Function produces the fair price of the structured product as decimal fraction of the notional invested
    def price(self):
        return 1 / (1 + self.rf) ** self.term - self.option_premium() / self.initial_spot

    # Get bid as the fair decimal fraction price minus the commission
    def bid(self, initial_commission, spread_from_mid_price, days_from_last_coupon):
        return self.price() - spread_from_mid_price - days_from_last_coupon / DAYS_IN_YEAR_CONVENTION * \
               self.coupon(initial_commission)

    # Get offer as the fair decimal fraction price plus the commission
    def offer(self, initial_commission, spread_from_mid_price, days_from_last_coupon):
        return self.price() + spread_from_mid_price + days_from_last_coupon / DAYS_IN_YEAR_CONVENTION * \
               self.coupon(initial_commission)

    # Calculate the annualized coupon rate in decimals
    def coupon(self, commission):
        return (self.price() - commission) / sum([self.cpn_freq / (1 + self.rf)**(j * self.cpn_freq)
                                                  for j in range(1, int(self.term / self.cpn_freq) + 1)])

    # The function provides decimal fraction of the invested capital that the client receives at maturity
    # (without the coupons received)
    def execute(self):
        initial_fixing = self.initial_spot
        final_fixing = self.spot

        # If at maturity price falls below the barrier_price => loses the difference between
        # strike (100%) and final_fixing (+100% of the capital invested as money market => "100%"s cancel out)
        if final_fixing <= self.barrier_price:
            return final_fixing / initial_fixing
        # Otherwise, 100% of the notional is returned
        else:
            return 1

    # Final result => return (+ or -) as decimal fraction, including the coupons received and excluding the commissions
    def final_result(self, commission_paid):
        return (1 + self.option_premium() / self.spot - commission_paid) * \
               (1 + self.rf)**self.term + self.execute() - 1


class InverseReverseConvertible:
    """
    I-RC = sold Vanilla EuropeanCall +
           bought far Vanilla EuropeanCall to hedge maximum loss to be 100% with strike (strike_percentage + 1) +
           money market investment
    """
    def __init__(self, time_till_maturity: float, coupon_frequency: float, underlying_mean: float,
                 underlying_volatility: float, current_spot: float, initial_spot_fixing: float, risk_free_rate: float,
                 strike_decimal):
        self.term = time_till_maturity
        self.mean = underlying_mean
        self.sigma = underlying_volatility
        self.spot = current_spot
        self.initial_spot = initial_spot_fixing
        self.rf = risk_free_rate
        self.cpn_freq = coupon_frequency
        if strike_decimal is not None:
            self.strike_price = strike_decimal * self.initial_spot

    # Function allows to calculate the required strike_percentage by defining it as a threshold that is allowed to be
    # breached only in loss_probability * 100% cases => VaR with "alpha" = loss_probability by definition
    def set_strike_level_by_value_at_risk(self, loss_probability):
        # p transformation to account for bought otm call => classical tail loss probability should EXCLUDE the
        # probability, hedged by the bought OTM call
        p_transformed = loss_probability + 1 - scipy.stats.lognorm.cdf(2, self.sigma, loc=self.mean)

        self.strike_price = np.exp(self.mean * self.term - scipy.stats.norm.ppf(p_transformed) * self.sigma *
                                   np.sqrt(self.term)) * self.initial_spot

    # Function calculates premium of the option (in price space) by simple Black-Scholes-Merton
    def option_premium(self):
        return EuropeanCall(self.term, self.sigma, self.strike_price / self.initial_spot, self.spot, self.initial_spot,
                            self.mean).option_premium() \
               - EuropeanCall(self.term, self.sigma, self.strike_price / self.initial_spot + 1, self.spot,
                              self.initial_spot, self.mean).option_premium()

    # Function produces the fair price of the structured product as decimal fraction of the notional invested
    def price(self):
        return 1 / (1 + self.rf) ** self.term - self.option_premium() / self.initial_spot

    # Get bid as the fair decimal fraction price minus the commission
    def bid(self, initial_commission, spread_from_mid_price, days_from_last_coupon):
        return self.price() - spread_from_mid_price - days_from_last_coupon / DAYS_IN_YEAR_CONVENTION * \
               self.coupon(initial_commission)

    # Get offer as the fair decimal fraction price plus the commission
    def offer(self, initial_commission, spread_from_mid_price, days_from_last_coupon):
        return self.price() + spread_from_mid_price + days_from_last_coupon / DAYS_IN_YEAR_CONVENTION * \
               self.coupon(initial_commission)

    # Calculate the annualized coupon rate in decimals
    def coupon(self, commission):
        return (self.price() - commission) / sum([self.cpn_freq / (1 + self.rf)**(j * self.cpn_freq)
                                                  for j in range(1, int(self.term / self.cpn_freq) + 1)])

    # The function provides decimal fraction of the invested capital that the client receives at maturity
    # (without the coupons received)
    def execute(self):
        initial_fixing = self.initial_spot
        final_fixing = self.spot

        # If at maturity price grows above TWICE the strike_price =>
        # loses the maximum = 100% of notional (the rest of the growth is hedged by bought OTM call)
        if final_fixing >= 2 * self.strike_price:
            return 0
        # Else if at maturity price grows above the strike_price => loses the difference between final_fixing and strike
        elif final_fixing >= self.strike_price:
            return 1 - (final_fixing - self.strike_price) / initial_fixing
        # Otherwise, 100% of the notional is returned
        else:
            return 1

    # Final result => return (+ or -) as decimal fraction, including the coupons received and excluding the commissions
    def final_result(self, commission_paid):
        return (1 + self.option_premium() / self.spot - commission_paid) * \
               (1 + self.rf)**self.term + self.execute() - 1


class InverseBarrierReverseConvertible:
    """
    I-BRC = sold Up&In Barrier EuropeanCall +
            bought far Vanilla EuropeanCall to hedge maximum loss to be 100% with strike (strike_percentage + 1) +
            money market investment
    """
    def __init__(self, time_till_maturity: float, coupon_frequency: float, underlying_mean: float,
                 underlying_volatility: float, current_spot: float, initial_spot_fixing: float, risk_free_rate: float,
                 initial_commission: float, barrier_decimal):
        self.term = time_till_maturity
        self.mean = underlying_mean
        self.sigma = underlying_volatility
        self.spot = current_spot
        self.initial_spot = initial_spot_fixing
        self.rf = risk_free_rate
        self.cpn_freq = coupon_frequency
        self.initial_commission = initial_commission
        if barrier_decimal is not None:
            self.barrier_price = barrier_decimal * self.initial_spot

    # Function allows to calculate the required barrier_percentage by defining it as a threshold that is allowed to be
    # breached only in loss_probability * 100% cases => VaR with "alpha" = loss_probability by definition
    def set_barrier_level_by_value_at_risk(self, loss_probability):
        # p transformation to account for bought otm call => classical tail loss probability should EXCLUDE the
        # probability, hedged by the bought OTM call
        p_transformed = loss_probability + 1 - scipy.stats.lognorm.cdf(2, self.sigma, loc=self.mean)

        self.barrier_price = np.exp(self.mean * self.term - scipy.stats.norm.ppf(p_transformed) * self.sigma *
                                    np.sqrt(self.term)) * self.initial_spot

    # Function calculates premium of the option (in price space) by simple Black-Scholes-Merton
    def option_premium(self):
        return UpInBarrierCall(self.term, self.sigma, 1, self.barrier_price / self.initial_spot, self.spot,
                               self.initial_spot, self.mean).option_premium() - \
               EuropeanCall(self.term, self.sigma, (2 - self.initial_commission) * (1 + self.rf)**self.term,
                            self.spot, self.initial_spot, self.mean).option_premium()

    # Function produces the fair price of the structured product as decimal fraction of the notional invested
    def price(self):
        return 1 / (1 + self.rf) ** self.term - self.option_premium() / self.initial_spot

    # Get bid as the fair decimal fraction price minus the commission
    def bid(self, initial_commission, spread_from_mid_price, days_from_last_coupon):
        return self.price() - spread_from_mid_price - days_from_last_coupon / DAYS_IN_YEAR_CONVENTION * \
               self.coupon(initial_commission)

    # Get offer as the fair decimal fraction price plus the commission
    def offer(self, initial_commission, spread_from_mid_price, days_from_last_coupon):
        return self.price() + spread_from_mid_price + days_from_last_coupon / DAYS_IN_YEAR_CONVENTION * \
               self.coupon(initial_commission)

    # Calculate the annualized coupon rate in decimals
    def coupon(self, commission):
        return (self.price() - commission) / sum([self.cpn_freq / (1 + self.rf)**(j * self.cpn_freq)
                                                  for j in range(1, int(self.term / self.cpn_freq) + 1)])

    # The function provides decimal fraction of the invested capital that the client receives at maturity
    # (without the coupons received)
    def execute(self):
        initial_fixing = self.initial_spot
        final_fixing = self.spot

        # If at maturity price grows above [TWICE the strike_price] = 200% minus the FV of commission collected =>
        # loses the maximum = 100% of notional (the rest of the growth is hedged by bought OTM call)
        if final_fixing / initial_fixing >= 2 - self.initial_commission * (1 + self.rf)**self.term:
            return 0
        # Else if at maturity price grows above the barrier_price => loses the difference between
        # final_fixing and strike = 100%
        elif final_fixing >= self.barrier_price:
            return 2 - final_fixing / initial_fixing
        # Otherwise, 100% of the notional is returned
        else:
            return 1

    # Final result => return (+ or -) as decimal fraction, including the coupons received and excluding the commissions
    def final_result(self):
        return (1 + self.option_premium() / self.spot - self.initial_commission) * \
               (1 + self.rf)**self.term + self.execute() - 1


class PerfectBondMatch:
    """
    Product that aims to match the bond's expected default probability and expected recovery rate to provide comparable
    statistical qualities for a given bond.

    PBM = sold BarrierPut with strike X > 100% to match recovery rate +
          bought EuropeanPut with strike X - 100% (hedges to have payoff >= -100%) +
          money market investment
    """
    def __init__(self, time_till_maturity: float, coupon_frequency: float, loss_probability: float,
                 recovery_rate: float, underlying_mean: float, underlying_volatility: float,
                 current_spot: float, initial_spot_fixing: float, risk_free_rate: float):
        self.term = time_till_maturity
        self.p = loss_probability
        self.rr = recovery_rate
        self.mean = underlying_mean
        self.sigma = underlying_volatility
        self.spot = current_spot
        self.initial_spot = initial_spot_fixing
        self.rf = risk_free_rate
        self.cpn_freq = coupon_frequency
        self.barrier_price = self.value_at_risk() * self.initial_spot
        self.strike_price = self.expected_shortfall() * self.initial_spot

    # Function allows to calculate the required barrier_percentage by defining it as a threshold that is allowed to be
    # breached only in loss_probability * 100% cases => VaR with "alpha" = loss_probability by definition
    def value_at_risk(self):
        return np.exp(self.mean * self.term - scipy.stats.norm.ppf(1 - self.p) * self.sigma * np.sqrt(self.term))

    # Function allows to calculate the required strike_percentage by defining it in such way that distance between
    # strike_percentage and expected final price % from the initial_spot_fixing will be equal to the
    # comparable bond's recovery_rate => Expected Shortfall (CVaR) with "alpha" = loss_probability by definition
    def expected_shortfall(self):
        expected_shortfall = np.exp(self.mean * self.term + self.sigma ** 2 * self.term / 2) * \
                             scipy.stats.norm.cdf(scipy.stats.norm.ppf(self.p) - self.sigma) / self.p

        return 1 + expected_shortfall - self.rr

    # Function calculates premium of the option (in price space) by simple Black-Scholes-Merton
    def option_premium(self):
        # Hedging put is needed only in case strike for BarrierPut is higher than 100% of initial spot
        if self.strike_price / self.initial_spot - 1 > 0:
            vanilla_put_to_hedge = EuropeanPut(self.term, self.sigma, self.strike_price / self.initial_spot - 1,
                                               self.spot, self.initial_spot, self.mean).option_premium()
        else:
            vanilla_put_to_hedge = 0

        return DownInBarrierPut(self.term, self.sigma, self.strike_price / self.initial_spot,
                                self.barrier_price / self.initial_spot, self.spot, self.initial_spot,
                                self.mean).option_premium() - vanilla_put_to_hedge

    # Function produces the fair price of the structured product as decimal fraction of the notional invested
    def price(self):
        return 1 / (1 + self.rf) ** self.term - self.option_premium() / self.initial_spot

    # Get bid as the fair decimal fraction price minus the commission
    def bid(self, initial_commission, spread_from_mid_price, days_from_last_coupon):
        return self.price() - spread_from_mid_price - days_from_last_coupon / DAYS_IN_YEAR_CONVENTION * \
               self.coupon(initial_commission)

    # Get offer as the fair decimal fraction price plus the commission
    def offer(self, initial_commission, spread_from_mid_price, days_from_last_coupon):
        return self.price() + spread_from_mid_price + days_from_last_coupon / DAYS_IN_YEAR_CONVENTION * \
               self.coupon(initial_commission)

    # Calculate the annualized coupon rate in decimals
    def coupon(self, commission):
        return (self.price() - commission) / sum([self.cpn_freq / (1 + self.rf)**(j * self.cpn_freq)
                                                  for j in range(1, int(self.term / self.cpn_freq) + 1)])

    # The function provides decimal fraction of the invested capital that the client receives at maturity
    # (without the coupons received)
    def execute(self):
        initial_fixing = self.initial_spot
        final_fixing = self.spot

        # If final_fixing falls below the barrier level at maturity => loses the difference
        # between strike and final_fixing
        if final_fixing <= self.barrier_price:
            return (final_fixing - self.strike_price) / initial_fixing + 1
        # Otherwise, 100% of the notional is returned
        else:
            return 1

    # Final result => return (+ or -) as decimal fraction, including the coupons received and excluding the commissions
    def final_result(self, commission_paid):
        return (1 + self.option_premium() / self.spot - commission_paid) * \
               (1 + self.rf)**self.term + self.execute() - 1


class InversePerfectBondMatch:
    """
    Product that aims to match the bond's expected default probability and expected recovery rate to provide comparable
    statistical qualities for a given bond, but have different implied market direction
    (hedges, if bonds and stocks are expected to be positively correlated).

    I-PBM = sold BarrierCall with strike X < 100% to match recovery rate +
            bought EuropeanCall with strike X - 100% (hedges to have payoff >= -100%) +
            money market investment
    """
    def __init__(self, time_till_maturity: float, coupon_frequency: float, loss_probability: float,
                 recovery_rate: float, underlying_mean: float, underlying_volatility: float,
                 current_spot: float, initial_spot_fixing: float, risk_free_rate: float, initial_commission: float):
        self.term = time_till_maturity
        self.p = loss_probability
        self.rr = recovery_rate
        self.mean = underlying_mean
        self.sigma = underlying_volatility
        self.spot = current_spot
        self.initial_spot = initial_spot_fixing
        self.rf = risk_free_rate
        self.cpn_freq = coupon_frequency
        self.barrier_price = self.value_at_risk() * self.initial_spot
        self.strike_price = self.expected_shortfall() * self.initial_spot
        self.initial_commission = initial_commission

    # Function allows to calculate the required barrier_percentage by defining it as a threshold that is allowed to be
    # breached only in loss_probability * 100% cases => VaR with "alpha" = loss_probability by definition
    def value_at_risk(self):
        # p transformation to account for bought otm call => classical tail loss probability should EXCLUDE the
        # probability, hedged by the bought OTM call
        p_transformed = self.p + 1 - scipy.stats.lognorm.cdf(2, self.sigma, loc=self.mean)

        return np.exp(self.mean * self.term - scipy.stats.norm.ppf(p_transformed) * self.sigma * np.sqrt(self.term))

    # Function allows to calculate the required strike_percentage by defining it in such way that distance between
    # strike_percentage and expected final price % from the initial_spot_fixing will be equal to the
    # comparable bond's recovery_rate => Expected Shortfall (CVaR) with "alpha" = loss_probability by definition
    def expected_shortfall(self):
        # p transformation to account for bought otm call => classical tail loss probability should EXCLUDE the
        # probability, hedged by the bought OTM call
        p_transformed = self.p + 1 - scipy.stats.lognorm.cdf(2, self.sigma, loc=self.mean)

        expected_shortfall = np.exp(-self.mean * self.term + self.sigma ** 2 * self.term / 2) * scipy.stats.norm.cdf(
            scipy.stats.norm.ppf(p_transformed) - self.sigma) / p_transformed

        return 1 - (expected_shortfall - self.rr)

    # Function calculates premium of the option (in price space) by simple Black-Scholes-Merton
    def option_premium(self):
        return UpInBarrierCall(self.term, self.sigma, self.strike_price / self.initial_spot,
                               self.barrier_price / self.initial_spot, self.spot, self.initial_spot,
                               self.mean).option_premium() - \
               EuropeanCall(self.term, self.sigma,
                            (1 + self.strike_price / self.initial_spot - self.initial_commission) *
                            (1 + self.rf)**self.term, self.spot, self.initial_spot, self.mean).option_premium()

    # Function produces the fair price of the structured product as decimal fraction of the notional invested
    def price(self):
        return 1 / (1 + self.rf) ** self.term - self.option_premium() / self.initial_spot

    # Get bid as the fair decimal fraction price minus the commission
    def bid(self, initial_commission, spread_from_mid_price, days_from_last_coupon):
        return self.price() - spread_from_mid_price - days_from_last_coupon / DAYS_IN_YEAR_CONVENTION * \
               self.coupon(initial_commission)

    # Get offer as the fair decimal fraction price plus the commission
    def offer(self, initial_commission, spread_from_mid_price, days_from_last_coupon):
        return self.price() + spread_from_mid_price + days_from_last_coupon / DAYS_IN_YEAR_CONVENTION * \
               self.coupon(initial_commission)

    # Calculate the annualized coupon rate in decimals
    def coupon(self, commission):
        return (self.price() - commission) / sum([self.cpn_freq / (1 + self.rf)**(j * self.cpn_freq)
                                                  for j in range(1, int(self.term / self.cpn_freq) + 1)])

    # The function provides decimal fraction of the invested capital that the client receives at maturity
    # (without the coupons received)
    def execute(self):
        initial_fixing = self.initial_spot
        final_fixing = self.spot

        # If at maturity price grows above [TWICE the strike_price] = 200% minus the FV of commission collected =>
        # loses the maximum = 100% of notional (the rest of the growth is hedged by bought OTM call)
        if final_fixing / initial_fixing >= \
                1 + self.strike_price / initial_fixing - self.initial_commission * (1 + self.rf)**self.term:
            return 0
        # Else if at maturity price grows above the barrier_price => loses the difference between
        # final_fixing and strike
        elif final_fixing >= self.barrier_price:
            return 1 - final_fixing / initial_fixing + self.strike_price / initial_fixing
        # Otherwise, 100% of the notional is returned
        else:
            return 1

    # Final result => return (+ or -) as decimal fraction, including the coupons received and excluding the commissions
    def final_result(self, commission_paid):
        return (1 + self.option_premium() / self.spot - commission_paid) * \
               (1 + self.rf)**self.term + self.execute() - 1
