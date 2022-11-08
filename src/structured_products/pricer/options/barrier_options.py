"""
The file provides blueprints to create barrier option objects
(derivatives that "kick-in" or "kick-out" at some price level).
----------------------------
The convention for all fractions is decimals
(e.g., if the price is 5% from notional, the ClassObject.price() will return 0.05)
"""
import scipy.stats
import numpy as np
from src.structured_products.pricer.options.vanilla_options import EuropeanCall, EuropeanPut


class UpInBarrierCall:
    """
    European call that "kicks-in", only if the price at maturity exceeds the barrier_price level
    """
    def __init__(self, time_till_maturity: float, underlying_volatility: float, strike_percentage: float,
                 barrier_percentage: float, current_spot: float, initial_spot_fixing: float, risk_free_rate: float):
        self.term = time_till_maturity
        self.sigma = underlying_volatility
        self.spot = current_spot
        self.initial_spot = initial_spot_fixing
        self.rf = risk_free_rate

        if strike_percentage > barrier_percentage:
            assert 'Strike should be below the barrier'
        else:
            self.strike_price = strike_percentage * self.initial_spot
            self.barrier_price = barrier_percentage * self.initial_spot

    # Function calculates premium of the option (in price space) by simple Black-Scholes-Merton
    def option_premium(self):

        # The "lambda" constant, responsible for discounting & normalization of the volatility
        l_const = (self.rf + self.sigma**2 / 2) / self.sigma**2
        y = np.log(self.barrier_price**2 / (self.spot * self.strike_price)) / \
            (self.sigma * np.sqrt(self.term)) + l_const * self.sigma * np.sqrt(self.term)
        y1 = np.log(self.barrier_price / self.spot) / (self.sigma * np.sqrt(self.term)) + \
             l_const * self.sigma * np.sqrt(self.term)
        x1 = np.log(self.spot / self.barrier_price) / (self.sigma * np.sqrt(self.term)) + \
             l_const * self.sigma * np.sqrt(self.term)

        ui_call_price = self.spot * scipy.stats.norm.cdf(x1) - \
                        self.strike_price * np.exp(-self.rf * self.term) * \
                        scipy.stats.norm.cdf(x1 - self.sigma * np.sqrt(self.term)) - \
                        self.spot * (self.barrier_price / self.spot) ** (2 * l_const) * (scipy.stats.norm.cdf(-y) -
                                                                                         scipy.stats.norm.cdf(-y1)) + \
                        self.strike_price * np.exp(-self.rf * self.term) * (self.barrier_price / self.spot) ** \
                        (2 * l_const - 2) * \
                        (scipy.stats.norm.cdf(-y + self.sigma * np.sqrt(self.term)) -
                         scipy.stats.norm.cdf(-y1 + self.sigma * np.sqrt(self.term)))

        return ui_call_price

    # Option price as decimal fraction of the current spot
    def price(self):
        return self.option_premium() / self.spot

    # Get bid as decimal fraction price minus the commission
    def bid(self, spread_from_mid_price):
        return self.price() - spread_from_mid_price

    # Get offer as decimal fraction price plus the commission
    def offer(self, spread_from_mid_price):
        return self.price() + spread_from_mid_price

    # Execute the option => get the realization of the payoff function (in price space)
    def execute(self):
        initial_fixing = self.initial_spot
        final_fixing = self.spot

        if final_fixing >= self.barrier_price:
            return final_fixing - initial_fixing
        else:
            return 0

    # Final result => payoff in decimal fraction, excluding the commissions and initial premium paid
    def final_result(self, commission_paid):
        return self.execute() / self.initial_spot - \
               (self.option_premium() / self.spot - commission_paid) * (1 + self.rf)**self.term


class UpOutBarrierCall:
    """
    European call that "kicks-out", if the price at maturity exceeds barrier_price level
    (product "lives", only if [strike_price <= price at maturity <= barrier_price]
    """
    def __init__(self, time_till_maturity: float, underlying_volatility: float, strike_percentage: float,
                 barrier_percentage: float, current_spot: float, initial_spot_fixing: float, risk_free_rate: float):
        self.term = time_till_maturity
        self.sigma = underlying_volatility
        self.spot = current_spot
        self.initial_spot = initial_spot_fixing
        self.rf = risk_free_rate

        self.strike_price = strike_percentage * self.initial_spot
        self.barrier_price = barrier_percentage * self.initial_spot

    # Function is defined to be dependent on other objects for premiums
    # This way allows to avoid some mistakes while derivation of the formula, allowing
    # to correct mistakes in other objects directly
    def option_premium(self):
        return EuropeanCall(self.term, self.sigma, self.strike_price, self.spot, self.initial_spot, self.rf).price() \
               - UpInBarrierCall(self.term, self.sigma, self.strike_price, self.barrier_price / self.initial_spot,
                                 self.spot, self.initial_spot, self.rf)

    # Option price as decimal fraction of the current spot
    def price(self):
        return self.option_premium() / self.spot

    # Get bid as decimal fraction price minus the commission
    def bid(self, spread_from_mid_price):
        return self.price() - spread_from_mid_price

    # Get offer as decimal fraction price plus the commission
    def offer(self, spread_from_mid_price):
        return self.price() + spread_from_mid_price

    # Execute the option => get the realization of the payoff function (in price space)
    def execute(self):
        initial_fixing = self.initial_spot
        final_fixing = self.spot

        if (final_fixing > initial_fixing) and (final_fixing <= self.barrier_price):
            return final_fixing - initial_fixing
        else:
            return 0

    # Final result => payoff in decimal fraction, excluding the commissions and initial premium paid
    def final_result(self, commission_paid):
        return self.execute() / self.initial_spot - \
               (self.option_premium() / self.spot - commission_paid) * (1 + self.rf)**self.term


class DownInBarrierPut:
    """
    European put that "kicks-in", only if the price at maturity falls below the barrier_price level
    """
    def __init__(self, time_till_maturity: float, underlying_volatility: float, strike_percentage: float,
                 barrier_percentage: float, current_spot: float, initial_spot_fixing: float, risk_free_rate: float):
        self.term = time_till_maturity
        self.sigma = underlying_volatility
        self.spot = current_spot
        self.initial_spot = initial_spot_fixing
        self.rf = risk_free_rate

        self.strike_price = strike_percentage * self.initial_spot
        self.barrier_price = barrier_percentage * self.initial_spot

    # Function calculates premium of the option (in price space) by simple Black-Scholes-Merton
    def option_premium(self):

        # The "lambda" constant, responsible for discounting & normalization of the volatility
        l_const = (self.rf + self.sigma ** 2 / 2) / self.sigma ** 2
        y = np.log(self.barrier_price ** 2 / (self. spot * self.strike_price)) \
            / (self.sigma * np.sqrt(self.term)) + l_const * self.sigma * np.sqrt(self.term)
        y1 = np.log(self.barrier_price / self.spot) / \
             (self.sigma * np.sqrt(self.term)) + l_const * self.sigma * np.sqrt(self.term)
        x1 = np.log(self.spot / self.barrier_price) / \
             (self.sigma * np.sqrt(self.term)) + l_const * self.sigma * np.sqrt(self.term)

        di_put_price = (-self.spot * scipy.stats.norm.cdf(-x1) +
                        self.strike_price * np.exp(-self.rf * self.term) *
                        scipy.stats.norm.cdf(-x1 + self.sigma * np.sqrt(self.term)) +
                        self.spot * (self.barrier_price / self.spot) ** (2 * l_const) *
                        (scipy.stats.norm.cdf(y) - scipy.stats.norm.cdf(y1)) -
                        self.strike_price * np.exp(-self.rf * self.term) *
                        (self.barrier_price / self.spot) ** (2 * l_const - 2) *
                        (scipy.stats.norm.cdf(y - self.sigma * np.sqrt(self.term)) -
                         scipy.stats.norm.cdf(y1 - self.sigma * np.sqrt(self.term))))

        return di_put_price

    # Option price as decimal fraction of the current spot
    def price(self):
        return self.option_premium() / self.spot

    # Get bid as decimal fraction price minus the commission
    def bid(self, spread_from_mid_price):
        return self.price() - spread_from_mid_price

    # Get offer as decimal fraction price plus the commission
    def offer(self, spread_from_mid_price):
        return self.price() + spread_from_mid_price

    # Execute the option => get the realization of the payoff function (in price space)
    def execute(self):
        initial_fixing = self.initial_spot
        final_fixing = self.spot

        if final_fixing <= self.barrier_price:
            return final_fixing - initial_fixing
        else:
            return 0

    # Final result => payoff in decimal fraction, excluding the commissions and initial premium paid
    def final_result(self, commission_paid):
        return self.execute() / self.initial_spot - \
               (self.option_premium() / self.spot - commission_paid) * (1 + self.rf)**self.term


class DownOutBarrierCall:
    """
    European put that "kicks-out", if the price at maturity falls below the barrier_price level
    (product "lives", only if [barrier_price <= price at maturity <= strike_price]
    """
    def __init__(self, time_till_maturity: float, underlying_volatility: float, strike_percentage: float,
                 barrier_percentage: float, current_spot: float, initial_spot_fixing: float, risk_free_rate: float):
        self.term = time_till_maturity
        self.sigma = underlying_volatility
        self.spot = current_spot
        self.initial_spot = initial_spot_fixing
        self.rf = risk_free_rate

        self.strike_price = strike_percentage * self.initial_spot
        self.barrier_price = barrier_percentage * self.initial_spot

    # Function is defined to be dependent on other objects for premiums
    # This way allows to avoid some mistakes while derivation of the formula, allowing
    # to correct mistakes in other objects directly
    def option_premium(self):
        return EuropeanPut(self.term, self.sigma, self.strike_price, self.spot, self.initial_spot, self.rf).price() \
               - DownInBarrierPut(self.term, self.sigma, self.strike_price, self.barrier_price / self.initial_spot,
                                  self.spot, self.initial_spot, self.rf)

    # Option price as decimal fraction of the current spot
    def price(self):
        return self.option_premium() / self.spot

    # Get bid as decimal fraction price minus the commission
    def bid(self, spread_from_mid_price):
        return self.price() - spread_from_mid_price

    # Get offer as decimal fraction price plus the commission
    def offer(self, spread_from_mid_price):
        return self.price() + spread_from_mid_price

    # Execute the option => get the realization of the payoff function (in price space)
    def execute(self):
        initial_fixing = self.initial_spot
        final_fixing = self.spot

        if (final_fixing < initial_fixing) and (final_fixing >= self.barrier_price):
            return initial_fixing - final_fixing
        else:
            return 0

    # Final result => payoff in decimal fraction, excluding the commissions and initial premium paid
    def final_result(self, commission_paid):
        return self.execute() / self.initial_spot - \
               (self.option_premium() / self.spot - commission_paid) * (1 + self.rf)**self.term
