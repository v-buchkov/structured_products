"""
The file provides blueprints to create capital protected structured notes objects
----------------------------
The convention for all fractions is decimals
(e.g., if the price is 95% from notional, the ClassObject.price() will return 0.95)
"""
from src.structured_products.pricer.options.vanilla_options import EuropeanCall, EuropeanPut


class CapProtectionUpside:
    """
    CapProtection Note with Upside Participation = bought Vanilla EuropeanCall + money market investment

    Discounting rate (risk-free rate) is assumed to be equal to mean return of the underlying
    """
    def __init__(self, time_till_maturity: float, strike_percentage: float, participation_level: float,
                 underlying_mean: float, underlying_volatility: float, current_spot: float, initial_spot_fixing: float,
                 risk_free_rate: float):
        self.term = time_till_maturity
        self.mean = underlying_mean
        self.sigma = underlying_volatility
        self.spot = current_spot
        self.initial_spot = initial_spot_fixing
        self.rf = risk_free_rate
        self.strike_price = strike_percentage * self.initial_spot
        self.participation_level = participation_level

    # Function calculates premium of the option (in price space) by simple Black-Scholes-Merton
    def option_premium(self):
        return self.participation_level * \
               EuropeanCall(self.term, self.sigma, self.strike_price / self.initial_spot, self.spot, self.initial_spot,
                            self.mean).option_premium()

    # Function produces the fair price of the structured product as decimal fraction of the notional invested
    def price(self):
        return self.option_premium() / self.initial_spot + 1 / (1 + self.rf)**self.term

    # Get bid as the fair % price minus the commission
    def bid(self, spread_from_mid_price):
        return self.price() - spread_from_mid_price

    # Get offer as the fair % price plus the commission
    def offer(self, spread_from_mid_price):
        return self.price() + spread_from_mid_price

    # The function provides decimal fraction of the invested capital that the client receives at maturity
    def execute(self):
        initial_fixing = self.initial_spot
        final_fixing = self.spot

        # If grows above strike => pays the (final_fixing - strike_price)
        if final_fixing > self.strike_price:
            return 1 + self.participation_level * (final_fixing - self.strike_price) / initial_fixing
        # Otherwise - only 100% of capital invested is returned
        else:
            return 1

    # Final result => return (+ or -) as decimal fraction, excluding the commissions
    def final_result(self, commission_paid):
        return self.execute() - 1 - commission_paid * (1 + self.rf) ** self.term


class CapProtectionCappedUpside:
    """
    CapProtection Note with Capped Upside Participation =
    bought Vanilla EuropeanCall + money market investment - sold Vanilla EuropeanCall at strike = cap_percentage

    Discounting rate (risk-free rate) is assumed to be equal to mean return of the underlying
    """
    def __init__(self, time_till_maturity: float, strike_percentage: float, cap_percentage: float,
                 participation_level: float, underlying_mean: float, underlying_volatility: float,
                 current_spot: float, initial_spot_fixing: float, risk_free_rate: float):
        self.term = time_till_maturity
        self.mean = underlying_mean
        self.sigma = underlying_volatility
        self.spot = current_spot
        self.initial_spot = initial_spot_fixing
        self.rf = risk_free_rate
        self.strike_price = strike_percentage * self.initial_spot
        self.cap_price = cap_percentage * self.initial_spot
        self.participation_level = participation_level

    # Function calculates premium of the option (in price space) by simple Black-Scholes-Merton
    def option_premium(self):
        return self.participation_level * \
               EuropeanCall(self.term, self.sigma, self.strike_price / self.initial_spot,
                            self.spot, self.initial_spot, self.mean).option_premium() \
               - self.participation_level * \
               EuropeanCall(self.term, self.sigma, self.cap_price / self.initial_spot,
                            self.spot, self.initial_spot, self.mean).option_premium()

    # Function produces the fair price of the structured product as decimal fraction of the notional invested
    def price(self):
        return self.option_premium() / self.initial_spot + 1 / (1 + self.rf)**self.term

    # Get bid as the fair % price minus the commission
    def bid(self, spread_from_mid_price):
        return self.price() - spread_from_mid_price

    # Get offer as the fair % price plus the commission
    def offer(self, spread_from_mid_price):
        return self.price() + spread_from_mid_price

    # The function provides decimal fraction of the invested capital that the client receives at maturity
    def execute(self):
        initial_fixing = self.initial_spot
        final_fixing = self.spot

        # If grows above strike
        if final_fixing > self.strike_price:
            # If above the cap => (cap_price - strike_price) is maximum that can be paid
            if final_fixing < self.cap_price:
                return 1 + self.participation_level * (final_fixing - self.strike_price) / initial_fixing
            # If NOT above the cap => pays the (final_fixing - strike_price)
            else:
                return 1 + self.participation_level * (self.cap_price - self.strike_price) / initial_fixing
        # Otherwise - only 100% of capital invested is returned
        else:
            return 1

    # Final result => return (+ or -) as decimal fraction, excluding the commissions
    def final_result(self, commission_paid):
        return self.execute() - 1 - commission_paid * (1 + self.rf) ** self.term


class CapProtectionDownside:
    """
    CapProtection Note with Downside Participation = bought Vanilla EuropeanPut + money market investment

    Discounting rate (risk-free rate) is assumed to be equal to mean return of the underlying
    """
    def __init__(self, time_till_maturity: float, strike_percentage: float, participation_level: float,
                 underlying_mean: float, underlying_volatility: float, current_spot: float, initial_spot_fixing: float,
                 risk_free_rate: float):
        self.term = time_till_maturity
        self.mean = underlying_mean
        self.sigma = underlying_volatility
        self.spot = current_spot
        self.initial_spot = initial_spot_fixing
        self.rf = risk_free_rate
        self.strike_price = strike_percentage * self.initial_spot
        self.participation_level = participation_level

    # Function calculates premium of the option (in price space) by simple Black-Scholes-Merton
    def option_premium(self):
        return self.participation_level * \
               EuropeanPut(self.term, self.sigma, self.strike_price / self.initial_spot,
                           self.spot, self.initial_spot, self.mean).option_premium()

    # Function produces the fair price of the structured product as decimal fraction of the notional invested
    def price(self):
        return self.option_premium() / self.initial_spot + 1 / (1 + self.rf)**self.term

    # Get bid as the fair % price minus the commission
    def bid(self, spread_from_mid_price):
        return self.price() - spread_from_mid_price

    # Get offer as the fair % price plus the commission
    def offer(self, spread_from_mid_price):
        return self.price() + spread_from_mid_price

    # The function provides decimal fraction of the invested capital that the client receives at maturity
    def execute(self):
        initial_fixing = self.initial_spot
        final_fixing = self.spot

        # If falls below strike => pays the (strike_price - final_fixing)
        if final_fixing > self.strike_price:
            return 1 + self.participation_level * (final_fixing - self.strike_price) / initial_fixing
        # Otherwise - only 100% of capital invested is returned
        else:
            return 1

    # Final result => return (+ or -) as decimal fraction, excluding the commissions
    def final_result(self, commission_paid):
        return self.execute() - 1 - commission_paid * (1 + self.rf) ** self.term


class CapProtectionFlooredDownside:
    """
    CapProtection Note with Floored Downside Participation =
    bought Vanilla EuropeanPut + money market investment - sold Vanilla EuropeanPut at strike = floor_percentage

    Discounting rate (risk-free rate) is assumed to be equal to mean return of the underlying
    """
    def __init__(self, time_till_maturity: float, strike_percentage: float, floor_percentage: float,
                 participation_level: float, underlying_mean: float, underlying_volatility: float,
                 current_spot: float, initial_spot_fixing: float, risk_free_rate: float):
        self.term = time_till_maturity
        self.mean = underlying_mean
        self.sigma = underlying_volatility
        self.spot = current_spot
        self.initial_spot = initial_spot_fixing
        self.rf = risk_free_rate
        self.strike_price = strike_percentage * self.initial_spot
        self.floor_price = floor_percentage * self.initial_spot
        self.participation_level = participation_level

    # Function calculates premium of the option (in price space) by simple Black-Scholes-Merton
    def option_premium(self):
        return self.participation_level * \
               EuropeanPut(self.term, self.sigma, self.strike_price / self.initial_spot,
                           self.spot, self.initial_spot, self.mean).option_premium() \
               - self.participation_level * \
               EuropeanPut(self.term, self.sigma, self.floor_price / self.initial_spot,
                           self.spot, self.initial_spot, self.mean).option_premium()

    # Function produces the fair price of the structured product as decimal fraction of the notional invested
    def price(self):
        return self.option_premium() / self.initial_spot + 1 / (1 + self.rf)**self.term

    # Get bid as the fair % price minus the commission
    def bid(self, spread_from_mid_price):
        return self.price() - spread_from_mid_price

    # Get offer as the fair % price plus the commission
    def offer(self, spread_from_mid_price):
        return self.price() + spread_from_mid_price

    # The function provides decimal fraction of the invested capital that the client receives at maturity
    def execute(self):
        initial_fixing = self.initial_spot
        final_fixing = self.spot

        # If falls below strike
        if final_fixing < self.strike_price:
            # If NOT below the floor => pays the (strike_price - final_fixing)
            if final_fixing > self.floor_price:
                return 1 + self.participation_level * (self.strike_price - final_fixing) / initial_fixing
            # If below the floor => (strike_price - floor_price) is maximum that can be paid
            else:
                return 1 + self.participation_level * (self.strike_price - self.floor_price) / initial_fixing
        # Otherwise - only 100% of capital invested is returned
        else:
            return 1

    # Final result => return (+ or -) as decimal fraction, excluding the commissions
    def final_result(self, commission_paid):
        return self.execute() - 1 - commission_paid * (1 + self.rf) ** self.term
