"""
The file provides blueprints to create bonus notes objects
----------------------------
The convention for all fractions is decimals
(e.g., if the price is 95% from notional, the ClassObject.price() will return 0.95)
"""
from src.structured_products.pricer.options.vanilla_options import EuropeanCall, EuropeanPut
from src.structured_products.pricer.options.barrier_options import UpInBarrierCall, DownInBarrierPut


class UncappedBonusNote:
    """
    Bonus Note (Without Cap) = Sold Barrier Put + Bought Call at strike = (1 + X),
                               such that (Put Premium - Call Premium) = X

    Discounting rate (risk-free rate) is assumed to be equal to mean return of the underlying
    """
    def __init__(self, time_till_maturity: float, coupon_frequency: float, barrier_decimal: float,
                 participation_level: float, underlying_mean: float, underlying_volatility: float, current_spot: float,
                 initial_spot_fixing: float, risk_free_rate: float):
        self.term = time_till_maturity
        self.mean = underlying_mean
        self.sigma = underlying_volatility
        self.spot = current_spot
        self.initial_spot = initial_spot_fixing
        self.rf = risk_free_rate
        self.cpn_freq = coupon_frequency
        self.barrier_price = barrier_decimal * self.initial_spot
        self.participation_level = participation_level

    # Function provides bonus level as decimal fraction from notional invested
    # (fixed amount that will be paid out, in case the final return is between barrier and (1 + bonus_level))
    def bonus_level(self):
        # Collect the premium from sold barrier put and get the FV at the moment of maturity
        sold_put_premium = DownInBarrierPut(self.term, self.sigma, 1, self.barrier_price / self.initial_spot,
                                            self.spot, self.initial_spot,
                                            self.mean).option_premium() * (1 + self.rf)**self.term \
                           / self.initial_spot

        # Find the solution for X = bonus level by iteration
        bonus_solution = None
        for bonus in range(5, 100, 5):
            # FV of the premium for the call with strike 1 + X
            bought_call_premium = self.participation_level * \
                                  EuropeanCall(self.term, self.sigma, (1 + bonus / 100), self.spot, self.initial_spot,
                                               self.mean).option_premium() * (1 + self.rf) ** self.term
            # If the difference between sold put premium and bought call premium (amount left for paying out the bonus)
            # exceeds the given bonus => the solution was found
            if sold_put_premium - bought_call_premium >= bonus / 100:
                bonus_solution = bonus
                break

        return bonus_solution

    # Function produces the fair price of the structured product as decimal fraction of the notional invested
    def price(self, initial_bonus_level):
        # Price is the difference between option premiums plus the money market investment
        return (-DownInBarrierPut(self.term, self.sigma, 1, self.barrier_price / self.initial_spot, self.spot,
                                  self.initial_spot, self.mean).option_premium() +
                self.participation_level * EuropeanCall(self.term, self.sigma, (1 + initial_bonus_level / 100),
                                                        self.spot, self.initial_spot, self.mean).option_premium()) / \
               self.initial_spot + 1 / (1 + self.rf)**self.term

    # Get bid as the fair % price minus the commission
    def bid(self, initial_bonus_level, spread_from_mid_price):
        return self.price(initial_bonus_level) - spread_from_mid_price

    # Get offer as the fair % price plus the commission
    def offer(self, initial_bonus_level, spread_from_mid_price):
        return self.price(initial_bonus_level) + spread_from_mid_price

    # The function provides decimal fraction of the invested capital that the client receives at maturity
    # (without the coupons received)
    def execute(self, initial_bonus_level):
        initial_fixing = self.initial_spot
        final_fixing = self.spot
        bonus = initial_bonus_level

        # If final_fixing is between the barrier_price and bonus = participation strike => only bonus is paid out
        if (final_fixing >= self.barrier_price) and (final_fixing <= (1 + bonus) * initial_fixing):
            return 1 + bonus
        # If above bonus, amount of upside change is paid out (in accord with participation level)
        elif final_fixing > (1 + bonus) * initial_fixing:
            return self.participation_level * final_fixing / initial_fixing
        # Otherwise, loss from initial level - Down&In put loss
        else:
            return final_fixing / initial_fixing

    # Final result => payoff in %, excluding the commissions and initial premium paid
    def final_result(self, initial_bonus_level, commission_paid):
        return (1 - commission_paid) * (1 + self.rf)**self.term + self.execute(initial_bonus_level) - 1


class CappedBonusNote:
    """
    Bonus Note (With Cap) = Sold Barrier Put + Bought Call at strike = (1 + X) + Sold Call with strike = (1 + Cap),
                            such that (Barrier Put Premium - Bought Call Premium + Sold OTM Call Premium) = X

    Discounting rate (risk-free rate) is assumed to be equal to mean return of the underlying
    """
    def __init__(self, time_till_maturity: float, barrier_percentage: float, cap_percentage: float,
                 participation_level: float, underlying_mean: float, underlying_volatility: float, current_spot: float,
                 initial_spot_fixing: float, risk_free_rate: float):
        self.term = time_till_maturity
        self.mean = underlying_mean
        self.sigma = underlying_volatility
        self.spot = current_spot
        self.initial_spot = initial_spot_fixing
        self.rf = risk_free_rate
        self.barrier_price = barrier_percentage * self.initial_spot
        self.cap_price = cap_percentage * self.initial_spot
        self.participation_level = participation_level

    # Function provides bonus level as decimal fraction from notional invested
    # (fixed amount that will be paid out, in case the final return is between barrier and (1 + bonus_level))
    def bonus_level(self):
        # Collect the premium from sold barrier put and get the FV at the moment of maturity
        sold_put_premium = DownInBarrierPut(self.term, self.sigma, 1, self.barrier_price / self.initial_spot, self.spot,
                                            self.initial_spot, self.mean).option_premium() * (
                                       1 + self.rf) ** self.term / self.initial_spot

        # Collect the premium from sold OTM call and get the FV at the moment of maturity
        sold_otm_call_premium = self.participation_level * EuropeanCall(self.term, self.sigma,
                                                                        (1 + self.cap_price / self.initial_spot),
                                                                        self.spot, self.initial_spot,
                                                                        self.mean).option_premium() * (
                                       1 + self.rf) ** self.term / self.initial_spot

        # Find the solution for X = bonus level by iteration
        bonus_solution = None
        for bonus in range(5, 100, 5):
            # FV of the premium for the call with strike 1 + X
            bought_call_premium = self.participation_level * \
                                  EuropeanCall(self.term, self.sigma, (1 + bonus / 100), self.spot, self.initial_spot,
                                               self.mean).option_premium() * (1 + self.rf) ** self.term
            # If the difference between sold put premium and bought call premium (amount left for paying out the bonus)
            # exceeds the given bonus => the solution was found
            if sold_put_premium - bought_call_premium + sold_otm_call_premium >= bonus / 100:
                bonus_solution = bonus
                break

        return bonus_solution

    # Function produces the fair price of the structured product as decimal fraction of the notional invested
    def price(self, initial_bonus_level):
        # Price is the difference between option premiums plus the money market investment
        return (-DownInBarrierPut(self.term, self.sigma, 1, self.barrier_price / self.initial_spot, self.spot,
                                  self.initial_spot, self.mean).option_premium() -
                self.participation_level *
                EuropeanCall(self.term, self.sigma, self.cap_price / self.initial_spot, self.spot, self.initial_spot,
                             self.mean).option_premium() +
                self.participation_level *
                EuropeanCall(self.term, self.sigma, (1 + initial_bonus_level / 100), self.spot, self.initial_spot,
                             self.mean).option_premium()) / self.initial_spot + 1 / (1 + self.rf) ** self.term

    # Get bid as the fair % price minus the commission
    def bid(self, initial_bonus_level, spread_from_mid_price):
        return self.price(initial_bonus_level) - spread_from_mid_price

    # Get offer as the fair % price plus the commission
    def offer(self, initial_bonus_level, spread_from_mid_price):
        return self.price(initial_bonus_level) + spread_from_mid_price

    # The function provides decimal fraction of the invested capital that the client receives at maturity
    # (without the coupons received)
    def execute(self, initial_bonus_level):
        initial_fixing = self.initial_spot
        final_fixing = self.spot
        bonus = initial_bonus_level

        # If final_fixing is above the cap_price => only cap_percentage will be paid out
        # (in accord with participation level)
        if final_fixing >= self.cap_price:
            return 1 + self.participation_level * self.cap_price / self.initial_spot
        # If final_fixing is between the barrier_price and bonus = participation strike => only bonus is paid out
        elif (final_fixing >= self.barrier_price) and (final_fixing <= (1 + bonus) * initial_fixing):
            return 1 + bonus
        # If above bonus, amount of upside change is paid out (in accord with participation level)
        elif final_fixing > (1 + bonus) * initial_fixing:
            return self.participation_level * final_fixing / initial_fixing
        # Otherwise, loss from initial level - Down&In put loss
        else:
            return final_fixing / initial_fixing

    # Final result => payoff in %, excluding the commissions and initial premium paid
    def final_result(self, initial_bonus_level, commission_paid):
        return (1 - commission_paid) * (1 + self.rf) ** self.term + self.execute(initial_bonus_level) - 1


class UnflooredDownsideBonusNote:
    """
    Downside Bonus Note (Without Floor) = Sold Barrier Call + Bought Vanilla Call (to hedge return >= -100%) +
                                          Bought Put at strike = (1 - X),
                                          such that
                                          (Barrier Call Premium - Bought Call Premium - Bought Put Premium) = X

    Discounting rate (risk-free rate) is assumed to be equal to mean return of the underlying
    """
    def __init__(self, time_till_maturity: float, coupon_frequency: float, barrier_decimal: float,
                 participation_level: float, underlying_mean: float, underlying_volatility: float, current_spot: float,
                 initial_spot_fixing: float, risk_free_rate: float, initial_commission: float):
        self.term = time_till_maturity
        self.mean = underlying_mean
        self.sigma = underlying_volatility
        self.spot = current_spot
        self.initial_spot = initial_spot_fixing
        self.rf = risk_free_rate
        self.cpn_freq = coupon_frequency
        self.barrier_price = barrier_decimal * self.initial_spot
        self.initial_commission = initial_commission
        self.participation_level = participation_level

    # Function provides bonus level as decimal fraction from notional invested
    # (fixed amount that will be paid out, in case the final return is between barrier and (1 + bonus_level))
    def bonus_level(self):
        # Collect the premium from sold barrier call (minus hedge to keep return >= -100%)
        # and get the FV at the moment of maturity
        sold_call_premium = (UpInBarrierCall(self.term, self.sigma, 1, self.barrier_price / self.initial_spot,
                                             self.spot, self.initial_spot, self.mean).option_premium() -
                             EuropeanCall(self.term, self.sigma, (2 - self.initial_commission) *
                                          (1 + self.rf)**self.term, self.spot, self.initial_spot,
                                          self.mean).option_premium()) * (1 + self.rf)**self.term / self.initial_spot

        # Find the solution for X = bonus level by iteration
        bonus_solution = None
        for bonus in range(5, 100, 5):
            # FV of the premium for the call with strike 1 + X
            bought_put_premium = self.participation_level * \
                                 EuropeanPut(self.term, self.sigma, (1 - bonus / 100), self.spot, self.initial_spot,
                                             self.mean).option_premium() * (1 + self.rf) ** self.term
            # If the difference between sold put premium and bought call premium (amount left for paying out the bonus)
            # exceeds the given bonus => the solution was found
            if sold_call_premium - bought_put_premium >= bonus / 100:
                bonus_solution = bonus
                break

        return bonus_solution

    # Function produces the fair price of the structured product as decimal fraction of the notional invested
    def price(self, initial_bonus_level):
        # Price is the difference between option premiums plus the money market investment
        return (-UpInBarrierCall(self.term, self.sigma, 1, self.barrier_price / self.initial_spot, self.spot,
                                 self.initial_spot, self.mean).option_premium() +
                EuropeanCall(self.term, self.sigma, (2 - self.initial_commission) * (1 + self.rf)**self.term, self.spot,
                             self.initial_spot, self.mean).option_premium() +
                self.participation_level * EuropeanPut(self.term, self.sigma, (1 - initial_bonus_level / 100),
                                                       self.spot, self.initial_spot,
                                                       self.mean).option_premium()) / self.initial_spot + \
               1 / (1 + self.rf)**self.term

    # Get bid as the fair % price minus the commission
    def bid(self, initial_bonus_level, spread_from_mid_price):
        return self.price(initial_bonus_level) - spread_from_mid_price

    # Get offer as the fair % price plus the commission
    def offer(self, initial_bonus_level, spread_from_mid_price):
        return self.price(initial_bonus_level) + spread_from_mid_price

    # The function provides decimal fraction of the invested capital that the client receives at maturity
    # (without the coupons received)
    def execute(self, initial_bonus_level):
        initial_fixing = self.initial_spot
        final_fixing = self.spot
        bonus = initial_bonus_level

        price_change = final_fixing / initial_fixing

        # If the price is above max loss amount (with accord to FV of charged commission => a bit lower than 100%)
        if price_change >= (2 - self.initial_commission) * (1 + self.rf)**self.term:
            return (self.initial_commission - 1) * (1 + self.rf)**self.term
        # Else if final_fixing is between the barrier_price and bonus = participation strike => only bonus is paid out
        elif (1 - bonus) * initial_fixing <= price_change <= self.barrier_price:
            return 1 + bonus
        # If above bonus, amount of downside change is paid out (in accord with participation level)
        elif price_change <= (1 - bonus) * initial_fixing:
            return 1 + self.participation_level * (1 - price_change)
        # Otherwise, loss from initial level - Up&In loss
        else:
            return 2 - price_change

    # Final result => payoff in %, excluding the commissions and initial premium paid
    def final_result(self, initial_bonus_level, commission_paid):
        return (1 - commission_paid) * (1 + self.rf)**self.term + self.execute(initial_bonus_level) - 1


class FlooredDownsideBonusNote:
    """
    Downside Bonus Note (With Floor) = Sold Barrier Call + Bought Vanilla Call (to hedge return >= -100%) +
                                       Bought Put at strike = (1 - X) + Sold Vanilla Put at strike = (1 - Floor),
                                       such that
                                       (Barrier Call Premium - Bought Call Premium - Bought Put Premium +
                                       Sold OTM Put Premium) = X

    Discounting rate (risk-free rate) is assumed to be equal to mean return of the underlying
    """
    def __init__(self, time_till_maturity: float, barrier_percentage: float, floor_percentage: float,
                 participation_level: float, underlying_mean: float, underlying_volatility: float, current_spot: float,
                 initial_spot_fixing: float, risk_free_rate: float, initial_commission: float):
        self.term = time_till_maturity
        self.mean = underlying_mean
        self.sigma = underlying_volatility
        self.spot = current_spot
        self.initial_spot = initial_spot_fixing
        self.rf = risk_free_rate
        self.barrier_price = barrier_percentage * self.initial_spot
        self.floor_price = floor_percentage * self.initial_spot
        self.initial_commission = initial_commission
        self.participation_level = participation_level

    # Function provides bonus level as decimal fraction from notional invested
    # (fixed amount that will be paid out, in case the final return is between barrier and (1 + bonus_level))
    def bonus_level(self):
        # Collect the premium from sold barrier call (minus hedge to keep return >= -100%)
        # and get the FV at the moment of maturity
        sold_call_premium = (UpInBarrierCall(self.term, self.sigma, 1, self.barrier_price / self.initial_spot,
                                             self.spot, self.initial_spot, self.mean).option_premium() -
                             EuropeanCall(self.term, self.sigma, (2 - self.initial_commission) *
                                          (1 + self.rf) ** self.term, self.spot, self.initial_spot,
                                          self.mean).option_premium()) * (1 + self.rf) ** self.term / self.initial_spot

        # Collect the premium from sold OTM put and get the FV at the moment of maturity
        sold_otm_put_premium = self.participation_level * EuropeanPut(self.term, self.sigma,
                                                                      (1 - self.floor_price / self.initial_spot),
                                                                      self.spot, self.initial_spot,
                                                                      self.mean).option_premium() * (
                                1 + self.rf) ** self.term / self.initial_spot

        # Find the solution for X = bonus level by iteration
        bonus_solution = None
        for bonus in range(5, 100, 5):
            # FV of the premium for the call with strike 1 + X
            bought_put_premium = self.participation_level * \
                                 EuropeanPut(self.term, self.sigma, (1 - bonus / 100), self.spot, self.initial_spot,
                                             self.mean).option_premium() * (1 + self.rf) ** self.term
            # If the difference between sold put premium and bought call premium (amount left for paying out the bonus)
            # exceeds the given bonus => the solution was found
            if sold_call_premium - bought_put_premium + sold_otm_put_premium >= bonus / 100:
                bonus_solution = bonus
                break

        return bonus_solution

    # Function produces the fair price of the structured product as decimal fraction of the notional invested
    def price(self, initial_bonus_level):
        # Price is the difference between option premiums plus the money market investment
        return (-UpInBarrierCall(self.term, self.sigma, 1, self.barrier_price / self.initial_spot, self.spot,
                                 self.initial_spot, self.mean).option_premium() +
                EuropeanCall(self.term, self.sigma, (2 - self.initial_commission) * (1 + self.rf) ** self.term,
                             self.spot, self.initial_spot, self.mean).option_premium() +
                self.participation_level * EuropeanPut(self.term, self.sigma, (1 - initial_bonus_level / 100),
                                                       self.spot, self.initial_spot, self.mean).option_premium() -
                self.participation_level * EuropeanPut(self.term, self.sigma,
                                                       (1 - self.floor_price / self.initial_spot),
                                                       self.spot, self.initial_spot, self.mean).option_premium()
                ) / self.initial_spot + \
               1 / (1 + self.rf) ** self.term

    # Get bid as the fair % price minus the commission
    def bid(self, initial_bonus_level, spread_from_mid_price):
        return self.price(initial_bonus_level) - spread_from_mid_price

    # Get offer as the fair % price plus the commission
    def offer(self, initial_bonus_level, spread_from_mid_price):
        return self.price(initial_bonus_level) + spread_from_mid_price

    # The function provides decimal fraction of the invested capital that the client receives at maturity
    # (without the coupons received)
    def execute(self, initial_bonus_level):
        initial_fixing = self.initial_spot
        final_fixing = self.spot
        bonus = initial_bonus_level

        price_change = final_fixing / initial_fixing

        # If the price is above max loss amount (with accord to FV of charged commission => a bit lower than 100%)
        if price_change >= (2 - self.initial_commission) * (1 + self.rf) ** self.term:
            return (self.initial_commission - 1) * (1 + self.rf) ** self.term
        # If final_fixing is below the floor_price => only floor_percentage will be paid out
        # (in accord with participation level)
        elif final_fixing <= self.floor_price:
            return 1 + self.participation_level * (1 - self.floor_price / self.initial_spot)
        # Else if final_fixing is between the barrier_price and bonus = participation strike => only bonus is paid out
        elif (1 - bonus) * initial_fixing <= price_change <= self.barrier_price:
            return 1 + bonus
        # If above bonus, amount of downside change is paid out (in accord with participation level)
        elif price_change <= (1 - bonus) * initial_fixing:
            return 1 + self.participation_level * (1 - price_change)
        # Otherwise, loss from initial level - Up&In loss
        else:
            return 2 - price_change

    # Final result => payoff in %, excluding the commissions and initial premium paid
    def final_result(self, initial_bonus_level, commission_paid):
        return (1 - commission_paid) * (1 + self.rf) ** self.term + self.execute(initial_bonus_level) - 1
