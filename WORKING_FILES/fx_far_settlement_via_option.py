from src.structured_products.pricer.options.vanilla_options import EuropeanCall

rub_insignificant_amount = 50000

notional = 55000000
current_spot = 60.50
settlement_needed_days = 5

vol = 0.20

for delta_spot in range(50, 1000, 50):
    potential_strike = current_spot + delta_spot / 100

    option_premium_decimal = EuropeanCall(time_till_maturity=settlement_needed_days/365, underlying_volatility=vol,
                                          current_spot=current_spot, initial_spot_fixing=current_spot,
                                          risk_free_rate=0.08, strike_percentage=potential_strike/current_spot).price()

    if option_premium_decimal * notional * current_spot <= rub_insignificant_amount:
        print(f'Strike = {potential_strike}')
        break
