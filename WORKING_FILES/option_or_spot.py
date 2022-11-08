from src.structured_products.pricer.options.vanilla_options import EuropeanCall, EuropeanPut

expected_price = 80

spot_ref = 58.24
term = 1
vol = 0.3081
rf_rub = 0.075
rf_usd = 0.02

if expected_price - spot_ref > 0:
    option = EuropeanCall(time_till_maturity=term, current_spot=spot_ref, initial_spot_fixing=spot_ref,
                          risk_free_rate=rf_rub-rf_usd, strike_percentage=1, underlying_volatility=vol)
else:
    option = EuropeanPut(time_till_maturity=term, current_spot=spot_ref, initial_spot_fixing=spot_ref,
                         risk_free_rate=rf_rub-rf_usd, strike_percentage=1, underlying_volatility=vol)

option_pnl = option.delta() * (expected_price - spot_ref) + option.gamma() * (expected_price - spot_ref)**2 / 2 - \
             option.option_premium()
spot_pnl = option.delta() * (expected_price - spot_ref)

print(option_pnl, spot_pnl)

gamma_gain = option.gamma() * (expected_price - spot_ref)**2 / 2
gamma_cost = option.option_premium()

print(gamma_gain, gamma_cost)
