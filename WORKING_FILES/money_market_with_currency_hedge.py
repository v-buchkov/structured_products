from src.structured_products.predictors.volatility import GJRGarch
from src.structured_products.results_analysis import calculate_ytm

from src.structured_products.pricer.options.vanilla_options import EuropeanCall

years_till_maturity = 2
coupon_rate_decimal = 0.034
coupon_frequency_years = 0.5

fx_spot = 61.44
base_ccy_rf_rate = 0.02

with open('underlying_data/usdrub.csv', 'r') as f:
    file_data = [line for line in f]
    past_prices = [float(p.split(',')[1].replace('"', '')) for p in file_data[1:]]
    past_prices.reverse()
    past_returns = [past_prices[i] / past_prices[i - 1] - 1 for i in range(1, len(past_prices))]
    past_returns = [r for r in past_returns if r != 0]

# volatility = np.sqrt(252) * np.sqrt(GJRGarch(1, 1, past_returns, [0] * len(past_returns)).long_term_variance())
volatility = 0.31

print(f'Volatility = {round(volatility * 100, 2)}% annualized')

call_premium = 0.13 * EuropeanCall(time_till_maturity=years_till_maturity, underlying_volatility=volatility,
                                   current_spot=fx_spot, initial_spot_fixing=fx_spot, risk_free_rate=base_ccy_rf_rate,
                                   strike_percentage=1).price()

print(f'Call premium = {round(call_premium * 100, 2)}%')

final_coupon_per_annum = calculate_ytm(time_till_maturity=years_till_maturity,
                                       purchase_price_decimal=(1 + call_premium),
                                       coupon_per_annum_decimal=coupon_rate_decimal,
                                       frequency_years=coupon_frequency_years, redemption_amount_decimal=1)

print(f'Final coupon = {round(final_coupon_per_annum, 2) * 100}% per annum')
