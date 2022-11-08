from src.structured_products.pricer.options.vanilla_options import EuropeanCall

spot_ref = 58.24
term = 1
distribution = [0.0, 0.3081]
rf_rub = 0.075
rf_usd = -0.005
strike_decimal = 1
smile_coef = 1.1

mrc = rf_rub * 0.0475 * 0.75

have_from_funding = 1 - 1 / (1 + rf_rub - mrc) ** term

call_premium_percent = EuropeanCall(term, distribution, strike_decimal, spot_ref, spot_ref, rf_rub - rf_usd).price()

print('Participation = {:.2f}%'.format(have_from_funding / call_premium_percent * 100))

for strike in range(2, 100):
    strike_percentage = strike / 100
    shifted_option_premium = EuropeanCall(term, [distribution[0], distribution[1] * smile_coef],
                                          1 + strike_percentage, spot_ref, spot_ref, rf_rub - rf_usd).price()

    # print(call_premium_percent - shifted_option_premium, have_from_funding)
    if call_premium_percent - shifted_option_premium > have_from_funding:
        print('Cap level = {:.2f}%'.format(strike - 1))
        break

for strike in range(2, 100):
    strike_percentage = strike / 100
    shifted_option_premium = EuropeanCall(term, [distribution[0], distribution[1] * smile_coef],
                                          1 + strike_percentage, spot_ref, spot_ref, rf_rub - rf_usd).price()
    if shifted_option_premium < have_from_funding:
        print('Strike for 100% particip = {:.2f}%'.format(strike))
        break
