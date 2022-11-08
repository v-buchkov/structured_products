import datetime as dt

import numpy as np
import scipy.stats
from src.structured_products.pricer.fully_funded_notes.bonus_notes import CappedBonusNote
from src.structured_products.predictors.mean import ARMA
from src.structured_products.predictors.volatility import GJRGarch

confidence_level = 0.95

till_maturity_years = (dt.datetime(year=2022, month=9, day=6)-dt.datetime(year=2022, month=4, day=22)).days / 365
# mean_return_constant = ARMA(2, 2, past_returns).long_term_mean()

capm_mean = 0.02 + 1.06 * (0.07 + 0.03 - 0.02)
risk_free_rate = 0.02

with open('underlying_data/OGZD.L.csv', 'r') as f:
    file_data = [line for line in f]
    past_prices = [float(p.split(',')[1]) for p in file_data[1:] if p.split(',')[1] != 'null']
    past_returns = [past_prices[i] / past_prices[i - 1] - 1 for i in range(1, len(past_prices))]
    past_returns = [r for r in past_returns if r != 0]
    # print(past_returns)

arma_estimated = ARMA(2, 2, past_returns)

arma_constants = arma_estimated.coefficients
arma_forecasted = arma_estimated.forecasted_returns_list(arma_constants)
arma_long_term_mean = arma_estimated.long_term_mean()
arma_one_step_forecast = -(arma_constants[0] + arma_constants[1] * past_returns[-1] +
                           arma_constants[2] * past_returns[-2] +
                           arma_constants[3] * (past_returns[-1] - arma_forecasted[-1]) +
                           arma_constants[4] * (past_returns[-2] - arma_forecasted[-2]))
# print(mean_return_constant)
standard_deviation = np.sqrt(GJRGarch(1, 1, past_returns, arma_forecasted).long_term_variance())

for mean_return in [capm_mean, arma_long_term_mean, arma_one_step_forecast]:
    print('---')
    point_price = CappedBonusNote(time_till_maturity=till_maturity_years, underlying_mean=mean_return,
                                  underlying_volatility=standard_deviation, barrier_percentage=0.82, cap_percentage=1.5,
                                  participation_level=1.25, current_spot=0.581, initial_spot_fixing=6.862,
                                  risk_free_rate=risk_free_rate).price(initial_bonus_level=10)

    mean_right_tail = mean_return + standard_deviation * scipy.stats.norm.ppf((1 + confidence_level) / 2)

    price_right_tail = CappedBonusNote(time_till_maturity=till_maturity_years, underlying_mean=mean_right_tail,
                                       underlying_volatility=standard_deviation, barrier_percentage=0.82,
                                       cap_percentage=1.5, participation_level=1.25, current_spot=0.581,
                                       initial_spot_fixing=6.862,
                                       risk_free_rate=risk_free_rate).price(initial_bonus_level=10)

    print(f'Price = {round(point_price * 100, 2)}%')
    print(f'{int(confidence_level * 100)}% Confidence Interval Right Tail = {round(price_right_tail * 100, 2)}%')
