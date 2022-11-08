# Ctrl+Shift+Fn+F10

# This file is designed for calculating the statistical parameters of structured products by Monte Carlo method
# The distribution is assumed to be Geometric Brownian Motion with historical volatility and constatnt mean return

import numpy as np
import openpyxl
from tqdm import tqdm
import random
import os
from data.default_table import default_table
from datetime import datetime
import multiprocessing as multi
import time as measure_time

s = measure_time.time()

# MC conditions
# drift is assumed at the most recent dividend yield. Can take (!) analyst recommendation level or AM prediction
# input [True] to take historical mean
mean_dict = {'pd': [0.04, 0.06], 'copper': [0.03], 'esg': [True], 'mrna': [0.0], 'nvax': [0.0],
             'robotics': [0.054], 'sxpp': [True], 'wm': [0.016, 0.007, 0.015], 'ls': [0.05, 0.05],
             'rbi': [0.05, 0.05], 'pimco': [0.022], 'm&g': [True],
             'ltq1': [0.025, 0.01, 0.008, 0.013, 0.009, 0.02], 'gazprom': [0.1],
             'lux_clothes': [0.013, 0.02, 0.006], 'lux_brands': [0.013, 0.007, 0.02], 'lux': [0.015, 0.02, 0.00],
             'space': [0.004, 0.000], 'networks': [0.027, 0.00, 0.00],
             'lux_multi': [0.028, 0.032, 0.023, 0.02, 0.036, 0.077],
             'games': [0.04094, 0.04244, 0.04175], 'cybersec': [0.04178, 0.0377, 0.03767], 'nuclear_cpn': [0.0492],
             'nuclear_kg': [0.0492, 0.0466, 0.0496, 0.0502], 'clear_enr': [0.05, 0.0859, 0.0742],
             'fast_food': [0.048, 0.0525, 0.0577], 'lux_indiv': [0.028, 0.032, 0.0579, 0.02, 0.036, 0.077],
             'rus_6m': [0.15, 0.12, 0.10], 'kuzin': [0.0, 0.0076, 0.0, 0.0004, 0.032, 0.0],
             'best_div': [0.0489, 0.0533, 0.0377], 'low_debt': [0.024, 0.0338, 0.027],
             'cmdty_multi': [0.0519, 0.0555, 0.0586, 0.0647, 0.0553, 0.0432], 'spx': [0.015], 'yandex': [0.0],
             'brent': [0.04]}

paths = 10000
multi_thread = 7

# SP conditions, cpn p.a.
underlying = 'pd'
years = 2
barrier = 65
# autocall in years (e.g., 0.25 for quarterly, 0.083 for monthly)
autocall = 0.5
coupon = 6
gearing = 0
memory = 0
tail_event = 50
worst = 1

autocall_start = 1
geared_put = False
autocall_level = 100
# Set in % from spot level (e.g., for +30% set cap = 130)
cap = 1000
bonus = 0
digital = 0
digital_t = 0
cpn_barrier = barrier
# Set in % from spot level (e.g., for +0% set strike = 100)
participation_strike = 100
# set True for most recent value - 'XX.XX.XXXX' otherwise
fixing = True
issuer_rating = 'A-'
recovery = 30

# ------------------------------------------------------------------------------------------
# Create a new folder in output, if one does not exist already
try:
    os.mkdir('output/MC_{}'.format(underlying))
except FileExistsError:
    pass

t = int(252*years)

if years >= 1:
    default_p = default_table[issuer_rating][years-1]/100
else:
    default_p = default_table[issuer_rating][0]/100

n_stocks = len(os.listdir(path='../data/{}'.format(underlying)))

# read data => generate list [[stock1_value1, stock1_value2, ...], [stock2_value1, stock2_value2, ...], ...]
stocks = []
start_values = []
st_with_dates = []
skip = 7
hp = {}
for k in range(1, n_stocks+1):
    with_dates = {}
    ws = (openpyxl.load_workbook('../data/{}/{}.xlsx'.format(underlying, k))).active.values
    list_ = [item for item in ws]
    # record the start values
    start_values.append(float(list_[skip+1][1]))
    for i in range(skip+2, len(list_)):
        if list_[i][0] is None:
            break
        else:
            try:
                with_dates[list_[i-1][0].strftime("%d.%m.%Y")] = [float(list_[i-1][1]), i - 1]
                if k == 1:
                    if i == 1:
                        sim_start = list_[i - 1][0].strftime("%d.%m.%Y")
                    hp[list_[i - 1][0].strftime("%d.%m.%Y")] = [
                        np.log(float(list_[i - 1][1])) - np.log(float(list_[i][1]))]
                else:
                    hp[list_[i-1][0].strftime("%d.%m.%Y")].append(
                        np.log(float(list_[i-1][1]))-np.log(float(list_[i][1])))
            except KeyError:
                with_dates[list_[i-1][0].strftime("%d.%m.%Y")] = [0, i-1]
                if k == 1:
                    hp[list_[i - 1][0].strftime("%d.%m.%Y")] = [0]

    st_with_dates.append(with_dates)

for i in range(n_stocks):
    stocks.append([])

for item in hp.values():
    if len(item) == n_stocks:
        for i in range(len(item)):
            stocks[i].append(item[i])

if type(fixing) == bool and (not fixing):
    fixing_values = np.divide(np.array(start_values),  np.array([item[fixing][0] for item in st_with_dates]))
    t -= (datetime.strptime(sim_start, '%d.%m.%Y') - datetime.strptime(fixing, '%d.%m.%Y')).days
else:
    fixing_values = np.array([1 for i in range(n_stocks)])

# reduce dataset to the shortest lifetime of all stocks
stocks = [item[:min([len(it) for it in stocks])] for item in stocks]

for item in stocks:
    item.reverse()

# calculate historical mean return
if type(mean_dict[underlying][0]) == bool and mean_dict[underlying][0]:
    mean_returns = []
    for stock in stocks:
        mean_returns.append(np.mean(stock)*252)
else:
    mean_returns = mean_dict[underlying]

# calculate var-covar matrix
var_covar = 252 * np.cov(stocks)

print(var_covar)

# output = [[stock1_path1, stock2_path1, ..., stock6_path1], [stock1_path2, stock2_path2, ..., stock6_path2], ...]
time = np.linspace(0, t / 252, t)
d_time = time[1] - time[0]
cholesky = np.linalg.cholesky(var_covar)

# determine final fixing values
drift = np.exp(np.array([(mean_returns[j] - 0.5 * var_covar[j][j]) for j in range(n_stocks)]) * d_time)
bar_levels = barrier / (100 * fixing_values) - 1


class RestrList:

    def __init__(self, out):
        self.out = np.array([ret - 1 for ret in out]) * fixing_values

    def ac(self, term):
        self.out *= drift**term * 100
        return self.out

    def final(self):
        self.out *= drift**t * 100
        self.out.sort()
        return self.out[:worst]


def randomness(length):
    diffusion = np.exp([np.matmul(cholesky, np.random.normal(0, 1, size=n_stocks)) * np.sqrt(d_time)
                        for m in range(length)])

    return RestrList(np.prod(np.vstack(diffusion), axis=0))


# determine autocall levels
if autocall == 0:
    autocall = years


def monte_carlo(path_length):

    np.random.seed((os.getpid() * int(measure_time.time())) % 123456789)

    # generate random points of the issuer's default
    defaulted = [random.randrange(0, int(path_length/multi_thread * default_p))
                 for note in range(int(path_length/multi_thread * default_p))]

    for p in tqdm(range(int(path_length/multi_thread))):

        if p in defaulted:
            pass

        else:

            iterator = 0

            # print('---')

            while np.less(iterator, int(years * 252)):

                iterator += int(252 * autocall)

                if (iterator > 0) and (iterator < int(252 * years)):
                    restr = randomness(iterator).ac(term=iterator)

                    if all(np.greater(restr, autocall_level - 100)):
                        note_returns[5].append(coupon * (iterator / 252))
                        # dur += iterator / 252
                        break

                else:
                    restr = randomness(iterator).final()

                    if type(geared_put) == bool and geared_put:
                        g_coef = (barrier * restr[0]) / (100 + restr[0] - barrier) / 100
                    else:
                        g_coef = 1

                    if all(np.less(restr, barrier - 100)):
                        if all(it < tail_event - coupon * int(t / 252) - 100 for it in restr):
                            note_returns[0].append(restr[0] / g_coef + coupon * int(t / 252))
                        else:
                            note_returns[1].append(restr[0] / g_coef + coupon * int(t / 252))
                    elif gearing > 0 and all(it > participation_strike - 100 for it in restr):
                        if all(it > cap - 100 for it in restr):
                            note_returns[4].append((cap - 100) * gearing + coupon * int(t / 252) + bonus)
                        else:
                            note_returns[3].append((restr[0] - participation_strike - 100) * gearing + coupon
                                                   * int(t / 252) + bonus)
                    else:
                        note_returns[2].append(coupon * int(t / 252) + bonus)

    # note_returns = sum(finals)
    #
    # print(note_returns)
    #
    # note_mean = float(np.mean(note_returns))
    # note_std = float(np.std(note_returns))
    #
    # print('---')

    # print('{}% mean return ({}% annualized)'.format(round(note_mean, 2), round(
    #         100 * (1 + note_mean / 100) ** (1 / years) - 100, 2)))
    #
    # print('{} Sharpe Ratio'.format(round(note_mean / note_std, 2)))
    #
    # if (barrier > 0) and (len(loss) > 0):
    #     print('{}% recovery'.format(round(100+np.mean(loss)), 2))
    # # if autocall != 0:
    # #     dur = 100 * (dur + (path_length - len(autocalled)) * years) / (sum(note_returns) + 100 * len(note_returns))
    # #     if dur != 0:
    # #         if dur > 1:
    # #             print('Expected duration {} years'.format(round(dur, 2)))
    # #         else:
    # #             print(
    # #                 'Expected duration {} months'.format(round(12 * dur, 0)))
    # print('---')
    # if barrier > 0:
    #     print('{}% loss'.format(round(100 * len(loss) / path_length, 2)))
    # if (coupon > 0) or (memory > 0) or (digital > 0):
    #     print('{}% just coupon'.format(round(100 * len(just_cpn) / path_length, 2)))
    # if gearing > 0:
    #     print('{}% participation'.format(round(100 * len(particip) / path_length, 2)))
    #     if cap < 500:
    #         print('{}% capped'.format(round(100 * len(capped) / path_length, 2)))
    # if autocall > 0:
    #     print('{}% autocall'.format(round(100 * len(autocalled) / path_length, 2)))
    #
    # # if barrier > 0 and tail > 0:
    # #     print('---')
    # #     print('{}% probability of {}% tail'.format(round(100 * tail, 2), tail_event))
    # #     print('---')


def execute_calculation():
    for th in range(multi_thread):
        thread = multi.Process(target=monte_carlo, args=(notes_generated, ))
        thread.start()

calculate = multi.Process(target=execute_calculation)
calculate.start()
calculate.join()

print('\nExecuted in {} secs\n'.format(round(measure_time.time() - s, 2)))
