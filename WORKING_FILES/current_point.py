import openpyxl
import numpy as np
import scipy.stats
import tqdm
import itertools
import statsmodels.api as sm
from src.structured_products.DataParsing.GetRiskFreeRates import get_rf_rates


def optimize_by_sharpe_ratio(asset1, asset2, corr):
    sr_set = {}

    frontier = []
    for w in range(0, 100):
        w /= 100

        cov = w * (1 - w) * corr * asset1[1] * asset2[1]

        portfolio_mean = w * asset1[0] + (1 - w) * asset2[0]
        portfolio_std = np.sqrt(w**2 * asset1[1]**2 + (1 - w)**2 * asset2[1]**2 + 2 * cov)

        sr_set[portfolio_mean / portfolio_std] = [w, 1 - w]
        frontier.append([portfolio_mean, portfolio_std])

    best_sr = max(sr_set.keys())
    return [[round(weight, 2) for weight in sr_set[best_sr]], round(best_sr, 2), frontier]


def three_assets_optimize_by_sharpe_ratio(asset1, asset2, asset3, corr12, corr23, corr13):
    sr_set = {}

    frontier = []
    for w1, w2 in itertools.product(range(0, 100), range(0, 100)):
        w1 /= 100
        w2 /= 100

        cov12 = w1 * w2 * corr12 * asset1[1] * asset2[1]
        cov23 = w2 * (1 - w1 - w2) * corr23 * asset2[1] * asset3[1]
        cov13 = w1 * (1 - w1 - w2) * corr13 * asset1[1] * asset3[1]

        portfolio_mean = w1 * asset1[0] + w2 * asset2[0] + (1 - w1 - w2) * asset3[0]
        portfolio_std = np.sqrt(w1**2 * asset1[1]**2 + w2**2 * asset2[1]**2 + (1 - w1 - w2)**2 * asset3[1]**2
                                + 2 * cov12 + 2 * cov23 + 2 * cov13)

        sr_set[portfolio_mean / portfolio_std] = [w1, w2, 1 - w1 - w2]
        frontier.append([portfolio_mean, portfolio_std])

    best_sr = max(sr_set.keys())

    return [[round(weight, 2) for weight in sr_set[best_sr]], round(best_sr, 2), frontier]


def four_assets_optimize_by_sharpe_ratio(asset1, asset2, asset3, asset4,
                                         corr12, corr23, corr13, corr14, corr24, corr34):
    sr_set = {}

    frontier = []
    for w1, w2, w3 in itertools.product(range(0, 100), range(0, 100), range(0, 100)):
        w1 /= 100
        w2 /= 100
        w3 /= 100

        cov12 = w1 * w2 * corr12 * asset1[1] * asset2[1]
        cov23 = w2 * w3 * corr23 * asset2[1] * asset3[1]
        cov13 = w1 * w3 * corr13 * asset1[1] * asset3[1]
        cov14 = w1 * (1 - w1 - w2 - w3) * corr14 * asset1[1] * asset4[1]
        cov24 = w2 * (1 - w1 - w2 - w3) * corr24 * asset2[1] * asset4[1]
        cov34 = w3 * (1 - w1 - w2 - w3) * corr34 * asset3[1] * asset4[1]

        portfolio_mean = w1 * asset1[0] + w2 * asset2[0] + w3 * asset3[0] + (1 - w1 - w2 - w3) * asset4[0]
        portfolio_std = np.sqrt(w1**2 * asset1[1]**2 + w2**2 * asset2[1]**2 +
                                w3**2 * asset3[1]**2 + (1 - w1 - w2 - w3)**2 * asset4[1]**2
                                + 2 * cov12 + 2 * cov23 + 2 * cov13 + 2 * cov14 + 2 * cov24 + 2 * cov34)

        sr_set[portfolio_mean / portfolio_std] = [w1, w2, w3, (1 - w1 - w2 - w3)]
        frontier.append([portfolio_mean, portfolio_std])

    best_sr = max(sr_set.keys())

    return [[round(weight, 2) for weight in sr_set[best_sr]], round(best_sr, 2), frontier]


def call_value_at_risk(p, mu, sigma, term):
    # p transformation to account for bought otm call
    p += 1 - scipy.stats.lognorm.cdf(2, sigma, loc=mu)

    return np.exp(mu * term - scipy.stats.norm.ppf(p) * sigma * np.sqrt(term))


def put_value_at_risk(p, mu, sigma, term):
    return np.exp(mu * term - scipy.stats.norm.ppf(1-p) * sigma * np.sqrt(term))


def call_price_delta(spot, var, sigma, term, rf=0):

    d1 = (np.log(1 / var) + (rf + sigma**2/2)*term) / (sigma * np.sqrt(term))
    d2 = d1 - sigma * np.sqrt(term)

    d1_otm = (np.log(1 / (var + 1)) + (rf + sigma**2/2)*term) / (sigma * np.sqrt(term))
    d2_otm = d1_otm - sigma * np.sqrt(term)

    call = [(spot * scipy.stats.norm.cdf(d1) - var * spot * np.exp(-rf * term) * scipy.stats.norm.cdf(d2)) / spot,
            scipy.stats.norm.cdf(d1)]

    call_otm = [(spot * scipy.stats.norm.cdf(d1_otm) -
                 (var + 1) * spot * np.exp(-rf * term) * scipy.stats.norm.cdf(d2_otm)) / spot,
                scipy.stats.norm.cdf(d1_otm)]

    return [call[0] - call_otm[0], -call[1] + call_otm[1]]


def ui_call_price_delta(spot, var, sigma, term, rf=0):

    l_const = (rf + sigma**2 / 2) / sigma**2
    y = np.log(var**2) / (sigma * np.sqrt(term)) + l_const * sigma * np.sqrt(term)
    y1 = np.log(var) / (sigma * np.sqrt(term)) + l_const * sigma * np.sqrt(term)
    x1 = np.log(1 / var) / (sigma * np.sqrt(term)) + l_const * sigma * np.sqrt(term)

    d1_otm = (np.log(1 / 2) + (rf + sigma**2/2)*term) / (sigma * np.sqrt(term))
    d2_otm = d1_otm - sigma * np.sqrt(term)

    ui_call = [(spot * scipy.stats.norm.cdf(x1) -
                spot * np.exp(-rf * term) * scipy.stats.norm.cdf(x1 - sigma * np.sqrt(term)) -
                spot * var**(2*l_const) * (scipy.stats.norm.cdf(-y) - scipy.stats.norm.cdf(-y1)) +
                spot * np.exp(-rf * term) * var**(2*l_const - 2) *
                (scipy.stats.norm.cdf(-y + sigma * np.sqrt(term)) -
                 scipy.stats.norm.cdf(-y1 + sigma * np.sqrt(term)))) / spot]

    call_otm = [(spot * scipy.stats.norm.cdf(d1_otm) -
                 2 * spot * np.exp(-rf * term) * scipy.stats.norm.cdf(d2_otm)) / spot]

    return [ui_call[0] - call_otm[0], 0]


def put_price_delta(spot, var, sigma, term, rf=0):

    d1 = (np.log(1 / var) + (rf + sigma**2/2)*term) / (sigma * np.sqrt(term))
    d2 = d1 - sigma * np.sqrt(term)

    return [(var * spot * np.exp(-rf * term) * scipy.stats.norm.cdf(-d2) - spot * scipy.stats.norm.cdf(-d1)) / spot,
            scipy.stats.norm.cdf(d1) - 1]


def di_put_price_delta(spot, var, sigma, term, rf=0):

    l_const = (rf + sigma**2 / 2) / sigma**2
    y = np.log(var**2) / (sigma * np.sqrt(term)) + l_const * sigma * np.sqrt(term)
    y1 = np.log(var) / (sigma * np.sqrt(term)) + l_const * sigma * np.sqrt(term)
    x1 = np.log(1 / var) / (sigma * np.sqrt(term)) + l_const * sigma * np.sqrt(term)

    di_put_price = (-spot * scipy.stats.norm.cdf(-x1) +
                    spot * np.exp(-rf * term) * scipy.stats.norm.cdf(-x1 + sigma * np.sqrt(term)) +
                    spot * var**(2*l_const) * (scipy.stats.norm.cdf(y) - scipy.stats.norm.cdf(y1)) -
                    spot * np.exp(-rf * term) * var**(2*l_const - 2) *
                    (scipy.stats.norm.cdf(y - sigma * np.sqrt(term)) -
                     scipy.stats.norm.cdf(y1 - sigma * np.sqrt(term)))) / spot

    return [di_put_price, 0]


def put_expected_shortfall(p, mu, sigma, term):
    exp_part = np.exp(mu * term + sigma**2 * term / 2)
    norm_part = scipy.stats.norm.cdf(scipy.stats.norm.ppf(p) - sigma) / p

    return exp_part * norm_part - 1


ws = (openpyxl.load_workbook('jnk.xlsx')).active.values
xlsx_list = [item[:-1] for item in ws]
hy = []
for point in xlsx_list:
    hy.append([point[0], point[1]])

ws = (openpyxl.load_workbook('spx.xlsx')).active.values
xlsx_list = [item[:-1] for item in ws]
spx = {}
for point in xlsx_list:
    spx[point[0]] = [point[1]]

rf_dataset = get_rf_rates()

for point in hy:
    rf_key = str(point[0]).split('-')[0]
    key = point[0]
    if (key in spx.keys()) and (rf_key in rf_dataset.keys()):
        spx[key].append(point[1])
        spx[key].append(rf_dataset[rf_key])

merged = [datapoint for datapoint in spx.values() if len(datapoint) > 2]
merged.reverse()
merged = merged[:-7]

returns_merged = []
for day in range(1, len(merged)):
    rf_yield = (1 + merged[day][2] / 100) ** (1 / 365) - 1
    returns_merged.append([np.log(merged[day][0]) - np.log(merged[day - 1][0]) - rf_yield,
                           np.log(merged[day][1]) - np.log(merged[day - 1][1]) - rf_yield])

spx_returns = [point[0] for point in returns_merged]
arma_estimated = ArmaEstimation(np.array(spx_returns))
arma_coefficients = arma_estimated.coefficients

garch_estimated = StandardGarch(np.array(spx_returns), mu=0)
garch_coefficients = garch_estimated.coefficients
# print(['{:2f}'.format(c) for c in garch_coefficients])
gjr_garch_estimated = GJRGarch(np.array(spx_returns), mu=0)
gjr_garch_coefficients = gjr_garch_estimated.coefficients
# print(['{:2f}'.format(c) for c in gjr_garch_coefficients])

sigma_long_term = np.sqrt(252 * garch_coefficients[0] / (1 - garch_coefficients[1] - garch_coefficients[2]))
gjr_sigma = np.sqrt(252 * gjr_garch_coefficients[0] /
                    (1 - gjr_garch_coefficients[1] - gjr_garch_coefficients[2] / 2 - gjr_garch_coefficients[3]))

hy_returns = [point[1] for point in returns_merged]

spx_historical = [np.mean(spx_returns) * 252, np.std(spx_returns) * np.sqrt(252)]
hy_historical = [np.mean(hy_returns) * 252, np.std(hy_returns) * np.sqrt(252)]
correlation = scipy.stats.pearsonr(spx_returns, hy_returns)[0]

beta_stocks_bonds = sum([r[0] * r[1] for r in returns_merged]) / sum([r[0]**2 for r in returns_merged])
# print(beta_stocks_bonds)
errors = [r[1] - beta_stocks_bonds * r[0] for r in returns_merged]

hy_returns = [point[1] for point in returns_merged]
spx_returns = [point[0] for point in returns_merged]
model = sm.OLS(endog=spx_returns, exog=errors)
results = model.fit()
print(results.summary())
# model = sm.OLS(endog=hy_returns, exog=spx_returns)
# results = model.fit()
# print(results.summary())
# print(results.summary())
# print('---')

alpha = 0.02284
rf_rate = 0.026274
years = 2

transaction_cost_spx = 0.05 / 100
transaction_cost_hy = 0.05 / 100
transaction_cost_vanilla_option = 0.15 / 100
transaction_cost_barrier_option = 0.5 / 100

paths = 20000

# Calculate days for Monte Carlo simulation
t = 252 * years

# Create the linear space
time = np.linspace(0, t / 252, int(t))
# Delta time change in the linear space
d_time = time[1] - time[0]

# print(r_market * beta_stocks_bonds)

for vol in [sigma_long_term, gjr_sigma]:
    print('Vol: {}'.format(vol))

    mrp_market = np.exp(252 * arma_coefficients[0] / (1 - arma_coefficients[1]) + vol**2/2) - 1

    hy_distribution = [rf_rate + beta_stocks_bonds * mrp_market, abs(beta_stocks_bonds) * vol]
    # print([r_market, vol])
    # print(hy_distribution)
    strike_call = call_value_at_risk(alpha, rf_rate + mrp_market, vol, years)
    strike_put = put_value_at_risk(alpha, rf_rate + mrp_market, vol, years)

    call_data = call_price_delta(1, strike_call, vol, years)
    ui_call_data = ui_call_price_delta(1, strike_call, vol, years)
    put_data = put_price_delta(1, strike_put, vol, years)
    di_put_data = di_put_price_delta(1, strike_put, vol, years)
    # put_es = put_expected_shortfall(alpha, r_market, vol, 0, years) - (strike - 1)
    # print([put_data[0] + alpha * put_es, put_data[0]**2 +
    #                     alpha * (-2 * put_es + spx_historical[1]**2 + put_es**2)])

    var_covar = [[vol**2, correlation * vol * hy_distribution[1]],
                 [correlation * vol * hy_distribution[1], hy_historical[1]**2]]

    # Constant drift of log-normal, accumulated mean return is assumed to be at the level of r_market
    const_drift_spx = (rf_rate + mrp_market - vol ** 2 / 2) * d_time
    const_drift_hy = (rf_rate + beta_stocks_bonds * mrp_market - hy_distribution[1]**2 / 2) * d_time
    cholesky = np.linalg.cholesky(np.array(var_covar))

    spx_simulated = []
    hy_simulated = []
    call_simulated = []
    ui_call_simulated = []
    put_simulated = []
    di_put_simulated = []

    for k in tqdm.tqdm(range(paths)):
        path_spx = [1]
        path_hy = [1]

        for d in range(1, int(t)):
            joint_randomness = np.matmul(cholesky, np.random.normal(0, 1, size=2))

            spx_change = np.exp(const_drift_spx + joint_randomness[0] * np.sqrt(d_time))

            hy_change = np.exp(const_drift_hy + joint_randomness[1] * np.sqrt(d_time))

            path_spx.append(path_spx[d - 1] * spx_change)
            path_hy.append(path_hy[d - 1] * hy_change)

        r = path_spx[-1] - 1
        spx_simulated.append(r)
        hy_simulated.append(path_hy[-1] - 1)

        if r <= strike_put - 1:
            put_loss = r - strike_put + 1
            di_put_loss = r
        else:
            put_loss = 0
            di_put_loss = 0
        put_simulated.append(put_data[0] + put_loss)
        di_put_simulated.append(di_put_data[0] + di_put_loss)

        if r > strike_call:
            call_loss = -1
            ui_call_loss = -1
        elif r > 1:
            call_loss = -r + strike_call - 1
            ui_call_loss = -1
        elif r >= strike_call - 1:
            call_loss = -r + strike_call - 1
            ui_call_loss = -r
        else:
            call_loss = 0
            ui_call_loss = 0
        call_simulated.append(call_data[0] + call_loss)
        ui_call_simulated.append(ui_call_data[0] + ui_call_loss)

    # put_simulated = [put_price_delta(100, 100 * r, strike, std_simulated, 1)[0] for r in rv]
    spx_distribution = [np.mean(spx_simulated) - transaction_cost_spx, np.std(spx_simulated)]
    hy_distribution = [np.mean(hy_simulated) - transaction_cost_hy, np.std(hy_simulated)]

    rho_call_bonds = scipy.stats.pearsonr(call_simulated, hy_simulated)[0]
    rho_put_bonds = scipy.stats.pearsonr(put_simulated, hy_simulated)[0]
    rho_stocks_call = scipy.stats.pearsonr(spx_simulated, call_simulated)[0]
    rho_stocks_put = scipy.stats.pearsonr(spx_simulated, put_simulated)[0]
    rho_stocks_bonds = scipy.stats.pearsonr(spx_simulated, hy_simulated)[0]
    rho_call_put = scipy.stats.pearsonr(call_simulated, put_simulated)[0]
    rho_ui_call_bonds = scipy.stats.pearsonr(ui_call_simulated, hy_simulated)[0]
    rho_di_put_bonds = scipy.stats.pearsonr(di_put_simulated, hy_simulated)[0]
    rho_stocks_ui_call = scipy.stats.pearsonr(ui_call_simulated, spx_simulated)[0]
    rho_stocks_di_put = scipy.stats.pearsonr(di_put_simulated, spx_simulated)[0]
    rho_ui_call_di_put = scipy.stats.pearsonr(ui_call_simulated, di_put_simulated)[0]

    # print([np.mean(spx_simulated), np.std(spx_simulated)])
    # print([np.mean(hy_simulated), np.std(hy_simulated)])
    call_distribution = [np.mean(call_simulated) - transaction_cost_vanilla_option, np.std(call_simulated)]
    # print(call_distribution)
    put_distribution = [np.mean(put_simulated) - transaction_cost_vanilla_option, np.std(put_simulated)]
    # print(put_distribution)
    ui_call_distribution = [np.mean(ui_call_simulated) - transaction_cost_barrier_option, np.std(ui_call_simulated)]
    # print(ui_call_distribution)
    di_put_distribution = [np.mean(di_put_simulated) - transaction_cost_barrier_option, np.std(di_put_simulated)]
    # print(di_put_distribution)

    print(call_distribution, put_distribution)
    print(ui_call_distribution, di_put_distribution)

    stocks_bonds = optimize_by_sharpe_ratio(spx_distribution, hy_distribution, rho_stocks_bonds)
    vanilla_call = optimize_by_sharpe_ratio(call_distribution, hy_distribution, rho_call_bonds)
    vanilla_put = optimize_by_sharpe_ratio(put_distribution, hy_distribution, rho_put_bonds)
    barrier_call = optimize_by_sharpe_ratio(ui_call_distribution, hy_distribution, rho_ui_call_bonds)
    barrier_put = optimize_by_sharpe_ratio(di_put_distribution, hy_distribution, rho_di_put_bonds)

    print('Only Bonds: {}'.format(round(hy_distribution[0] / hy_distribution[1], 2)))

    print('Stocks & Bonds: {}'.format(stocks_bonds[:2]))

    print('Vanilla Call: {}'.format(vanilla_call[:2]))
    print('Vanilla Put: {}'.format(vanilla_put[:2]))

    print('Barrier Call: {}'.format(barrier_call[:2]))
    print('Barrier Put: {}'.format(barrier_put[:2]))

    print('\nThree assets:')

    vanilla_call_three = three_assets_optimize_by_sharpe_ratio(call_distribution,
                                                               spx_distribution, hy_distribution,
                                                               corr12=rho_stocks_call,
                                                               corr23=rho_stocks_bonds,
                                                               corr13=rho_call_bonds)
    vanilla_put_three = three_assets_optimize_by_sharpe_ratio(put_distribution,
                                                              spx_distribution, hy_distribution,
                                                              corr12=rho_stocks_put,
                                                              corr23=rho_stocks_bonds,
                                                              corr13=rho_put_bonds)

    barrier_call_three = three_assets_optimize_by_sharpe_ratio(ui_call_distribution,
                                                               spx_distribution, hy_distribution,
                                                               corr12=rho_stocks_ui_call,
                                                               corr23=rho_stocks_bonds,
                                                               corr13=rho_ui_call_bonds)

    barrier_put_three = three_assets_optimize_by_sharpe_ratio(di_put_distribution,
                                                              spx_distribution, hy_distribution,
                                                              corr12=rho_stocks_di_put,
                                                              corr23=rho_stocks_bonds,
                                                              corr13=rho_di_put_bonds)

    print('Vanilla Call: {}'.format(vanilla_call_three[:2]))

    print('Vanilla Put: {}'.format(vanilla_put_three[:2]))

    print('Barrier Call: {}'.format(barrier_call_three[:2]))

    print('Barrier Put: {}'.format(barrier_put_three[:2]))

    print('\nFour assets:')

    vanilla_four = four_assets_optimize_by_sharpe_ratio(call_distribution, put_distribution,
                                                        spx_distribution, hy_distribution,
                                                        corr12=rho_call_put,
                                                        corr23=rho_stocks_put,
                                                        corr13=rho_stocks_call,
                                                        corr14=rho_call_bonds,
                                                        corr24=rho_put_bonds,
                                                        corr34=rho_stocks_bonds)

    barrier_four = four_assets_optimize_by_sharpe_ratio(ui_call_distribution, di_put_distribution,
                                                        spx_distribution, hy_distribution,
                                                        corr12=rho_ui_call_di_put,
                                                        corr23=rho_stocks_di_put,
                                                        corr13=rho_stocks_ui_call,
                                                        corr14=rho_ui_call_bonds,
                                                        corr24=rho_di_put_bonds,
                                                        corr34=rho_stocks_bonds)

    print('Vanilla options: {}'.format(vanilla_four[:2]))

    print('Barrier options: {}'.format(barrier_four[:2]))

    with open('frontier_stocks_bonds', 'w') as f:
        for item in stocks_bonds[2]:
            f.write('{},{}\n'.format(item[0], item[1]))

    with open('frontier_barrier_bonds', 'w') as f:
        for item in barrier_put[2]:
            f.write('{},{}\n'.format(item[0], item[1]))

    with open('frontier_barrier_three', 'w') as f:
        for item in barrier_put_three[2]:
            f.write('{},{}\n'.format(item[0], item[1]))

    with open('frontier_barrier_four', 'w') as f:
        for item in barrier_four[2]:
            f.write('{},{}\n'.format(item[0], item[1]))

    print('---')
