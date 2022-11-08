import openpyxl
import numpy as np
import numpy_financial as npf
import scipy.stats
import tqdm
import itertools
from src.structured_products.DataParsing.GetRiskFreeRates import get_rf_rates

# set ETF ticker to backtest
etf = 'jnk'
# start = % (in decimals) of available points to use for training models
start = 0.2
# time until maturity of the structured product
years = 1
# optimal probability of loss under VaR setting (set equal to the corresponding probability of default)

# add parsing ratings curves & probability of default
alpha_optimal = 0.11161

# transaction costs (set to be constant)
transaction_cost_spx = 0.05 / 100
transaction_cost_hy = 0.05 / 100
transaction_cost_barrier_option = 0.5 / 100


# get optimal portfolio via maximizing Sharpe numerically
# need to pass lists of means and var-covar matrix


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


ws = (openpyxl.load_workbook('{}.xlsx'.format(etf))).active.values
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

returns_historical_merged = []
for day in range(1, len(merged)):
    rf_yield = (1 + merged[day][2] / 100) ** (1 / 365) - 1
    returns_historical_merged.append([np.log(merged[day][0]) - np.log(merged[day - 1][0]) - rf_yield,
                                      np.log(merged[day][1]) - np.log(merged[day - 1][1]) - rf_yield])


def get_optimal_weights(rf_rate, returns_merged, years_left, r_market, alpha, vol):

    spx_returns = [p[0] for p in returns_merged]
    hy_returns = [p[1] for p in returns_merged]
    correlation = scipy.stats.pearsonr(spx_returns, hy_returns)[0]

    beta_stocks_bonds = sum([r[0] * r[1] for r in returns_merged]) / sum([r[0]**2 for r in returns_merged])

    strike_call = call_value_at_risk(alpha, r_market, vol, years_left)
    ui_call_data = ui_call_price_delta(1, strike_call, vol, years_left)
    strike_put = put_value_at_risk(alpha, r_market, vol, years_left)
    di_put_data = di_put_price_delta(1, strike_put, vol, years_left)

    paths = 10000

    # Calculate days for Monte Carlo simulation
    t = int(252 * years_left)

    # Create the linear space
    time = np.linspace(0, t / 252, t)
    # Delta time change in the linear space
    d_time = time[1] - time[0]

    var_covar = [[vol**2, correlation * vol * beta_stocks_bonds * vol],
                 [correlation * vol * beta_stocks_bonds * vol, (beta_stocks_bonds * vol)**2]]

    # Constant drift of log-normal, mean return is assumed to be at the level of rates difference
    const_drift_spx = (rf_rate + r_market - 0.5 * var_covar[0][0] ** 2) * d_time
    const_drift_hy = (rf_rate + beta_stocks_bonds * r_market - 0.5 * var_covar[1][1] ** 2) * d_time
    cholesky = np.linalg.cholesky(np.array(var_covar))

    spx_simulated = []
    hy_simulated = []
    ui_call_simulated = []
    di_put_simulated = []

    for k in range(paths):
        path_spx = [1]
        path_hy = [1]

        for d in range(1, t):
            joint_randomness = np.matmul(cholesky, np.random.normal(0, 1, size=2))

            spx_change = np.exp(const_drift_spx + joint_randomness[0] * np.sqrt(d_time))

            hy_change = np.exp(const_drift_hy + joint_randomness[1] * np.sqrt(d_time))

            path_spx.append(path_spx[d - 1] * spx_change)
            path_hy.append(path_hy[d - 1] * hy_change)

        r = path_spx[-1] - 1
        spx_simulated.append(r)
        hy_simulated.append(path_hy[-1] - 1)

        if r <= strike_put - 1:
            di_put_loss = r
        else:
            di_put_loss = 0
        di_put_simulated.append(di_put_data[0] + di_put_loss)

        if r > strike_call:
            ui_call_loss = -1
        elif r > 1:
            ui_call_loss = -1
        elif r >= strike_call - 1:
            ui_call_loss = -r
        else:
            ui_call_loss = 0
        ui_call_simulated.append(ui_call_data[0] + ui_call_loss)

    # put_simulated = [put_price_delta(100, 100 * r, strike, std_simulated, 1)[0] for r in rv]
    spx_distribution = [np.mean(spx_simulated), np.std(spx_simulated)]
    hy_distribution = [np.mean(hy_simulated), np.std(hy_simulated)]

    rho_stocks_bonds = scipy.stats.pearsonr(spx_simulated, hy_simulated)[0]
    rho_ui_call_bonds = scipy.stats.pearsonr(ui_call_simulated, hy_simulated)[0]
    rho_di_put_bonds = scipy.stats.pearsonr(di_put_simulated, hy_simulated)[0]
    rho_stocks_ui_call = scipy.stats.pearsonr(ui_call_simulated, spx_simulated)[0]
    rho_stocks_di_put = scipy.stats.pearsonr(di_put_simulated, spx_simulated)[0]
    rho_ui_call_di_put = scipy.stats.pearsonr(ui_call_simulated, di_put_simulated)[0]

    # UI mean is negative, because have to pay for far call
    ui_call_distribution = [np.mean(ui_call_simulated), np.std(ui_call_simulated)]
    di_put_distribution = [np.mean(di_put_simulated), np.std(di_put_simulated)]

    # Mean-Variance
    weights_frontier = optimize_by_sharpe_ratio([spx_distribution[0], hy_distribution[0]],
                                                calculate_var_covar_matrix(
                                                    [spx_distribution[1], hy_distribution[1]],
                                                    rho_stocks_bonds))[0]
    weights_frontier = optimize_by_sharpe_ratio(spx_distribution, hy_distribution, rho_stocks_bonds)[0]

    # Bonds + Put
    weights_two = optimize_by_sharpe_ratio(di_put_distribution, hy_distribution, rho_di_put_bonds)[0]

    # Mean-Variance + Put
    weights_three = three_assets_optimize_by_sharpe_ratio(di_put_distribution, spx_distribution, hy_distribution,
                                                          corr12=rho_stocks_di_put,
                                                          corr23=rho_stocks_bonds,
                                                          corr13=rho_di_put_bonds)[0]

    # Mean-Variance + Strangle
    weights_four = four_assets_optimize_by_sharpe_ratio(ui_call_distribution, di_put_distribution, spx_distribution,
                                                        hy_distribution, corr12=rho_ui_call_di_put,
                                                        corr23=rho_stocks_di_put,
                                                        corr13=rho_stocks_ui_call,
                                                        corr14=rho_ui_call_bonds,
                                                        corr24=rho_di_put_bonds,
                                                        corr34=rho_stocks_bonds)[0]

    return [[weights_frontier, weights_two, weights_three, weights_four],
            [strike_call, ui_call_data[0]],
            [strike_put, di_put_data[0]]]


def calculate_weight_to_buy(w_new, w_prev, capital_already_invested):
    return w_new + (w_new - w_prev) * capital_already_invested


just_bonds = []
frontier_portfolio = []
two_portfolio = []
three_portfolio = []
four_portfolio = []
ui_calls_bought = {}
di_puts_bought = {}
weights_previous = [[0, 0], [0, 0], [0, 0, 0], [0, 0, 0, 0]]
spx_price_previous = 0
hy_price_previous = 0

results = []

capital_invested = 0

# Variant 1 - Each day invest $X, s.t. weights in the new portfolio is optimal
for i in tqdm.tqdm(range(int(start * len(merged)) + 1, len(merged) - 1)):

    two_for_reinvestment = 0
    three_for_reinvestment = 0
    four_for_reinvestment = 0

    ui_call_result = 0
    di_put_result = 0

    train_dataset = returns_historical_merged[:i]

    rf_rate = float(merged[i+1][2]) / 100

    capital_invested *= (1 + rf_rate)**(1 / 365)

    capital_invested += 1

    spx_returns = [item[0] for item in train_dataset]

    arma_estimated = ArmaEstimation(np.array(spx_returns))
    garch_estimated = GJRGarch(np.array(spx_returns), mu=0)
    garch_coefficients = garch_estimated.coefficients
    vol_predict = np.sqrt(252 * garch_coefficients[0] / (1 - garch_coefficients[1] - garch_coefficients[2]))
    mrp_predict = np.exp(vol_predict ** 2 / 2) - 1

    spx_price = merged[i][0]
    hy_price = merged[i][1]

    function_output = get_optimal_weights(rf_rate, train_dataset, years, mrp_predict, alpha_optimal, vol_predict)
    weights = function_output[0]

    just_bonds.append(1 / (hy_price * (1 + transaction_cost_hy)))

    frontier_portfolio.append([weights[0][0] / (spx_price * (1 + transaction_cost_spx)),
                               weights[0][1] / (hy_price * (1 + transaction_cost_hy))])

    if (i in ui_calls_bought.keys()) and (i in di_puts_bought.keys()):

        ui_call_terms = ui_calls_bought[i]
        di_put_terms = di_puts_bought[i]

        if spx_price > 2 * ui_call_terms[1]:
            ui_call_loss = -1
            di_put_loss = 0
        elif spx_price > ui_call_terms[1]:
            ui_call_loss = (merged[i - years * 252][0] - spx_price) / merged[i - years * 252][0]
            di_put_loss = 0
        elif spx_price < di_put_terms[1]:
            ui_call_loss = 0
            di_put_loss = (spx_price - merged[i - years * 252][0]) / merged[i - years * 252][0]
        else:
            ui_call_loss = 0
            di_put_loss = 0

        if transaction_cost_barrier_option > ui_call_terms[2]:
            ui_call_result = (1 + ui_call_terms[2] * (1 - transaction_cost_barrier_option * 100)) * \
                             (1 + float(merged[i - years * 252][2]) / 100) + ui_call_loss
        else:
            ui_call_result = (1 + ui_call_terms[2] - transaction_cost_barrier_option) * \
                                 (1 + float(merged[i-years*252][2])/100) + ui_call_loss

        if transaction_cost_barrier_option > di_put_terms[2]:
            di_put_result = (1 + di_put_terms[2] * (1 - transaction_cost_barrier_option * 100)) * \
                            (1 + float(merged[i - years * 252][2]) / 100) + di_put_loss
        else:
            di_put_result = (1 + di_put_terms[2] - transaction_cost_barrier_option) * \
                            (1 + float(merged[i-years*252][2])/100) + di_put_loss

        two_for_reinvestment = di_put_terms[0][0] * di_put_result
        three_for_reinvestment = di_put_terms[0][1] * di_put_result
        four_for_reinvestment = ui_call_terms[0] * ui_call_result + di_put_terms[0][2] * di_put_result

    # 'expiry': [[weight_two, weight_three, weight_four], strike]
    if i < len(merged) - years * 252 - 1:
        ui_calls_bought[i + years * 252] = [weights[3][0] * (1 + four_for_reinvestment),
                                            function_output[1][0] * spx_price, function_output[1][1]]

        di_puts_bought[i + years * 252] = [[weights[1][0] * (1 + two_for_reinvestment),
                                            weights[2][0] * (1 + three_for_reinvestment),
                                            weights[3][1] * (1 + four_for_reinvestment)],
                                           function_output[2][0] * spx_price, function_output[2][1]]

        two_portfolio.append([0, weights[1][1] * (1 + two_for_reinvestment) / (hy_price * (1 + transaction_cost_hy))])

        three_portfolio.append([weights[2][1] * (1 + three_for_reinvestment) / (spx_price * (1 + transaction_cost_spx)),
                                weights[2][2] * (1 + three_for_reinvestment) / (hy_price * (1 + transaction_cost_hy))])

        four_portfolio.append([weights[3][2] * (1 + four_for_reinvestment) / (spx_price * (1 + transaction_cost_spx)),
                               weights[3][3] * (1 + four_for_reinvestment) / (hy_price * (1 + transaction_cost_hy))])
    else:
        two_portfolio.append([0, weights[0][1] * (1 + two_for_reinvestment) / (hy_price * (1 + transaction_cost_hy))])

        three_portfolio.append([weights[0][0] * (1 + three_for_reinvestment) / (spx_price * (1 + transaction_cost_spx)),
                                weights[0][1] * (1 + three_for_reinvestment) / (hy_price * (1 + transaction_cost_hy))])

        four_portfolio.append([weights[0][0] * (1 + four_for_reinvestment) / (spx_price * (1 + transaction_cost_spx)),
                               weights[0][1] * (1 + four_for_reinvestment) / (hy_price * (1 + transaction_cost_hy))])

    just_bonds_performance = sum(just_bonds) * hy_price

    if sum([u[0] for u in frontier_portfolio]) < 0:
        print(sum([u[0] for u in frontier_portfolio]))

    if sum([u[0] for u in two_portfolio]) < 0:
        print(sum([u[0] for u in two_portfolio]))

    if sum([u[0] for u in three_portfolio]) < 0:
        print(sum([u[0] for u in three_portfolio]))

    if sum([u[0] for u in four_portfolio]) < 0:
        print(sum([u[0] for u in four_portfolio]))

    if sum([u[1] for u in frontier_portfolio]) < 0:
        print(sum([u[1] for u in frontier_portfolio]))

    if sum([u[1] for u in two_portfolio]) < 0:
        print(sum([u[1] for u in two_portfolio]))

    if sum([u[1] for u in three_portfolio]) < 0:
        print(sum([u[1] for u in three_portfolio]))

    if sum([u[1] for u in four_portfolio]) < 0:
        print(sum([u[1] for u in four_portfolio]))

    frontier_performance = sum([u[0] for u in frontier_portfolio]) * spx_price + \
                           sum([u[1] for u in frontier_portfolio]) * hy_price

    two_portfolio_performance = sum([u[0] for u in two_portfolio]) * spx_price + \
                                sum([u[1] for u in two_portfolio]) * hy_price

    three_portfolio_performance = sum([u[0] for u in three_portfolio]) * spx_price + \
                                  sum([u[1] for u in three_portfolio]) * hy_price

    four_portfolio_performance = sum([u[0] for u in four_portfolio]) * spx_price + \
                                 sum([u[1] for u in four_portfolio]) * hy_price

    with open('results_final_{}'.format(etf), 'a') as r_file:
        r_file.write('{}-{}\n'.format([rf_rate, mrp_predict, vol_predict],
                                      [capital_invested, just_bonds_performance, frontier_performance,
                                       [two_portfolio_performance, di_put_result],
                                       [three_portfolio_performance, di_put_result],
                                       [four_portfolio_performance, di_put_result, ui_call_result]]))
    with open('weights_{}'.format(etf), 'a') as w_file:
        w_file.write('{}\n'.format(weights))
        # r_file.write('{}\n'.format(two_portfolio))
        # r_file.write('---')

    weights_previous = weights
    spx_price_previous = spx_price
    hy_price_previous = hy_price_previous

# with open('results', 'w') as r:
#     for item in results:
#         r.write('{}\n'.format(item))

capital_cash_flows_for_irr = -1 * np.ones(int(len(merged) - int(start * len(merged))))

just_bonds_performance = sum(just_bonds) * merged[-1][1]

just_bonds_cfs = np.append(capital_cash_flows_for_irr, just_bonds_performance)

just_bonds_irr = 100 * ((1 + npf.irr(just_bonds_cfs))**365 - 1)

frontier_performance = sum([u[0] for u in frontier_portfolio]) * merged[-1][0] + \
                       sum([u[1] for u in frontier_portfolio]) * merged[-1][1]

frontier_cfs = np.append(capital_cash_flows_for_irr, frontier_performance)

frontier_irr = 100 * ((1 + npf.irr(frontier_cfs))**365 - 1)

two_portfolio_performance = sum([u[0] for u in two_portfolio]) * merged[-1][0] + \
                       sum([u[1] for u in two_portfolio]) * merged[-1][1]

two_portfolio_cfs = np.append(capital_cash_flows_for_irr, two_portfolio_performance)

two_portfolio_irr = 100 * ((1 + npf.irr(two_portfolio_cfs))**365 - 1)

three_portfolio_performance = sum([u[0] for u in three_portfolio]) * merged[-1][0] + \
                       sum([u[1] for u in three_portfolio]) * merged[-1][1]

three_portfolio_cfs = np.append(capital_cash_flows_for_irr, three_portfolio_performance)

three_portfolio_irr = 100 * ((1 + npf.irr(three_portfolio_cfs))**365 - 1)

four_portfolio_performance = sum([u[0] for u in four_portfolio]) * merged[-1][0] + \
                       sum([u[1] for u in four_portfolio]) * merged[-1][1]

four_portfolio_cfs = np.append(capital_cash_flows_for_irr, four_portfolio_performance)

four_portfolio_irr = 100 * ((1 + npf.irr(four_portfolio_cfs))**365 - 1)

print('Capital invested = {:,.2f}'.format(capital_invested))

print('Just Bonds = {:,.2f}, IRR = {:,.2f}%'.format(just_bonds_performance, just_bonds_irr))

print('Mean-Variance Frontier = {:,.2f}, IRR = {:,.2f}%'.format(frontier_performance, frontier_irr))

print('Barrier Put & Bonds = {:,.2f}, IRR = {:,.2f}%'.format(two_portfolio_performance, two_portfolio_irr))

print('Stocks & Bonds & Barrier Put = {:,.2f}, IRR = {:,.2f}%'.format(three_portfolio_performance, three_portfolio_irr))

print('Barrier Call & Barrier Put & Stocks & Bonds = {:,.2f}, IRR = {:,.2f}%'.format(four_portfolio_performance,
                                                                                     four_portfolio_irr))
