import numpy as np
import scipy.stats
import tqdm

# INPUTS
# Number of paths for Monte Carlo simulation
PATHS = 10000

# Start value of the spot (initial fixing)
SPOT_START = 58.24
# Implied volatility in At-The-Money Option (sold option)
SIGMA_ATM = 0.3081
# Implied volatility in Out-of-The-Money Option (bought option)
SIGMA_OTM = SIGMA_ATM * 1.1
# Time till maturity in years (set YEARS_TILL_MATURITY = 0.5 for 6 months)
YEARS_TILL_MATURITY = 0.5
# Risk-Free rate in RUB leg
RISK_FREE_RATE_RUB = 0.07
# Risk-Free rate in CNH leg
RISK_FREE_RATE_CNH = -0.005

# RUB amount of notional
RUB_NOTIONAL = 500000000

# Risk-free commission in decimals from notional
PNL_NEEDED = 0.005

# Transaction cost in RUB per CNH
TRANSACTION_COST = 0.02

# Data for the VaR and confidence interval (may be different)
CONFIDENCE_LEVEL = 0.95
VaR_ALPHA = 0.01

# N of days before maturity, when greeks are irrelevant (too small time until maturity)
LAST_DAYS_IRRELEVANCE = 10


# FUNCTIONS
# Call price & Greeks
def get_call_price_and_greeks(spot, strike, term, sigma):

    # Factors that are used further
    volatility_factor = (r + (sigma ** 2) / 2)
    variance = sigma ** 2
    sigma_term = sigma * np.sqrt(term)

    # PROBABILITY DISTRIBUTION DATA
    # Calculate d1 and d2
    d1 = (np.log(spot/strike) + volatility_factor * term) / (sigma * np.sqrt(term))
    d2 = d1 - sigma * np.sqrt(term)

    # Calculate Standard Normal CDF of d1 and d2
    cdf_d1 = scipy.stats.norm.cdf(d1)
    cdf_d2 = scipy.stats.norm.cdf(d2)

    # Calculate Standard Normal PDF of d1 and d2
    pdf_d1 = scipy.stats.norm.pdf(d1)
    pdf_d2 = scipy.stats.norm.pdf(d2)
    pdf_d1_spot = pdf_d1 * spot

    # DERIVATIVES OF d1 AND d2 BY VARIABLES
    # Derivative by spot (same for d1 and d2)
    d1_ds = 1 / (spot * sigma_term)

    # Derivatives by volatility
    d1_dv = - np.log(spot/strike) / (np.sqrt(term) * variance) - r * np.sqrt(term) / variance + np.sqrt(term) / 2
    d2_dv = d1_dv - np.sqrt(term)

    # Derivative by interest rate
    d1_dr = np.sqrt(term) / sigma

    # Derivatives by time
    d1_dt = (r + variance/2) * sigma_term - (np.log(spot/strike) + volatility_factor)
    d2_dt = d1_dt - sigma / (2 * np.sqrt(term))

    # FINAL CALCULATIONS
    # Call price in currency
    call = cdf_d1 * spot - cdf_d2 * strike * np.exp(-r*term)
    # Greeks in points
    delta = cdf_d1 + spot * pdf_d1 * d1_ds - strike * np.exp(-r*term) * d1_ds * pdf_d2
    gamma = (pdf_d1 * d1_ds * (1 - 1 / (2 * sigma_term)) - strike / spot * np.exp(-r*term)
             / sigma_term * pdf_d2 * (-1/2 * d1_ds - 1 / spot)) / 2
    vega = (pdf_d1_spot * d1_dv - pdf_d2 * d2_dv * strike * np.exp(-r*term)) / 100
    theta = - (pdf_d1_spot * d1_dt - strike * np.exp(-r*term) * (pdf_d2 * d2_dt - r * cdf_d2))
    rho = (pdf_d1_spot * d1_dr - strike * np.exp(-r*term) * (pdf_d2 * d1_dr - term * cdf_d2)) / 100

    return [call,
            delta,
            gamma,
            vega,
            theta / 252,
            rho]


# The function to calculate Risk-Weighted Assets in RUB
def calculate_rwa(spot, vol_atm, vol_otm, delta_t0, delta_t1, gamma, vega_atm, vega_otm):

    # Constant for assumed variables change from CBR guidelines
    cbr_constant_for_spot_change = 0.08
    cbr_constant_for_volatility_change = 0.25
    provision_for_capital = 1 / cbr_constant_for_spot_change

    # Open Currency Position Risk: Difference between delta_t1 actual versus delta_t0 that was hedged
    ocp_risk = abs(delta_t1 - delta_t0) * cbr_constant_for_spot_change * spot
    # Gamma and Vega risks stay unhedged
    gamma_risk = abs(min(gamma, 0)) * (cbr_constant_for_spot_change * spot) ** 2
    vega_risk = notional * abs(cbr_constant_for_volatility_change * 100 * (-vega_atm * vol_atm + vega_otm * vol_otm))

    return provision_for_capital * (ocp_risk + gamma_risk + vega_risk)


# START OF THE PROGRAM EXECUTION
# Get notional in CNH
notional = RUB_NOTIONAL / SPOT_START

# Calculate rates difference
r = RISK_FREE_RATE_RUB - RISK_FREE_RATE_CNH

# T0 data for ATM option
greeks_atm = get_call_price_and_greeks(SPOT_START, SPOT_START, YEARS_TILL_MATURITY, SIGMA_ATM)

# Need to calculate the OTM call strike
atm_call_cost = notional * greeks_atm[0]
# Interest in RUB, available for paying for the total strategy (100% capital protection)
funding_interest = RUB_NOTIONAL * (1 - np.exp(-RISK_FREE_RATE_RUB * YEARS_TILL_MATURITY))
# The OTM call premium needed to offset the cost of ATM option in order to match the funding interest
otm_call_cost = atm_call_cost - funding_interest + RUB_NOTIONAL * PNL_NEEDED

# Iteration step (as, e.g., 31.23% strike would look strange for the client => take fixed step)
step_otm_solver = 5
# Find the respective strike of the OTM option
for potential_strike in range(20, 70, step_otm_solver):
    # Calculate the premium
    premium = get_call_price_and_greeks(SPOT_START, SPOT_START * (1 + potential_strike/100),
                                        YEARS_TILL_MATURITY, SIGMA_OTM)[0]
    # When the premium becomes less than the cost available => optimal was reached in previous iteration
    if premium * notional <= otm_call_cost:
        otm_strike_optimal = potential_strike - step_otm_solver
        break

print('---')
print('ATM call cost: {:,.2f}'.format(atm_call_cost))
print('Funding interest: {:,.2f}'.format(funding_interest))
print('OTM cost needed: {:,.2f}'.format(otm_call_cost))
print('OTM strike = {}%'.format(otm_strike_optimal))
print('---')

otm_strike_optimal = 1 + otm_strike_optimal / 100

# T0 data for the OTM option
greeks_otm = get_call_price_and_greeks(SPOT_START, otm_strike_optimal * SPOT_START, YEARS_TILL_MATURITY, SIGMA_OTM)

# Premium collected from the sold call-spread
initial_premium = greeks_atm[0] - greeks_otm[0]

# Greeks at the time the deal was made
initial_delta = notional * (-greeks_atm[1] + greeks_otm[1])
initial_gamma = notional * (-greeks_atm[2] + greeks_otm[2])
initial_vega = notional * (-greeks_atm[3] + greeks_otm[3])

print('Initial Data:')
print('Value (RUB): {:+,.2f}'.format(notional * (-greeks_atm[0] + greeks_otm[0])))
print('Delta: {:+,.2f}'.format(initial_delta))
print('Gamma: {:+,.2f}'.format(initial_gamma))
print('Vega: {:+,.2f}'.format(initial_vega))
print('Theta: {:+,.2f}'.format(notional * (-greeks_atm[4] + greeks_otm[4])))
print('Rho: {:+,.2f}'.format(notional * (-greeks_atm[5] + greeks_otm[5])))

print('---')
print('Total Risk for RWA (by CBR guidelines), RUB:')
print('Delta risk: {:+,.2f}'.format(initial_delta * 0.08 * SPOT_START))
print('Gamma risk: {:+,.2f}'.format(initial_gamma * 1/2 * (0.08 * SPOT_START)**2))
print('Vega risk: {:+,.2f}'.format(notional * 0.25 * 100 *
                                   (-greeks_atm[3] * SIGMA_ATM + greeks_otm[3] * SIGMA_OTM)))
print('---')
print('RWA: {:+,.2f}'.format(calculate_rwa(SPOT_START, SIGMA_ATM, SIGMA_OTM, initial_delta,
                                           initial_delta, initial_gamma, greeks_atm[3], greeks_otm[3])))

# Calculate days for Monte Carlo simulation
t = int(252 * YEARS_TILL_MATURITY)

# Create the linear space
time = np.linspace(0, t / 252, t)
# Delta time change in the linear space
d_time = time[1] - time[0]

# Constant drift of log-normal, mean return is assumed to be at the level of rates difference
const_drift = (r - 0.5 * SIGMA_ATM ** 2) * d_time

position_mc = []
greeks_output = []
RWA_series = []

print('---')

# At t0 initial
delta_for_hedging = -initial_delta * SPOT_START
# At t0 PnL from delta_hedge is only transaction cost from initial delta hedge
delta_pnl = -TRANSACTION_COST / SPOT_START * abs(delta_for_hedging)
# At t0 MtM is zero
MtM_position = 0

for k in tqdm.tqdm(range(PATHS)):

    # Create list of changes vs previous day by formula s(t+1) = s(t) * exp(mu + std.dev * Z)
    change = np.exp(const_drift + SIGMA_ATM * np.sqrt(d_time) * np.random.normal(0, 1, size=len(time)-1))

    # Create the list of spots by converting change into list of cumulative returns * SPOT_START
    path = [SPOT_START]
    for i in range(len(change)):
        path.append(path[i] * change[i])

    # Calculate the greeks, RWA and delta hedge PnL for each point in the list of simulated spots
    for q in range(1, len(path) - LAST_DAYS_IRRELEVANCE):

        curr_spot = path[q]

        greeks_atm = get_call_price_and_greeks(curr_spot, SPOT_START, (t-q)/t, SIGMA_ATM)
        greeks_otm = get_call_price_and_greeks(curr_spot, otm_strike_optimal * SPOT_START, (t-q)/t, SIGMA_OTM)

        # For each point of path calculate MtM & the set of greeks

        # MtM of option position + Initial premium placed at RISK_FREE_RATE_RUB
        MtM = notional * (-greeks_atm[0] + greeks_otm[0] + initial_premium * np.exp(RISK_FREE_RATE_RUB * q / 252))
        position_mc.append(MtM)
        # Get greeks in notional
        greeks_daily = [notional * (-greeks_atm[i] + greeks_otm[i]) for i in range(1, 6)]
        greeks_output.append(greeks_daily)

        # Calculate PnL from delta_hedge

        # Open Currency Position Risk: Difference between delta_t1 actual versus delta_t0 that was hedged
        ocp_pnl = delta_for_hedging * (curr_spot / path[q-1] - 1)
        # Funding cost of hedging delta
        funding_pnl = -(np.exp(RISK_FREE_RATE_RUB * 1 / 365) - 1) * delta_for_hedging
        # MtM PnL (expect to be netted with delta hedge, but assume position to be imperfectly hedged)
        MtM_pnl = MtM - MtM_position
        # Transaction cost from delta hedge
        transaction_pnl = -TRANSACTION_COST / curr_spot * abs(greeks_daily[0] + delta_for_hedging)

        # Sum all the effects
        delta_pnl += ocp_pnl + funding_pnl + MtM_pnl + transaction_pnl

        # Calculate RWA
        # Suppose that OCP exists as difference between delta(t) and delta(t-1) (some OCP remains unhedged)
        RWA_series.append(calculate_rwa(curr_spot, SIGMA_ATM, SIGMA_OTM, -delta_for_hedging / curr_spot,
                                        greeks_daily[0], greeks_daily[1], greeks_atm[3], greeks_otm[3]))

        MtM_position = MtM
        delta_for_hedging = -greeks_daily[0] * curr_spot

# Collect all greeks into one dataset (all possible greeks in all points of each trajectory)
delta_mc = [p[0] for p in greeks_output]
gamma_mc = [p[1] for p in greeks_output]
vega_mc = [p[2] for p in greeks_output]
theta_mc = [p[3] for p in greeks_output]
rho_mc = [p[4] for p in greeks_output]

print('---')

position_mc.sort()
delta_mc.sort()
gamma_mc.sort()
vega_mc.sort()
theta_mc.sort()
rho_mc.sort()
RWA_series.sort()

# Transform PATHS into number of overall points in each list of simulated values of a greek
PATHS *= t - LAST_DAYS_IRRELEVANCE - 1

# Alpha of the tails for confidence intervals
alpha = (1 - CONFIDENCE_LEVEL) / 2

print('Monte Carlo Simulation Results:')
# Get Confidence Interval & VaR (can be with different alphas)
print('Delta Hedge Cost, RUB = {:+,.2f}'.format(delta_pnl/PATHS))
print('---')

print('Position, RUB:')
print('Confidence Interval: [{:+,.2f}; {:+,.2f}]'.format(position_mc[int(PATHS * alpha)],
                                                         position_mc[int(PATHS * (1 - alpha))]))
print('{}% VaR = {:+,.2f}'.format(int(VaR_ALPHA * 100), position_mc[int(PATHS * VaR_ALPHA)]))
print('---')

print('Delta:')
print('Confidence Interval: [{:+,.2f}; {:+,.2f}]'.format(delta_mc[int(PATHS * alpha)],
                                                         delta_mc[int(PATHS * (1 - alpha))]))
print('{}% VaR = {:+,.2f}'.format(int(VaR_ALPHA * 100), delta_mc[int(PATHS * VaR_ALPHA)]))
print('---')

print('Gamma:')
print('Confidence Interval: [{:+,.2f}; {:+,.2f}]'.format(gamma_mc[int(PATHS * alpha)],
                                                         gamma_mc[int(PATHS * (1 - alpha))]))
print('{}% VaR = {:+,.2f}'.format(int(VaR_ALPHA * 100), gamma_mc[int(PATHS * VaR_ALPHA)]))
print('---')

print('Vega:')
print('Confidence Interval: [{:+,.2f}; {:+,.2f}]'.format(vega_mc[int(PATHS * alpha)],
                                                         vega_mc[int(PATHS * (1 - alpha))]))
print('{}% VaR = {:+,.2f}'.format(int(VaR_ALPHA * 100), vega_mc[int(PATHS * VaR_ALPHA)]))
print('---')

print('RWA, RUB:')
print('Confidence Interval: [{:+,.2f}; {:+,.2f}]'.format(RWA_series[int(PATHS * alpha)],
                                                         RWA_series[int(PATHS * (1 - alpha))]))
print('{}% VaR = {:+,.2f}'.format(int(VaR_ALPHA * 100), RWA_series[int(PATHS * (1 - VaR_ALPHA))]))
print('---')
