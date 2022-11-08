import numpy as np
from tqdm import tqdm

p_default = 0.05
vol_p_default = 1.5

recovery_rate = 0.5
rf_rate = 0.07

term_years = 5
coupon_frequency = 0.5

vol_gld = 0.1966
c_gld = 0.025

c_rub = 3.69 * 2 / 100
ytm_rub = 0.086

real_rate_rub = 0.075 - 0.03

correlation = 0.85


# ----------------------------------------------------------------------------------------------------------------------


def rub_expected_price(p_d: float) -> float:
    rub_final_no_default = (coupon_frequency * c_rub + 1) / ((1 + ytm_rub) ** term_years)
    rub_final_default = recovery_rate * (coupon_frequency * c_rub + 1) / ((1 + ytm_rub) ** term_years)

    return sum_rub_coupons + (1 - p_d) * rub_final_no_default + p_d * rub_final_default


def gld_expected_price(exp_spot: float, p_d: float) -> float:
    gld_final_no_default = (coupon_frequency * c_gld + 1) * pv_fwd_rates[-1]
    gld_final_default = recovery_rate * (coupon_frequency * c_gld + 1) * pv_fwd_rates[-1] +\
                        (1 - recovery_rate) * (exp_spot - pv_fwd_rates[-1])

    return sum_gld_coupons + (1 - p_d) * gld_final_no_default + p_d * gld_final_default


pv_fwd_rates = [np.exp(t * coupon_frequency * rf_rate) / ((1 + ytm_rub) ** (t * coupon_frequency))
                for t in range(1, int(term_years/coupon_frequency) + 1)]

sum_gld_coupons = sum([coupon_frequency * c_gld * fwd_rate for fwd_rate in pv_fwd_rates[:-1]])
sum_rub_coupons = sum([coupon_frequency * c_rub / ((1 + ytm_rub) ** (t * coupon_frequency))
                       for t in range(1, int(term_years/coupon_frequency) + 1)])

gld_exp_price = gld_expected_price((1 + real_rate_rub) ** term_years, p_default)
rub_exp_price = rub_expected_price(p_default)
print(gld_exp_price - rub_exp_price)

paths = 20000

# Calculate days for Monte Carlo simulation
t = 252 * term_years

# Create the linear space
time = np.linspace(0, t / 252, int(t))
# Delta time change in the linear space
d_time = time[1] - time[0]

var_covar = [[vol_gld**2, correlation * vol_gld * vol_p_default],
             [correlation * vol_gld * vol_p_default, vol_p_default**2]]

# Constant drift of log-normal, accumulated mean return is assumed to be at the level of r_market
const_drift_spot = (real_rate_rub - vol_gld ** 2 / 2) * d_time
const_drift_p_default = -vol_p_default**2 / 2 * d_time

cholesky = np.linalg.cholesky(np.array(var_covar))

gld_expected_price_simulated = []
rub_expected_price_simulated = []
arb_trade = []
for k in tqdm(range(paths)):
    path_spot = [1]
    path_p_default = [p_default]

    for d in range(1, int(t)):
        joint_randomness = np.matmul(cholesky, np.random.normal(0, 1, size=2))

        spot_change = np.exp(const_drift_spot + joint_randomness[0] * np.sqrt(d_time))

        p_default_change = np.exp(const_drift_p_default + joint_randomness[1] * np.sqrt(d_time))

        path_spot.append(path_spot[d - 1] * spot_change)
        path_p_default.append(min(max(path_p_default[d - 1] * p_default_change, 0), 1))

    gld_price = gld_expected_price(path_spot[-1] - 1, path_p_default[-1])
    rub_price = rub_expected_price(path_p_default[-1])

    gld_expected_price_simulated.append(gld_price)
    rub_expected_price_simulated.append(rub_price)

    arb_trade.append(gld_price - rub_price)

# expected_price_simulated.sort(reverse=True)
# print(sorted(gld_expected_price_simulated))
# print(sorted(rub_expected_price_simulated))

print(np.mean(arb_trade), np.std(arb_trade), sorted(arb_trade)[int(0.05 * len(arb_trade))], sorted(arb_trade)[0])
