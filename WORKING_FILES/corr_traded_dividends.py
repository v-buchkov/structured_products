from src.structured_products.pricer.fully_funded_notes.fixed_coupon_notes import BarrierReverseConvertible
import datetime

wof_bid = 80.67
reoffer = 96.4

starting_prices = [61.09, 52.82, 137.02]
current_prices = [50.93, 42.76, 130.66]

# -------------------------------------------------------------

wof_price = (wof_bid + 100 - reoffer + 0.5) / 100

index_current_price = sum(current_prices) / len(current_prices)
index_starting_price = sum(starting_prices) / len(starting_prices)

initial_price_index = BarrierReverseConvertible(current_spot=index_starting_price,
                                                initial_spot_fixing=index_starting_price, risk_free_rate=0.02,
                                                time_till_maturity=(datetime.datetime(year=2024, month=2, day=15) -
                                                                    datetime.datetime.today()).days/365,
                                                underlying_volatility=0.2687, barrier_decimal=0.65,
                                                coupon_frequency=0.25, underlying_mean=0.02).price()

current_price_index = BarrierReverseConvertible(current_spot=index_current_price,
                                                initial_spot_fixing=index_starting_price, risk_free_rate=0.02,
                                                time_till_maturity=(datetime.datetime(year=2024, month=2, day=15) -
                                                                    datetime.datetime.today()).days/365,
                                                underlying_volatility=0.2687, barrier_decimal=0.65,
                                                coupon_frequency=0.25, underlying_mean=0.02).price()

print(wof_price - 1, current_price_index - initial_price_index)
