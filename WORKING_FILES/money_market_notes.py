from src.structured_products.results_analysis import calculate_ytm

years_till_maturity = 2
coupon_rate_decimal = 0.034
coupon_frequency_years = 0.5

ngdem_commission_per_deal = 400
notional = 10000000

print(calculate_ytm(years_till_maturity, (1 + ngdem_commission_per_deal / notional), coupon_rate_decimal,
                    coupon_frequency_years, 1))
