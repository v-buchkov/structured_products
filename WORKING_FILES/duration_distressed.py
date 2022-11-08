import bond_pricing as bp

price = 35
cpn = 6
coupon_frequency = 0.5
term_years = 5

# ytm = 2 * ((100 - price) / term_years + cpn) / (100 + price)
ytm = bp.bond_yield(cpn=cpn/100, mat=term_years, price=price, freq=int(1/coupon_frequency))

duration_sum = sum([t * coupon_frequency * coupon_frequency * cpn / ((1 + ytm) ** (t * coupon_frequency))
                    for t in range(1, int(term_years/coupon_frequency) + 1)]) + \
               term_years * ((coupon_frequency * cpn + 100) / ((1 + ytm) ** term_years))

modified_duration = (duration_sum / price) / (1 + ytm / coupon_frequency)

print(modified_duration)
