"""
The file provides functions to analyze the results of the structured products performance
"""
import numpy as np
import numpy_financial as npf


def calculate_dollar_cost_averaging_irr(dollars_invested_each_day: float, days_in_strategy: int,
                                        final_portfolio_value: float):
    """
    The function returns Internal Rate of Return for a dollar cost averaging portfolio
    """
    # Normalize payoff by each day investment amount
    portfolio_payoff_normalized = final_portfolio_value / dollars_invested_each_day

    # Assume tht each day $1 invested (as payoff is normalized)
    capital_cash_flows_for_irr = -1 * np.ones(days_in_strategy)

    # Get cash flows list as $1 "days_in_strategy" times + final payoff as the last value
    cash_flows_list = np.append(capital_cash_flows_for_irr, portfolio_payoff_normalized)

    # Calculate annualized Internal Rate of Return in percentage
    irr = (1 + npf.irr(cash_flows_list))**365 - 1

    return irr


def calculate_ytm(time_till_maturity: float, purchase_price_decimal: float, coupon_per_annum_decimal: float,
                  frequency_years: float, redemption_amount_decimal: float):
    """
    The function returns annualized Yield To Maturity for a fixed income instrument
    """
    cash_flows = [-purchase_price_decimal]

    for i in range(int(time_till_maturity/frequency_years) - 1):
        cash_flows.append(coupon_per_annum_decimal * frequency_years)

    cash_flows.append((coupon_per_annum_decimal * frequency_years + redemption_amount_decimal))

    return npf.irr(np.array(cash_flows)) / frequency_years


def calculate_ytm_non_standard_cash_flows(percentage_cash_flow_list: list, payment_frequency_years: float):
    """
    The function returns annualized Yield To Maturity for some instrument with unusual cash flows

    Requires list of cash flows and frequency of payments as inputs
    """
    cash_flows = np.array(percentage_cash_flow_list) / abs(percentage_cash_flow_list[0])

    return npf.irr(cash_flows) / payment_frequency_years
