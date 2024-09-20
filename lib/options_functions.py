# Can also try a linear combination of the assets:
#  call(\alpha S1 + \beta S2, \alpha K1 + \beta K2)
# = \alpha call(S1, K1) + \beta call(S2, K2) ?
# Is this true for the Black-Scholes model ?
# If yes, try it with VQE method for D pricing to get more results /data
import numpy as np
from scipy.stats import norm
from typing import Union

def handle_dimensions(s, z, operation):
    s = np.array(s)
    z = np.array(z)
    if s.ndim == 0 or z.ndim == 0:
        return operation(s, z)
    elif s.ndim == 1 and z.ndim == 1:
        return operation(s, z)
    elif s.ndim == 1 and z.ndim == 0:
        return operation(s, z)
    elif s.ndim == 2 and z.ndim == 1 and s.shape[0] == len(z):
        return operation(s, z[:, np.newaxis])
    elif s.ndim == 2 and z.ndim == 2 and s.shape == z.shape:
        return operation(s, z)
    else:
        raise ValueError("Invalid input shapes.")

def call_option(s, z):
    return handle_dimensions(s, z, lambda s, z: np.maximum(0, s - z))

def put_option(s, z):
    return handle_dimensions(s, z, lambda s, z: np.maximum(0, z - s))

def digital_call_option(s, z, payout=1):
    return handle_dimensions(s, z, lambda s, z: payout * (s > z).astype(float))

def digital_put_option(s, z, payout=1):
    return handle_dimensions(s, z, lambda s, z: payout * (s < z).astype(float))

def exchange_option(s1, s2):
    return handle_dimensions(s1, s2, lambda s1, s2: np.maximum(0, s1 - s2))

def compound_call_on_call_option(call_value, z):
    return handle_dimensions(call_value, z, lambda c, z: np.maximum(0, c - z))

def compound_put_on_call_option(call_value, z):
    return handle_dimensions(call_value, z, lambda c, z: np.maximum(0, z - c))

def straddle_option(s, z):
    call_payoff = call_option(s, z)
    put_payoff = put_option(s, z)
    return call_payoff + put_payoff

def strangle_option(s, z_call, z_put):
    call_payoff = call_option(s, z_call)
    put_payoff = put_option(s, z_put)
    return call_payoff + put_payoff

def black_scholes_price(Pi, strike_prices, sigmas, r, T, payoff_functions, payouts=None):
    BS_prices = np.zeros(len(Pi))
    for i, payoff_function in enumerate(payoff_functions):
        d1 = (np.log(Pi[i] / strike_prices[i]) + (r + 0.5 * sigmas[i] ** 2) * T) / (sigmas[i] * np.sqrt(T))
        d2 = d1 - sigmas[i] * np.sqrt(T)
        if payoff_function == call_option:
            BS_prices[i] = Pi[i] * norm.cdf(d1) - strike_prices[i] * np.exp(-r * T) * norm.cdf(d2)
        elif payoff_function == put_option:
            BS_prices[i] = strike_prices[i] * np.exp(-r * T) * norm.cdf(-d2) - Pi[i] * norm.cdf(-d1)
        elif payoff_function == digital_call_option:
            payout = payouts[i] if payouts else 1
            BS_prices[i] = payout * np.exp(-r * T) * norm.cdf(d2)
        elif payoff_function == digital_put_option:
            payout = payouts[i] if payouts else 1
            BS_prices[i] = payout * np.exp(-r * T) * norm.cdf(-d2)
        # elif payoff_function == exchange_option:
        #     # Assuming s2 is another price in the context
        #     s2 = strike_prices[i]  # Overloading the strike_prices variable
        #     d1 = (np.log(Pi[i] / s2) + (0.5 * sigmas[i] ** 2) * T) / (sigmas[i] * np.sqrt(T))
        #     d2 = d1 - sigmas[i] * np.sqrt(T)
        #     BS_prices[i] = Pi[i] * norm.cdf(d1) - s2 * np.exp(-r * T) * norm.cdf(d2)
        # elif payoff_function == compound_call_on_call_option:
        #     # Here, Pi[i] is assumed to be the value of the underlying call option
        #     BS_prices[i] = Pi[i] * norm.cdf(d1) - strike_prices[i] * np.exp(-r * T) * norm.cdf(d2)
        # elif payoff_function == compound_put_on_call_option:
        #     BS_prices[i] = strike_prices[i] * np.exp(-r * T) * norm.cdf(-d2) - Pi[i] * norm.cdf(-d1)
        else:
            raise ValueError("Unsupported payoff function for asset {}".format(i))
    return BS_prices