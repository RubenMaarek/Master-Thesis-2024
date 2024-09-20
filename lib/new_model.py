import numpy as np
from scipy.stats import norm
from scipy.linalg import cholesky
from scipy.stats import random_correlation
import random
from lib.utils import (
    qp_solution,
    qubitize_price_system,
    extract_original_matrix,
    pad_zeros, make_hermitian,
    extract_N_and_K
)
from lib.options_functions import *
from typing import Union
payoff_functions_list = [call_option, put_option]  # Add more if needed

# def monte_carlo_option_price(S0, strike_price, sigma, mu, T, r, payoff_function, num_simulations=10000):
#     """
#     Monte Carlo estimation of the option price.
    
#     Args:
#         S0 (float): Initial price of the asset
#         strike_price (float): Strike price of the option
#         sigma (float): Volatility of the asset
#         mu (float): Drift (expected return) of the asset
#         T (float): Time to maturity
#         r (float): Risk-free rate
#         payoff_function (function): Payoff function of the option
#         num_simulations (int): Number of Monte Carlo simulations
    
#     Returns:
#         float: Monte Carlo estimated option price
#     """
#     # Simulate end prices at time T
#     ST = S0 * np.exp((mu - 0.5 * sigma ** 2) * T + sigma * np.sqrt(T) * np.random.normal(size=num_simulations))
#     # Calculate payoffs for each simulated price
#     payoffs = payoff_function(ST, strike_price)
#     # Discount back to present value
#     discounted_payoffs = np.exp(-r * T) * payoffs
#     # Return the mean of the discounted payoffs
#     return np.mean(discounted_payoffs)

def full_payoff_system(Pi, mus, sigmas, strike_ratios, K, payoff_functions, Rho, perturbation_scale=1e-5, r=0.1, T=1):
    """
    Payoff matrix of single-period model with correlated Brownian motions and Monte Carlo option pricing.
    Now includes QP solution to Sx = Pi.
    
    Args:
        Pi (list): list of current prices
        mus (list): list of drifts
        sigmas (list): list of volatilities
        strike_ratios : list of ratios such that strike_prices = strike_ratios * current_prices
        K (int): number of events
        payoff_functions : list of payoff functions, one for each asset
        Rho : correlation matrix of size NxN
        perturbation_scale (float): scale of the perturbation added to break linear dependencies
        num_simulations (int): Number of Monte Carlo simulations for option pricing
    
    Returns:
        All defining components of the system : S, Pi, mus, sigmas, D, strike_prices, MC_call_prices, qp_solution
    """
    # Ensure inputs are numpy arrays, of same length
    Pi, mus, sigmas, strike_ratios = map(np.asarray, (Pi, mus, sigmas, strike_ratios))
    
    if not (Pi.shape == mus.shape == sigmas.shape == strike_ratios.shape == np.array(payoff_functions).shape):
        raise ValueError("Pi, mus, sigmas, strike_ratios, and payoff_functions must all have the same shape.")
    
    N = len(Pi)

    L = cholesky(Rho, lower=True)  # Cholesky decomposition
    W = np.random.normal(0, 1, (N, K))  # Shape: (N, K), independent Brownian motions
    B = np.dot(L, W)  # Shape: (N, K), correlated Brownian motions

    S = np.zeros((N, K))
    D = np.zeros((N, K))
    strike_prices = strike_ratios * Pi
    
    MC_prices = np.zeros(N)
    
    for i in range(N):
        S[i] = Pi[i] * np.exp(sigmas[i] *np.sqrt(T)* B[i] + (mus[i] - 0.5 * sigmas[i] ** 2)*T)

        # # Add small perturbations to break linear dependencies
        # S[i] += np.random.normal(scale=perturbation_scale, size=S[i].shape)
        D[i] = payoff_functions[i](S[i], strike_prices[i])
        
        # Apply discount rates to the payoffs
        S[i] *= np.exp(-r * T)
        D[i] *= np.exp(-r * T)

        # Estimate the option price using Monte Carlo simulation
        MC_prices[i] = np.mean(D[i]) 

    # Solve the QP problem Sq = Pi
    qp_solution_vector = qp_solution(S, Pi)

    # Return everything including the string names of the functions
    payoff_function_names = [func.__name__ for func in payoff_functions]

    return S, Pi, mus, sigmas, D, strike_prices, payoff_function_names, MC_prices, qp_solution_vector


def random_payoff_functions(N):
    """
    Generate a random list of payoff functions of length N.
    """
    return [random.choice(payoff_functions_list) for _ in range(N)]

# Function to generate a random correlation matrix
def generate_random_correlation_matrix(N, alpha=0.2, beta=1.8):
    eigenvalues = np.random.uniform(alpha, beta, N)
    eigenvalues = N * eigenvalues / np.sum(eigenvalues)
    correlation_matrix = random_correlation.rvs(eigenvalues)
    return correlation_matrix

def random_payoff_system(N, K):
    """
    Generate a random payoff matrix of size (N,K)
    """
    Pi = np.random.uniform(80, 120, N)  # Prices between 80 and 120
    mus = np.linspace(0.01, 0.3, N) + np.random.normal(0, 0.01, N)  # Drifts with slight perturbations
    sigmas = np.linspace(0.1, 0.7, N) + np.random.normal(0, 0.02, N)  # Volatilities with slight perturbations
    strike_ratios = np.random.uniform(0.90, 1.20, N)
    payoff_functions = random_payoff_functions(N)  # Generate random list of payoff functions
    Rho = generate_random_correlation_matrix(N)

    return full_payoff_system(Pi, mus, sigmas, strike_ratios, K, payoff_functions, Rho)

def random_payoff_system_from_qubits(n_qubits):
    """
    Given n_qubits, generate a random payoff matrix S, and a random vector Pi
    of size 2^n_qubits
    """
    N, K = select_N_and_K(n_qubits)
    S, Pi, mus, sigmas, D, strike_prices, payoff_functions, MC_prices, qp_solution_vector = random_payoff_system(N, K)
    S, Pi, norm_Pi = qubitize_price_system(S, Pi)
    D = np.real(pad_zeros(make_hermitian(D)))
    return S, Pi, mus, sigmas, D, strike_prices, payoff_functions, MC_prices, qp_solution_vector, N, K, norm_Pi

def select_N_and_K(n_qubits, f:float = 2):
    """
    Given the system size of n_qubits, returns a pair (N,K) such that
    $ 2^(n_qubits-1) < N+K <= 2^n_qubits $
    ie: $ 2^{n-1}+ 1 <= N+K < 2^n + 1 $
    and f*N <= K to make an under-determined system (f>1).
    Used to generate a payoff_matrix given only the qubit-size of the system
    """
    min_sum = 2 ** (n_qubits - 1) + 1
    max_sum = 2**n_qubits + 1

    # Randomly choose the value of N+K
    sum_N_K = np.random.randint(min_sum, max_sum)

    # Ensure K>= f*N
    max_N = int(sum_N_K / (1 + f))
    min_N = 2
    if max_N < min_N:
        return None, None

    N = np.random.randint(min_N, max_N + 1)
    K = sum_N_K - N

    return N, K



