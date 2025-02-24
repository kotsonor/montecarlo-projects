import numpy as np
import math
from scipy.stats import norm, chi2
import pandas as pd
import matplotlib.pyplot as plt



# Generate Brownian motion
def create_covariance_matrix(n):
    indices = np.arange(1, n + 1) / n
    Sigma = np.minimum.outer(indices, indices)
    return Sigma

def generate_brownian_motion(n, R):
    B = np.random.multivariate_normal(np.zeros(n), create_covariance_matrix(n), R)
    return B


def crude_monte_carlo(S0, K, r, sigma, R, n):
    """
    Estimate option price using the Crude Monte Carlo method.

    Parameters:
    S0 (float): Initial stock price
    K (float): Strike price
    r (float): Risk-free interest rate
    sigma (float): Volatility
    R (int): Number of simulations
    n (int): Number of time steps

    Returns:
    float: Estimated option price
    """
    mu = r - (sigma**2) / 2
    B = generate_brownian_motion(n, R)
    t = np.arange(1, n + 1) / n
    S_t = S0 * np.exp(mu * t + sigma * B)
    A_n = np.mean(S_t, axis=1)
    I_r = np.exp(-r) * np.maximum(A_n - K, 0)
    I = np.exp(-r) * np.mean(np.maximum(A_n - K, 0))

    # return {"CMC": I,
    #         "Sample": I_r,
    #         "Var": np.var(I_r)/R}
    return I, I_r, np.var(I_r)/R


def black_scholes_call(S0, K, r, sigma):
    """
    Calculates the price of a European call option using the Black-Scholes formula.

    :param S0: Current price of the underlying asset (S(0))
    :param K: Strike price of the option (K)
    :param r: Risk-free interest rate (r)
    :param sigma: Volatility of the underlying asset (σ)
    :return: Price of the European call option
    """
    # Calculate d1 and d2
    d1 = (math.log(S0 / K) + r + (sigma**2) / 2) / sigma
    d2 = d1 - sigma

    # Calculate the call option price
    call_price = S0 * norm.cdf(d1) - K * math.exp(-r) * norm.cdf(d2)
    return call_price



def generate_cholesky_matrix(n):
    """
    Generate the same matrix A more efficiently using numpy.
    """
    i, j = np.tril_indices(n)
    A = np.zeros((n, n))
    A[i, j] = 1 / np.sqrt(n)
    return A


def generate_stratified_brownian_motion(n, R, m):
    """
    Generate stratified Brownian motion samples.

    Parameters:
    n (int): Dimension of the Brownian motion.
    R (int or np.array): Number of samples or array specifying samples per stratum.
    m (int): Number of strata.

    Returns:
    dict: Dictionary where keys are strata indices (1 to m) and values are samples (arrays).
    """
    A = generate_cholesky_matrix(n)  # Generate Cholesky decomposition
    stratum = {}

    # Check if R is a single integer or an array
    if isinstance(R, int):
        R_list = [R // m] * m  # Divide R evenly across strata
    elif isinstance(R, (np.ndarray, list)):
        if len(R) != m:
            raise ValueError("Length of R array must match the number of strata (m).")
        R_list = R
    else:
        raise TypeError("R must be an integer or a numpy array.")

    for i, R_i in enumerate(R_list):
        ksi = np.random.multivariate_normal(np.zeros(n), np.eye(n), R_i).T
        X = ksi / np.linalg.norm(ksi, axis=0)
        U = np.random.uniform(0, 1, R_i)  # Random U from uniform distribution (0, 1)
        quantile = i / m + (1 / m) * U  # Compute quantile
        D_squared = chi2.ppf(quantile, df=n)
        Z = np.sqrt(D_squared) * X
        B_i = A @ Z
        stratum[i + 1] = B_i.T  # dim = R_i x n

    return stratum


def stratified_crude_monte_carlo(S0, K, r, sigma,
                                 n, R, m):
    if isinstance(R, int):
        R_list = np.array([R // m] * m)  # Divide R evenly across strata
    elif isinstance(R, (np.ndarray, list)):
        if len(R) != m:
            raise ValueError("Length of R array must match the number of strata (m).")
        R_list = R
    else:
        raise TypeError("R must be an integer or a numpy array.")

    str_bm = generate_stratified_brownian_motion(n, R_list, m)
    I_sum = 0
    I_j_dictionary = {}
    var_j = []
    m = len(str_bm)
    for key, value in str_bm.items():
        B_i = value
        mu = r - (sigma ** 2) / 2
        t = np.arange(1, n + 1) / n
        S_t = S0 * np.exp(mu * t + sigma * B_i)
        A_n = np.mean(S_t, axis=1)
        I_j = np.exp(-r) * np.maximum(A_n - K, 0)
        I = np.exp(-r) * np.mean(np.maximum(A_n - K, 0))

        I_j_dictionary[key] = I_j
        I_sum += I
        var_j.append(np.var(I_j))

    I_result = I_sum / m
    R_int = np.sum(R_list)
    # Calculate optimal R_j
    p_j = 1 / m  # Equal weights
    sigma_j = np.sqrt(np.array(var_j))
    denom = np.sum(p_j * sigma_j)
    R_j = np.int64(p_j * sigma_j / denom * R_int)

    variance = p_j ** 2 * np.sum(var_j / R_list)

    return I_result, I_j_dictionary, R_j, variance




def k(B):
    S = S0 * np.exp((r - (sigma ** 2) / 2) * np.arange(1, n + 1) / n + sigma * B)
    A_n = np.mean(S)
    payoff = np.maximum(A_n - K, 0)
    discounted_payoff = np.exp(-r) * payoff
    return discounted_payoff


def antithetic_sampling(R, n):
    A = create_covariance_matrix(n)
    I_antithetic = 0
    Y_pairs = []

    for i in range(R // 2):
        Z = np.random.normal(0, 1, n)
        Z_antithetic = -Z

        B = Z @ A.T
        B_antithetic = Z_antithetic @ A.T

        Y = k(B)
        Y_prime = k(B_antithetic)

        Y_pairs.append((Y, Y_prime))
        Y_antithetic = (Y + Y_prime) / 2
        I_antithetic += Y_antithetic

    I_antithetic /= (R // 2)
    Y_values = np.array([Y for Y, Y_prime in Y_pairs])
    Y_prime_values = np.array([Y_prime for Y, Y_prime in Y_pairs])
    cov_Y_Y_prime = np.cov(Y_values, Y_prime_values)[0, 1]
    var_Y_1 = np.var(Y_values)
    var_Y_2 = np.var(Y_prime_values)
    corr_Y_Y_prime = cov_Y_Y_prime / np.sqrt(var_Y_1 * var_Y_2)
    var_Y = np.var(np.concatenate((Y_values, Y_prime_values)))
    var_I_antithetic = (var_Y / R) * (1 + corr_Y_Y_prime)

    return I_antithetic, var_I_antithetic, var_Y / R, corr_Y_Y_prime




def control_variate_sampling(R, n):
    # Create the covariance matrix for Brownian motion
    A = create_covariance_matrix(n)

    # Initialize arrays to store Y and X values
    Y_values = []
    X_values = []

    # Simulate R replications
    for i in range(R):
        # Generate standard normal random variables
        Z = np.random.normal(0, 1, n)

        # Simulate Brownian motion paths
        B = Z @ A.T

        # Compute Y (discounted payoff) and X (control variate)
        Y = k(B)  # Replace k(B) with the actual payoff function
        X = B[-1]  # Control variate: B(1), the value of Brownian motion at time T=1

        # Store Y and X values
        Y_values.append(Y)
        X_values.append(X)

    # Convert lists to numpy arrays
    Y_values = np.array(Y_values)
    X_values = np.array(X_values)

    # Compute covariance and variance
    cov_Y_X = np.cov(Y_values, X_values)[0, 1]  # Covariance between Y and X
    var_X = np.var(X_values)  # Variance of X
    var_Y = np.var(Y_values)  # Variance of Y

    # Compute the optimal beta
    beta = cov_Y_X / var_X

    # Compute the control variate estimator
    Y_cv = Y_values - beta * X_values  # Adjust Y using the control variate
    I_cv = np.mean(Y_cv)  # Compute the mean of the adjusted Y values

    # Compute the variance of the control variate estimator
    coef = (1 - (cov_Y_X ** 2) / (var_X * var_Y))
    var_Y_cv = (var_Y / R) * coef

    return I_cv, var_Y_cv, coef


def monte_carlo_pricing(S0, K, r, sigma, R, n):
    # Crude Monte Carlo
    cmc_price, _, cmc_variance = crude_monte_carlo(S0, K, r, sigma, R, n)

    # Stratified Monte Carlo (m = 5, 10, 20 strata)
    proportional_results = {}
    optimal_results = {}

    for m in [5, 10, 20]:
        # Proportional allocation
        strat_price_proportional = stratified_crude_monte_carlo(S0, K, r, sigma, n, R, m)
        strat_price, _, R_optimal, strat_variance = strat_price_proportional

        # Optimal allocation
        strat_price_optimal = stratified_crude_monte_carlo(S0, K, r, sigma, n, R_optimal, m)
        strat_price_optimal, _, _, strat_variance_optimal = strat_price_optimal

        # Store results
        proportional_results[m] = {"price": strat_price, "variance": strat_variance}
        optimal_results[m] = {"price": strat_price_optimal, "variance": strat_variance_optimal}

    # Generate the tables
    call_price = black_scholes_call(S0, K, r, sigma)

    proportional_data = {
        "CMC": [cmc_price, cmc_variance],
        "Strati5": [proportional_results[5]["price"], proportional_results[5]["variance"]],
        "Strati10": [proportional_results[10]["price"], proportional_results[10]["variance"]],
        "Strati20": [proportional_results[20]["price"], proportional_results[20]["variance"]],
    }
    proportional_df = pd.DataFrame(proportional_data, index=["Y", "Var(Y)"])
    proportional_df.loc["bias"] = abs(call_price - proportional_df.loc["Y"])

    optimal_data = {
        "CMC": [cmc_price, cmc_variance],
        "Strati5": [optimal_results[5]["price"], optimal_results[5]["variance"]],
        "Strati10": [optimal_results[10]["price"], optimal_results[10]["variance"]],
        "Strati20": [optimal_results[20]["price"], optimal_results[20]["variance"]],
    }
    optimal_df = pd.DataFrame(optimal_data, index=["Y", "Var(Y)"])
    optimal_df.loc["bias"] = abs(call_price - optimal_df.loc["Y"])

    return proportional_df, optimal_df


def antithetic_control(S0, K, r, sigma, R, n):
    call_price = black_scholes_call(S0, K, r, sigma)
    cmc_price, _, cmc_variance = crude_monte_carlo(S0, K, r, sigma, R, n)
    I_cv, var_I_cv, control_coef = control_variate_sampling(R, n)
    I_antithetic, var_I_antithetic, _, anti_coef = antithetic_sampling(R, n)
    optimal_data = {
        "CMC": [cmc_price, cmc_variance, 1],
        "Antithetic": [I_antithetic, var_I_antithetic, 1+anti_coef],
        "Control": [I_cv, var_I_cv, control_coef],
    }
    optimal_df = pd.DataFrame(optimal_data, index=["Y", "Var(Y)", "Coefficient"])
    optimal_df.loc["bias"] = abs(call_price - optimal_df.loc["Y"])

    return optimal_df




def simulate_and_plot_option_prices(S0, K, r, sigma, R, m, max_n=50):
    """
    Simulates and plots the comparison of option prices using stratified and crude Monte Carlo methods,
    including prediction bounds based on variances.

    Parameters:
    S0 (float): Initial stock price
    K (float): Strike price
    r (float): Risk-free interest rate
    sigma (float): Volatility
    R (int): Number of simulations
    m (int): Number of strata
    max_n (int): Maximum number of time steps (n) to simulate
    """
    # Arrays to store prices and variances for each method
    n_values = np.arange(1, max_n + 1)  # Values for n (number of time steps)

    strat_prices = []
    strat_opt_prices = []
    cmc_prices = []

    strat_variances = []
    strat_opt_variances = []
    cmc_variances = []

    # Run the simulation for each n value
    for n in n_values:
        # Crude Monte Carlo
        cmc_price, cmc_sample, cmc_variance = crude_monte_carlo(S0, K, r, sigma, R, n)
        cmc_prices.append(cmc_price)
        cmc_variances.append(cmc_variance)

        # Stratified Crude Monte Carlo
        strat_price_proportional, strat_sample, R_optimal, strat_variance = stratified_crude_monte_carlo(S0, K, r,sigma,
                                                                                                         n, R, m)
        strat_prices.append(strat_price_proportional)
        strat_variances.append(strat_variance)

        # Optimal Stratified Crude Monte Carlo
        strat_price_optimal, _, _, strat_variance_optimal = stratified_crude_monte_carlo(S0, K, r, sigma,
                                                                                         n, R_optimal, m)
        strat_opt_prices.append(strat_price_optimal)
        strat_opt_variances.append(strat_variance_optimal)

    # Calculate prediction bounds
    strat_upper = np.array(strat_prices) + 1.96 * np.sqrt(np.array(strat_variances))
    strat_lower = np.array(strat_prices) - 1.96 * np.sqrt(np.array(strat_variances))
    cmc_upper = np.array(cmc_prices) + 1.96 * np.sqrt(np.array(cmc_variances))
    cmc_lower = np.array(cmc_prices) - 1.96 * np.sqrt(np.array(cmc_variances))
    strat_opt_upper = np.array(strat_opt_prices) + 1.96 * np.sqrt(np.array(strat_opt_variances))
    strat_opt_lower = np.array(strat_opt_prices) - 1.96 * np.sqrt(np.array(strat_opt_variances))

    # Plot the prices
    plt.figure(figsize=(10, 6))
    plt.plot(n_values, cmc_prices, label="Crude Monte Carlo Price", color="blue", linestyle="-")
    plt.fill_between(n_values, cmc_lower, cmc_upper, color="blue", alpha=0.2, label="CMC Prediction Bounds")

    plt.plot(n_values, strat_prices, label="Stratified Price", color="green", linestyle="--")
    plt.fill_between(n_values, strat_lower, strat_upper, color="green", alpha=0.2, label="Stratified Bounds")

    plt.plot(n_values, strat_opt_prices, label="Optimal Stratified Price", color="orange", linestyle="-.")
    plt.fill_between(n_values, strat_opt_lower, strat_opt_upper, color="orange", alpha=0.2,
                     label="Optimal Stratified Bounds")

    # Add labels, legend, and title
    plt.xlabel("Number of Time Steps (n)")
    plt.ylabel("Option Price")
    plt.title("Option Pricing Comparison: Monte Carlo Methods")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()



if __name__ == "__main__":
    S0 = 100
    K = 100
    r = 0.05
    sigma = 0.25
    R = 10**3
    n = 1

    # Figure 1
    np.random.seed(42)
    proportional_df, optimal_df = monte_carlo_pricing(S0, K, r, sigma, R, n)
    print("Proportional Allocation Results for R=", R)
    print(proportional_df)
    print("\nOptimal Allocation Results for R=", R)
    print(optimal_df)

    # Figure 2
    np.random.seed(42)
    anti_df = antithetic_control(S0, K, r, sigma, R, n)
    print("\nAntithetic and Control Variate Results for R=", R)
    print(anti_df)


    # Figure 3
    np.random.seed(42)
    R = 10 ** 5  # Number of simulations
    m = 10  # Number of strata
    simulate_and_plot_option_prices(S0, K, r, sigma, R, m, max_n=35)



