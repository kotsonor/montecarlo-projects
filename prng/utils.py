import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import pandas as pd
from scipy.special import gammaincc
import urllib.request


def read_digits(url):
    data = []
    with urllib.request.urlopen(url) as f:
        for line in f:
            data.append(line.strip())
    datastring = []

    for line in data:
        datastring.append(line.decode("utf-8"))

    datastring = ''.join(datastring)
    datastring = list(map(int, list(datastring)))

    return (np.array(datastring))


'''
GENERATORS
'''


def lcg(n, modulus, a, c, seed):
    observations = []
    for _ in range(n):
        seed = (a * seed + c) % modulus
        observations.append(seed)
    return np.array(observations) / modulus


def glcg(n, modulus, a, initial_state):
    a = np.array(a)[::-1]
    state = list(initial_state)
    k = len(a)
    if len(state) != k:
        raise ValueError(
            f"The length of the initial state (seed) must match the number of coefficients 'a' which equals {k}."
        )

    results = []
    for _ in range(n):
        xn = (np.dot(a, state[-k:])) % modulus
        results.append(xn)
        state.append(xn)

    return np.array(results) / modulus


def rc4_ksa(m, key):
    S = np.arange(m)
    key_len = key.shape[0]
    j = 0
    for i in range(m):
        j = (j + S[i] + key[i % key_len]) % m
        S[i], S[j] = S[j], S[i]
    return S


def rc4_prga(S, n):
    S_len = S.shape[0]
    i = 0
    j = 0
    output = []
    for _ in range(n):
        i = (i + 1) % S_len
        j = (j + S[i]) % S_len
        S[i], S[j] = S[j], S[i]
        x = S[(S[i] + S[j]) % S_len]
        output.append(x)
    return np.array(output)


'''
TESTS
'''


def find_period(prng_values):
    values = {}
    for i, number in enumerate(prng_values):
        if number in values:
            return i - values[number]
        values[number] = i
    return None


def compute_obs_in_bins(partition, points):
    counts, _ = np.histogram(points, bins=partition)
    bins_probs = np.diff(partition)

    return counts, bins_probs


def generate_rc4_sequence(m, L, n_all, n):
    R = int(n_all / n)
    final_sequence = []

    for i in range(R):
        key = np.random.randint(0, m, size=L, dtype=int)
        S = rc4_ksa(m, key)
        prga_output = rc4_prga(S, n)
        final_sequence.extend(prga_output)
    scaled_sequence = np.array(final_sequence) / m

    return scaled_sequence


def perform_statistical_tests(x_sample, M=None):
    partition = np.linspace(0, 1, 11)  # [0, 0.1, 0.2, ..., 1]
    counts, probs = compute_obs_in_bins(partition, x_sample)
    chi2_stat, chi2_p_value = np.round(stats.chisquare(counts, x_sample.shape[0] * probs), 3)

    if M == 2**32:
         chi2_stat2, chi2_p_value2 = "-", "-"
    else:
        unique_values, counts2 = np.unique(x_sample, return_counts=True)
        counts = np.zeros(M)
        unique_values = np.int64(unique_values * M)
        counts[unique_values] = counts2
        chi2_stat2, chi2_p_value2 = np.round(stats.chisquare(counts, x_sample.shape[0] * 1/M), 3)

    ks_stat, ks_p_value = np.round(stats.kstest(x_sample, 'uniform', args=(0, 1)), 3)
    bft_chi2, bft_p_value = np.round(block_frequency_test(x_sample, np.int64(x_sample.shape[0]/100)), 3)

    results = pd.DataFrame({
        'Test': ['Chi-square A', 'Chi-square B', 'Kolmogorov-Smirnov', 'Block Frequency Test'],
        'Statistic': [chi2_stat, chi2_stat2, ks_stat, bft_chi2],
        'p-value': [chi2_p_value, chi2_p_value2, ks_p_value, bft_p_value]
    })

    return results


def block_frequency_test(x, M):
    x = (x > 0.5).astype(np.int8)
    n = x.shape[0]
    N = n // M
    x = x[:N * M].reshape(N, M)
    pi = np.mean(x, axis=1)
    chi_squared = 4 * M * np.sum((pi - 0.5) ** 2)
    p_value = gammaincc(N / 2, chi_squared / 2)

    return chi_squared, p_value


def second_level_testing(x_sample, m, n_subsets = 100):
    subset_size = len(x_sample) // n_subsets
    p_values = np.zeros((n_subsets, 4))

    for i in range(n_subsets):
        subset = x_sample[i * subset_size:(i + 1) * subset_size]
        results = perform_statistical_tests(subset, m)
        if m == 2**32:
            results.loc[1, "p-value"] = 0
        p_values[i] = results['p-value'].values

    return p_values


def test_p_values(p_values, gen_name):
    chi_p_values = p_values[:, 0]
    ks_p_values = p_values[:, 2]
    bft_p_values = p_values[:, 3]

    results = []
    # ['Block Frequency Test', 'Kolmogorov-Smirnov', 'Chi-square B', 'Chi-square A']
    if gen_name == "MT":
        tests = [(chi_p_values, "Chi-square A"), (ks_p_values, "Kolmogorov-Smirnov"), (bft_p_values, 'Block Frequency Test')]
    else:
        chi_p_values2 = p_values[:, 1]
        tests = [(chi_p_values, "Chi-square A"), (chi_p_values2, "Chi-square B"), (ks_p_values, "Kolmogorov-Smirnov"), (bft_p_values, 'Block Frequency Test')]

    for test_p_values, test_name in tests:
        partition = np.linspace(0, 1, 11)
        counts, probs = compute_obs_in_bins(partition, test_p_values)
        chi2_stat, chi2_p_value = np.round(stats.chisquare(counts, test_p_values.shape[0] * probs), 3)

        decision = "Reject" if chi2_p_value < 0.05 else "Cannot Reject"

        results.append([test_name, chi2_stat, chi2_p_value, decision])

    df = pd.DataFrame(results, columns=["Test Name", "Chi2 Statistic", "P-Value", "Decision"])

    return df


def plot_p_values(p_values, gen_name):
    chi_p_values = p_values[:, 0]
    chi_p_values2 = p_values[:, 1]
    ks_p_values = p_values[:, 2]
    bft_p_values = p_values[:, 3]

    if gen_name == "MT":
        y_values = np.concatenate([np.zeros_like(chi_p_values),
                               np.ones_like(chi_p_values2),
                               2*np.ones_like(ks_p_values),
                               ])

        p_values_combined = np.concatenate([bft_p_values, ks_p_values, chi_p_values])
    else:
        y_values = np.concatenate([np.zeros_like(chi_p_values),
                                np.ones_like(chi_p_values2),
                                2*np.ones_like(ks_p_values),
                                3*np.ones_like(bft_p_values),
                                ])

        p_values_combined = np.concatenate([bft_p_values, ks_p_values, chi_p_values2, chi_p_values])

    plt.figure(figsize=(10, 6))
    plt.scatter(p_values_combined, y_values, c=y_values, cmap='coolwarm', s=50, edgecolor='k')

    plt.title(f'Second level testing p-values for different statistical tests for {gen_name}')
    plt.xlabel('P-value')
    plt.ylabel('Test')
    if gen_name == "MT":
        plt.yticks([0, 1, 2], ['Block Frequency Test','Kolmogorov-Smirnov', 'Chi-square A'])
    else:
        plt.yticks([0, 1, 2, 3], ['Block Frequency Test','Kolmogorov-Smirnov', 'Chi-square B', 'Chi-square A'])
    plt.xlim(-0.1, 1.1)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()


def run_test_for_generator(generator, seed):
    n = 10**6

    if generator == "LCG":
        print("# LCG(13, 1, 5)")
        M, a, c = 13, 1, 5
        x = lcg(n, M, a, c, seed = 2)
        df = perform_statistical_tests(x, M)
        p_values = second_level_testing(x, M)

    elif generator == "GLCG":
        print("# GLCG(1024, {3, 7, 68})")
        M = 2**10
        a = [3, 7, 68]
        initial_state = seed
        x = glcg(n, M, a, initial_state)
        df = perform_statistical_tests(x, M)
        p_values = second_level_testing(x, M)

    elif generator == "RC4":
        print("# RC4(32)")
        np.random.seed(seed)
        m, L, n_sample = 32, 10, 10**3
        x = generate_rc4_sequence(m, L, n, n_sample)
        df = perform_statistical_tests(x, m)
        p_values = second_level_testing(x, m)

    elif generator == "MT":
        print("# MT")
        np.random.seed(seed)
        x = np.random.uniform(0, 1, n)
        df = perform_statistical_tests(x, 2**32)
        p_values = second_level_testing(x, 2**32)

    else:
        print("Unknown generator type. Please choose from 'LCG', 'GLCG', 'RC4', or 'MT'.")
        return

    plot_p_values(p_values, generator)
    decisions = test_p_values(p_values, generator)
    return df, decisions


def monobit_test_manual(bit_sequence):
    x = 2 * np.array(bit_sequence) - 1
    n = len(x)

    s_obs = np.sum(x) / np.sqrt(n)
    p_value = 2 * (1 - stats.norm.cdf(abs(s_obs)))

    return s_obs, p_value


def split_into_subsets(data, num_subsets):
    subset_size = len(data) // num_subsets
    return [data[i * subset_size:(i + 1) * subset_size] for i in range(num_subsets)]


def perform_tests(datasets, subset_count):
    results = []
    second_level_pvalues = []

    for name, data in datasets.items():
        chi_squared_bf, p_value_bf = block_frequency_test(data, len(data) // 100)
        s_obs_mb, p_value_mb = monobit_test_manual(data)

        results.append([name, chi_squared_bf, p_value_bf, s_obs_mb, p_value_mb])

        subsets = split_into_subsets(data, subset_count)
        p_values_bf = []
        p_values_mb = []

        for subset in subsets:
            _, p_bf = block_frequency_test(subset, len(subset) // 100)
            _, p_mb = monobit_test_manual(subset)
            p_values_bf.append(p_bf)
            p_values_mb.append(p_mb)

        partition = np.linspace(0, 1, 11)
        counts, probs = compute_obs_in_bins(partition, p_values_bf)
        counts2, probs2 = compute_obs_in_bins(partition, p_values_mb)

        _, chi_pvalue_bf = np.round(stats.chisquare(counts, len(p_values_bf) * probs), 3)
        _, chi_pvalue_mb = np.round(stats.chisquare(counts2, len(p_values_mb) * probs2), 3)
        second_level_pvalues.append([name, chi_pvalue_bf, chi_pvalue_mb])

    results_df = pd.DataFrame(results, columns=["Number", "BFT Stat", "BFT pval", "FM Stat", "FM pval"])
    second_level_df = pd.DataFrame(second_level_pvalues, columns=["Number", "2nd Level BFT pval", "2nd Level FM pval"])

    return np.round(results_df, 3), np.round(second_level_df, 3)
