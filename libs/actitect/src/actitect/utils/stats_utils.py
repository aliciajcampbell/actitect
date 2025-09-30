import logging.config
from math import comb
from typing import Tuple

import numpy as np
import scipy.stats as stats

logger = logging.getLogger(__name__)

__all__ = [
    'independent_stat_significance_test',
    'compute_pearson_ci', 'compute_mean_std_ci',
    'sensitivity_specificity_from_cm',
    'fischer_freeman_hilton_exact_test', 'compute_composite_dataset_weights']


def independent_stat_significance_test(distribution_a: np.ndarray, distribution_b: np.ndarray, names=('a', 'b')):
    """Quantify the difference between two distributions. Assumes independence of both variables

    Parameters:
        :param distribution_a: (np.ndarray) of shape (n_samples_a,) representing the distribution A.
        :param distribution_b: (np.ndarray) of shape (n_samples_b,) representing the distribution B.
        :param names: (tuple of strings) names for the two distributions to store results. Optional, default: ('a', 'b')

    Returns:
        :return: (dict)
            Dictionary containing several metrics.

    Metrics:
        - normality: Tests whether both distributions follow a normal distribution.
            - shapiro stat/p: Shapiro-Wilk test stat [0,1] (normal to not) and p_value [0,1]
             (smaller=stronger rejection of normal hypothesis), might be inaccurate for large sample size.
            - kolmogorov stat/p: Kolmogorov-Smirnov test, same metrics as above but for stat 0 indicates normal
            - anderson stat/thres(5% cl): Anderson-Darling test, stat [0,inf) (smaller=normal) and 5% CL threshold
        - difference:
            - 'T-test': Independent t-test, suited to compare normal distributions.
                - stat/z: the difference in units of std,
                - p: p_value to reject null-hypothesis of no significance, i.e. smaller=more significant.
                    Typically:
                        p ≤   0.05:    *   -> less than 5% CL that distributions are not sign. different
                        p ≤   0.01:   **   ->    -"-    1% CL              -"-
                        p ≤  0.001:  ***   ->    -"-  0.1% CL              -"-
                        p ≤ 0.0001: ****   ->    -"- 0.01% CL              -"-
                - r: effect size: small (0.1) to large (0.5) effect of significance
    """
    # test if distributions are normal or not:
    results = {'normality': {}, 'difference': {}}
    for data, name in zip((distribution_a, distribution_b), names):
        shapiro_stat, shapiro_p_value = stats.shapiro(data)  # stat: 0 (non-normal) 1 (normal)
        k_stat, k_p_value = stats.kstest(data, 'norm',
                                         args=(data.mean(), data.std(ddof=1)))  # stat: 0 (normal) 1 (non-normal)
        ad_test = stats.anderson(data)
        ad_stat, critical_value_at_5_cl = ad_test.statistic, ad_test.critical_values[2]

        results['normality'].update({f"{name}": {
            'shapiro': {'stat': shapiro_stat, 'p': shapiro_p_value},
            'kolmogorov': {'stat': k_stat, 'p': k_p_value},
            'anderson': {'stat': ad_stat, 'thres_5': critical_value_at_5_cl},
        }})

    n1, n2 = len(distribution_a), len(distribution_b)

    # t-test for normal distributions:
    t_stat, p_val_t = stats.ttest_ind(distribution_a, distribution_b, equal_var=False)
    df = (n1 - 1) + (n2 - 1)
    r_t = (t_stat ** 2 / (t_stat ** 2 + df)) ** 0.5  # effect size: small (.1) - large (.5) difference
    results['difference'].update({
        'T-test': {'stat/z': t_stat, 'p': p_val_t, 'r': r_t}
    })

    # Mann-Whitney U (or Wilcoxon rank-sum) for arbitrary distribution:
    u_stat, p_val_u = stats.mannwhitneyu(distribution_a, distribution_b, alternative='two-sided')
    mean_u = n1 * n2 / 2
    std_u = np.sqrt(n1 * n2 * (n1 + n2 + 1) / 12)
    z = (u_stat - mean_u) / std_u  # z-score: significance in std units
    r_u = z / np.sqrt(n1 + n2)  # effect size: small (.1) - large (.5) difference

    results['difference'].update({
        'Mann-Whitney-U': {'stat': u_stat, 'p': p_val_u, 'z': z, 'r': r_u}
    })

    return results


def compute_pearson_ci(r: float, n: int, confidence_level: float = .95):
    """Compute confidence interval for Pearson correlation using Fisher z-transform."""
    assert n >= 10, f"must have at least 10 samples to estimate Fisher z-transform, got {n}."

    z = np.arctanh(r)  # Fisher z-transform
    se = 1 / np.sqrt(n - 3)  # standard error
    std_r = (1 - r ** 2) / np.sqrt(n - 3)  # estimation using delta method
    z_crit = stats.norm.ppf((1 + confidence_level) / 2)
    z_ci_lower = z - z_crit * se
    z_ci_upper = z + z_crit * se
    return np.tanh(z_ci_lower), np.tanh(z_ci_upper), std_r


def compute_mean_std_ci(data: np.ndarray, confidence_level: float):
    """ Compute the mean, standard deviation, and confidence interval for the mean of a dataset.
    This function calculates:
      - The mean of the input data
      - The sample standard deviation (using N−1 in the denominator)
      - A confidence interval (CI) around the mean, assuming a t-distribution
    Parameters
        :param data: (np.ndarray) The input data array for which the statistics will be computed.
        :param confidence_level: (float) The desired confidence level (e.g., 0.95 for a 95% confidence interval).

    Returns
        :return mean_val: (float) The sample mean of the data.
        :return std_val: (float) The sample standard deviation of the data (ddof=1).
        :return ci_lower: (float or nan)  The lower bound of the confidence interval around the mean.
            Returns nan if n <= 1.
        :return ci_upper: (float or nan) The upper bound of the confidence interval around the mean.
            Returns nan if n <= 1.
        :return margin: (float or nan)  The margin of error used to compute the confidence interval.
            Returns nan if n <= 1. """
    n = len(data)
    mean_val, std_val = np.mean(data), np.std(data, ddof=1)
    margin, ci_lower, ci_upper = np.nan, np.nan, np.nan
    if n > 1:
        sem = stats.sem(data)  # Standard Error of the Mean
        margin = sem * stats.t.ppf((1 + confidence_level) / 2., n - 1)
        ci_lower = mean_val - margin
        ci_upper = mean_val + margin
    return mean_val, std_val, ci_lower, ci_upper, margin


def sensitivity_specificity_from_cm(cm: np.ndarray) -> Tuple[float, float]:
    """ Compute sensitivity (recall) and specificity from a 2×2 confusion matrix"""
    cm = np.asarray(cm)
    if cm.shape != (2, 2):
        raise ValueError(f"Expected 2×2 matrix, got shape {cm.shape}")
    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    specificity = tn / (tn + fp) if (tn + fp) > 0 else np.nan
    return sensitivity, specificity


def fischer_freeman_hilton_exact_test(table: np.ndarray) -> float:
    """ Exact two-sided Fisher–Freeman–Halton test for an R×2 contingency table.
    The p-value is the sum of probabilities of all tables with the same row totals
    and first-column total whose hypergeometric probability is ≤ that of the observed
    table. Implemented via depth-first enumeration with integer-only comparisons,
    early pruning, and row reordering for speed.
    Parameters
        :param table: (array_like), shape (R, 2) Non-negative integer counts.
    Returns
        :return: (float) Two sided p-value. """
    T = np.asarray(table, dtype=int)
    if T.ndim != 2 or T.shape[1] != 2 or T.size == 0:
        raise ValueError("`table` must be an R×2 array with non-negative integers.")
    if (T < 0).any():
        raise ValueError("Counts must be non-negative.")

    R = T.shape[0]
    row_tot = T.sum(axis=1)
    col1_tot = int(T[:, 0].sum())
    N = int(row_tot.sum())

    # Visit larger rows first → stronger pruning.
    order = np.argsort(-row_tot)
    row_tot = row_tot[order]
    col1_obs = T[:, 0][order].tolist()

    # Precompute C(n,k) per row.
    C = [[comb(row_tot[i], k) for k in range(row_tot[i] + 1)] for i in range(R)]

    # Observed numerator (denominator cancels when comparing probabilities).
    num_obs = 1
    for i, xi in enumerate(col1_obs):
        num_obs *= C[i][xi]
    total_num = 0  # accumulated numerator mass over allocations ≤ observed

    def _dfs(i: int, remaining: int, num_so_far: int) -> None:
        nonlocal total_num
        if i == R - 1:
            xi = remaining
            if 0 <= xi <= row_tot[i]:
                num = num_so_far * C[i][xi]
                if num <= num_obs:
                    total_num += num
            return

        # Feasible range for this row.
        tail = int(row_tot[i + 1:].sum())
        lo = max(0, remaining - tail)
        hi = min(row_tot[i], remaining)

        # Try ks in order of decreasing C(n,k) to trigger pruning earlier.
        ks = range(lo, hi + 1)
        ks = sorted(ks, key=lambda k: -C[i][k]) if lo != hi else ks
        for xi in ks:
            num_next = num_so_far * C[i][xi]
            if num_next > num_obs:  # prune branch
                continue
            _dfs(i + 1, remaining - xi, num_next)

    _dfs(0, col1_tot, 1)

    # Convert summed numerators to probability once.
    return total_num / comb(N, col1_tot)


def compute_composite_dataset_weights(ds_vec: np.ndarray, y_vec: np.ndarray, mode: str):
    if mode is None or ds_vec is None:
        return None
    if mode == 'dsw':
        N = len(ds_vec)
        uniq, counts = np.unique(ds_vec, return_counts=True)
        per_ds = dict(zip(uniq, counts))
        base = N / len(uniq)
        w = np.array([base / per_ds[d] for d in ds_vec], dtype=float)
        return w * (N / w.sum())
    elif mode == 'dswc':
        N = len(ds_vec)
        cells, counts = np.unique(np.stack([ds_vec, y_vec], axis=1), axis=0, return_counts=True)
        per_cell = {(d, int(c)): n for (d, c), n in zip(map(tuple, cells), counts)}
        G = len(np.unique(ds_vec))
        C = len(np.unique(y_vec))
        base = N / (G * C)
        w = np.array([base / per_cell[(d, int(y_))] for d, y_ in zip(ds_vec, y_vec)], dtype=float)
        return w * (N / w.sum())
    else:
        raise ValueError(f"Unknown mode {mode}")
