"""
Statistical Analysis for ADD-PINN Paper (Appendix)
===================================================
Reproduces all statistical tests for I-24 MOTION head-to-head comparisons.

Input:  per_seed_all.csv (2108 rows, all methods × all datasets × all seeds)
Output: LaTeX table for Appendix

Analysis unit: configuration mean (dataset × sensor count), n = 25
Each configuration mean is averaged over 10 random seeds.
This avoids pseudo-replication from correlated seeds.

Tests performed:
  1. Paired t-test (one-sided)
  2. Wilcoxon signed-rank test (non-parametric robustness check)
  3. Holm-Bonferroni correction (multiple comparison adjustment, 5 tests)
  4. Sign test / Binomial test (on W/L counts)
  5. Paired Cohen's d (effect size)
  6. 95% confidence interval for mean improvement

Code-to-Paper baseline ID mapping:
  code B2 -> paper B1 (NN)
  code B3 -> paper B2 (Vanilla PINN)
  code B4 -> paper B3 (PINN + RAR)
  code B5 -> paper B4 (PINN + Viscosity)
  code B7 -> paper B6 (ADD-PINN, Ours)
  code B8 -> paper B5 (XPINN)
"""

import pandas as pd
import numpy as np
from scipy import stats

# ---- Load data ----
df = pd.read_csv('per_seed_all.csv')
i24_datasets = ['20221121', '20221122', '20221123', '20221129', '20221202']
code_methods = ['B2', 'B3', 'B4', 'B5', 'B7', 'B8']

df_i24 = df[
    (df['Dataset'].astype(str).isin(i24_datasets)) &
    (df['Method'].isin(code_methods))
]

# ---- Step 1: Compute 25 configuration means ----
config_means = {}
for ds in i24_datasets:
    for s in [3, 4, 5, 6, 7]:
        config_means[(ds, s)] = {}
        for m in code_methods:
            vals = df_i24[
                (df_i24['Dataset'].astype(str) == ds) &
                (df_i24['Method'] == m) &
                (df_i24['Sensors'] == s)
            ]['L2']
            if len(vals) > 0:
                config_means[(ds, s)][m] = vals.mean()

# ---- Step 2: Paired comparisons ----
# (code_id, paper_id, display_name)
comparisons = [
    ('B2', 'B1', 'NN'),
    ('B3', 'B2', 'Vanilla PINN'),
    ('B4', 'B3', 'PINN + RAR'),
    ('B5', 'B4', 'PINN + Viscosity'),
    ('B8', 'B5', 'XPINN'),
]

results = []
p_values_raw = []

for m_code, m_paper_id, m_name in comparisons:
    diffs = []
    wins = losses = 0

    for key in sorted(config_means.keys()):
        b7_val = config_means[key]['B7']
        m_val = config_means[key][m_code]
        d = m_val - b7_val  # positive = B7 (ADD-PINN) is better
        diffs.append(d)
        if b7_val < m_val:
            wins += 1
        else:
            losses += 1

    diffs = np.array(diffs)
    n = len(diffs)
    mean_diff = np.mean(diffs)
    std_diff = np.std(diffs, ddof=1)
    se_diff = std_diff / np.sqrt(n)

    # Test 1: Paired t-test (one-sided: H1 = B7 is better)
    t_stat, p_ttest_two = stats.ttest_rel(
        [config_means[k][m_code] for k in sorted(config_means)],
        [config_means[k]['B7'] for k in sorted(config_means)]
    )
    p_ttest = p_ttest_two / 2 if t_stat > 0 else 1 - p_ttest_two / 2

    # Test 2: Wilcoxon signed-rank (non-parametric)
    w_stat, p_wilcoxon = stats.wilcoxon(diffs, alternative='greater')

    # Test 3: Sign test (binomial)
    p_sign = stats.binomtest(
        wins, wins + losses, 0.5, alternative='greater'
    ).pvalue

    # Effect size: paired Cohen's d
    d_paired = mean_diff / std_diff

    # 95% CI for mean improvement
    t_crit = stats.t.ppf(0.975, df=n - 1)
    ci_low = mean_diff - t_crit * se_diff
    ci_high = mean_diff + t_crit * se_diff

    p_values_raw.append(p_ttest)
    results.append({
        'code': m_code,
        'paper_id': m_paper_id,
        'name': m_name,
        'n': n,
        'wins': wins,
        'losses': losses,
        'mean_diff_pp': mean_diff * 100,
        'ci_low_pp': ci_low * 100,
        'ci_high_pp': ci_high * 100,
        't_stat': t_stat,
        'p_ttest': p_ttest,
        'p_wilcoxon': p_wilcoxon,
        'p_sign': p_sign,
        'd_paired': d_paired,
    })

# ---- Step 3: Holm-Bonferroni correction ----
m_total = len(p_values_raw)
sorted_indices = sorted(range(m_total), key=lambda i: p_values_raw[i])
p_holm = [None] * m_total
for rank, idx in enumerate(sorted_indices):
    adjusted = p_values_raw[idx] * (m_total - rank)
    p_holm[idx] = min(adjusted, 1.0)
# Enforce monotonicity
for rank in range(1, m_total):
    idx = sorted_indices[rank]
    prev_idx = sorted_indices[rank - 1]
    p_holm[idx] = max(p_holm[idx], p_holm[prev_idx])

for i, r in enumerate(results):
    r['p_holm'] = p_holm[i]

# ---- Step 4: Print results ----
print("=" * 90)
print("STATISTICAL ANALYSIS: ADD-PINN (B6) vs Baselines on I-24 MOTION")
print("=" * 90)
print(f"Analysis unit: configuration mean (dataset x sensor count), n = 25")
print(f"Each configuration mean is computed over 10 random seeds\n")

for r in results:
    print(f"--- B6 vs {r['paper_id']} ({r['name']}) ---")
    print(f"  Win/Loss:              {r['wins']}/{r['losses']}")
    print(f"  Mean delta L2:         {r['mean_diff_pp']:+.2f} pp")
    print(f"  95% CI:                [{r['ci_low_pp']:+.2f}, {r['ci_high_pp']:+.2f}] pp")
    print(f"  Paired t-test:         t = {r['t_stat']:.3f}, p = {r['p_ttest']:.6f}")
    print(f"  Wilcoxon signed-rank:  p = {r['p_wilcoxon']:.6f}")
    print(f"  Holm-corrected p:      p = {r['p_holm']:.6f}")
    print(f"  Sign test (binomial):  p = {r['p_sign']:.6f}")
    print(f"  Paired Cohen's d:      {r['d_paired']:.2f}")
    print()

# ---- Step 5: Effect size interpretation ----
print("EFFECT SIZE INTERPRETATION (paired Cohen's d):")
for r in results:
    if r['d_paired'] >= 0.8:
        interp = "large"
    elif r['d_paired'] >= 0.5:
        interp = "medium"
    else:
        interp = "small"
    print(f"  B6 vs {r['paper_id']}: d = {r['d_paired']:.2f} ({interp})")
