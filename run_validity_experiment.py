"""
================================================================================
🔍 Trait Validity Experiment
================================================================================

Validates that NBFA traits measure meaningful, distinct properties:

1. Discriminant Validity: Partial correlations controlling for model size
2. Predictive Validity: Traits predict generalization gap & robustness
3. Convergent Validity: Within-family variance < between-family variance (ANOVA)

Uses results from run_rigorous_experiment.py.

Author: 王唱晓
================================================================================
"""

import os
import sys
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, f_oneway, spearmanr
from collections import defaultdict

matplotlib.rcParams['font.family'] = ['Microsoft YaHei', 'SimHei', 'sans-serif']


def load_rigorous_results(results_dir):
    path = os.path.join(results_dir, 'rigorous_results.json')
    if not os.path.exists(path):
        print("❌ rigorous_results.json not found. Run run_rigorous_experiment.py first.")
        sys.exit(1)
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_nbfa_results(results_dir):
    path = os.path.join(results_dir, 'nbfa_results.json')
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


# ============================================================================
# 1. Discriminant Validity: Traits are not just proxies for model size
# ============================================================================

def partial_correlation(x, y, z):
    """Partial correlation of x and y, controlling for z."""
    # Residualize x on z
    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)
    z = np.array(z, dtype=float)

    if len(x) < 4:
        return 0.0, 1.0

    # Linear regression residuals
    z_mean = z - z.mean()
    z_var = np.dot(z_mean, z_mean)
    if z_var < 1e-10:
        return pearsonr(x, y)

    x_resid = x - z_mean * np.dot(x - x.mean(), z_mean) / z_var - x.mean()
    y_resid = y - z_mean * np.dot(y - y.mean(), z_mean) / z_var - y.mean()

    r, p = pearsonr(x_resid, y_resid)
    return float(r), float(p)


def discriminant_validity(results, output_dir):
    """Show traits predict accuracy beyond model size."""
    print("\n" + "="*60)
    print("🔬 Discriminant Validity: Controlling for Model Size")
    print("="*60)

    # Use 8-model pretrained results
    nbfa = load_nbfa_results(output_dir)
    models = list(nbfa.keys())

    # Known param counts
    param_counts = {m: nbfa[m].get('total_params', 0) for m in models}
    accuracies = {
        'VGG-16': 71.59, 'ResNet-50': 76.13, 'ResNet-152': 78.31,
        'DenseNet-121': 74.43, 'EfficientNet-B0': 77.69,
        'ConvNeXt-Tiny': 82.52, 'ViT-Small': 81.07, 'ViT-Base': 75.87,
    }

    valid_models = [m for m in models if m in accuracies]
    if len(valid_models) < 4:
        print("  ⚠️ Not enough models for analysis.")
        return {}

    accs = [accuracies[m] for m in valid_models]
    params = [param_counts[m] for m in valid_models]

    traits = ['E', 'N', 'O', 'A', 'C']
    trait_names = ['Extraversion', 'Neuroticism', 'Openness', 'Agreeableness', 'Conscientiousness']

    print(f"\n  {'Trait':<20} {'r(trait,acc)':<15} {'r_partial':<15} {'p_partial':<10}")
    print(f"  {'-'*60}")

    partial_results = {}
    for t, tname in zip(traits, trait_names):
        trait_vals = [nbfa[m]['raw_scores'][t] for m in valid_models]

        r_raw, p_raw = pearsonr(trait_vals, accs)
        r_partial, p_partial = partial_correlation(trait_vals, accs, params)

        partial_results[t] = {
            'r_raw': float(r_raw), 'p_raw': float(p_raw),
            'r_partial': float(r_partial), 'p_partial': float(p_partial),
        }

        sig = '*' if p_partial < 0.1 else 'n.s.'
        print(f"  {tname:<20} {r_raw:>+8.3f} (p={p_raw:.3f})  "
              f"{r_partial:>+8.3f} (p={p_partial:.3f}) {sig}")

    return partial_results


# ============================================================================
# 2. Predictive Validity: Traits predict gen gap & robustness
# ============================================================================

def predictive_validity(results, output_dir):
    """Correlate personality traits with generalization gap and robustness."""
    print("\n" + "="*60)
    print("🔬 Predictive Validity: Traits → Generalization & Robustness")
    print("="*60)

    if not results:
        print("  ⚠️ No rigorous results available.")
        return {}

    traits = ['E', 'N', 'O', 'A', 'C']
    trait_names = ['Extraversion', 'Neuroticism', 'Openness', 'Agreeableness', 'Conscientiousness']

    # Extract data from rigorous experiments
    trait_vals = {t: [] for t in traits}
    gen_gaps = []
    robust_drops = []
    accs = []

    for r in results:
        for t in traits:
            trait_vals[t].append(r['personality'][t])
        gen_gaps.append(r['gen_gap'])
        robust_drops.append(r['robustness_drop'])
        accs.append(r['final_test_acc'])

    # Correlations
    pred_results = {}
    print(f"\n  {'Trait':<20} {'r(gap)':<12} {'r(robust)':<12} {'r(acc)':<12}")
    print(f"  {'-'*56}")

    for t, tname in zip(traits, trait_names):
        x = np.array(trait_vals[t])
        r_gap, p_gap = pearsonr(x, gen_gaps) if len(x) > 2 else (0, 1)
        r_rob, p_rob = pearsonr(x, robust_drops) if len(x) > 2 else (0, 1)
        r_acc, p_acc = pearsonr(x, accs) if len(x) > 2 else (0, 1)

        pred_results[t] = {
            'r_gen_gap': float(r_gap), 'p_gen_gap': float(p_gap),
            'r_robustness': float(r_rob), 'p_robustness': float(p_rob),
            'r_accuracy': float(r_acc), 'p_accuracy': float(p_acc),
        }

        sig_gap = '*' if p_gap < 0.05 else ''
        sig_rob = '*' if p_rob < 0.05 else ''
        sig_acc = '*' if p_acc < 0.05 else ''
        print(f"  {tname:<20} {r_gap:>+.3f}{sig_gap:<4} {r_rob:>+.3f}{sig_rob:<4} {r_acc:>+.3f}{sig_acc}")

    # Plot: Neuroticism vs Generalization Gap (expected strong positive correlation)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), facecolor='#0a0a1a')
    plot_pairs = [
        ('N', gen_gaps, 'Generalization Gap (%)', 'Neuroticism → Overfitting'),
        ('E', accs, 'Test Accuracy (%)', 'Extraversion → Performance'),
        ('C', robust_drops, 'Robustness Drop (%)', 'Conscientiousness → Robustness'),
    ]

    for ax, (trait, y_vals, ylabel, title) in zip(axes, plot_pairs):
        ax.set_facecolor('#0a0a1a')
        x = np.array(trait_vals[trait])
        y = np.array(y_vals)

        ax.scatter(x, y, c='#4ECDC4', s=30, alpha=0.5, edgecolors='white', linewidth=0.3)

        # Trend line
        if len(x) > 2:
            z = np.polyfit(x, y, 1)
            x_line = np.linspace(x.min(), x.max(), 100)
            ax.plot(x_line, np.polyval(z, x_line), '--', color='#FF6B6B', linewidth=2)
            r, p = pearsonr(x, y)
            ax.set_title(f'{title}\nr={r:.3f}, p={p:.4f}',
                        color='white', fontsize=11, fontweight='bold')
        else:
            ax.set_title(title, color='white', fontsize=11, fontweight='bold')

        ax.set_xlabel(f'Trait {trait}', color='white')
        ax.set_ylabel(ylabel, color='white')
        ax.tick_params(colors='white')
        ax.spines['bottom'].set_color('white')
        ax.spines['left'].set_color('white')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    plt.suptitle('Predictive Validity: Personality Traits as Network Diagnostics',
                fontsize=14, fontweight='bold', color='white', y=1.02)
    plt.tight_layout()
    path = os.path.join(output_dir, 'predictive_validity.png')
    plt.savefig(path, dpi=200, bbox_inches='tight', facecolor='#0a0a1a')
    plt.close()
    print(f"\n  💾 Predictive validity plot saved: {path}")

    return pred_results


# ============================================================================
# 3. Convergent Validity: ANOVA — within-arch vs between-arch variance
# ============================================================================

def convergent_validity(results, output_dir):
    """One-way ANOVA: do architectures cluster in personality space?"""
    print("\n" + "="*60)
    print("🔬 Convergent Validity: Architecture → Personality (ANOVA)")
    print("="*60)

    if not results:
        print("  ⚠️ No rigorous results available.")
        return {}

    traits = ['E', 'N', 'O', 'A', 'C']
    trait_names = ['Extraversion', 'Neuroticism', 'Openness', 'Agreeableness', 'Conscientiousness']

    # Group by architecture (only baseline condition to avoid confounds)
    arch_groups = defaultdict(lambda: {t: [] for t in traits})
    for r in results:
        if r['condition'] == 'baseline':
            for t in traits:
                arch_groups[r['arch']][t].append(r['personality'][t])

    archs = list(arch_groups.keys())
    if len(archs) < 2:
        print("  ⚠️ Need ≥2 architectures for ANOVA.")
        return {}

    anova_results = {}
    print(f"\n  {'Trait':<20} {'F-statistic':<15} {'p-value':<10} {'Sig':<5} {'η²':<8}")
    print(f"  {'-'*58}")

    for t, tname in zip(traits, trait_names):
        groups = [arch_groups[a][t] for a in archs]
        groups = [g for g in groups if len(g) >= 2]  # Need ≥2 obs per group

        if len(groups) < 2:
            continue

        f_stat, p_value = f_oneway(*groups)

        # Effect size (eta squared)
        all_vals = [v for g in groups for v in g]
        grand_mean = np.mean(all_vals)
        ss_between = sum(len(g) * (np.mean(g) - grand_mean)**2 for g in groups)
        ss_total = sum((v - grand_mean)**2 for v in all_vals)
        eta_sq = ss_between / ss_total if ss_total > 0 else 0

        sig = '***' if p_value < 0.001 else '**' if p_value < 0.01 else '*' if p_value < 0.05 else 'n.s.'

        anova_results[t] = {
            'f_stat': float(f_stat), 'p_value': float(p_value),
            'eta_squared': float(eta_sq), 'significant': p_value < 0.05,
        }

        print(f"  {tname:<20} {f_stat:>10.3f}    {p_value:>8.4f}   {sig:<5} {eta_sq:.3f}")

    # Interpretation
    n_sig = sum(1 for v in anova_results.values() if v['significant'])
    print(f"\n  📊 {n_sig}/5 traits show significant architecture dependence.")
    if n_sig >= 3:
        print("  ✅ Strong convergent validity: personality profiles cluster by architecture.")
    elif n_sig >= 1:
        print("  ⚠️ Partial convergent validity: some traits are architecture-dependent.")
    else:
        print("  ❌ Weak convergent validity: traits may not reliably distinguish architectures.")

    return anova_results


# ============================================================================
# 4. Trait Independence: Are the 5 traits orthogonal?
# ============================================================================

def trait_independence(results, output_dir):
    """Check that traits are not redundant (low inter-trait correlation)."""
    print("\n" + "="*60)
    print("🔬 Trait Independence: Inter-trait Correlations")
    print("="*60)

    if not results:
        print("  ⚠️ No results available.")
        return {}

    traits = ['E', 'N', 'O', 'A', 'C']

    # Collect all trait values
    scores = np.array([[r['personality'][t] for t in traits] for r in results])

    # Correlation matrix
    n_traits = len(traits)
    corr = np.corrcoef(scores.T)

    print(f"\n         {'E':>8} {'N':>8} {'O':>8} {'A':>8} {'C':>8}")
    for i, t in enumerate(traits):
        row = "  " + f"{t:<5}"
        for j in range(n_traits):
            if i == j:
                row += f"{'1.000':>8}"
            else:
                row += f"{corr[i,j]:>8.3f}"
        print(row)

    # Average absolute off-diagonal correlation
    off_diag = []
    for i in range(n_traits):
        for j in range(i+1, n_traits):
            off_diag.append(abs(corr[i, j]))

    avg_corr = np.mean(off_diag)
    max_corr = np.max(off_diag)

    print(f"\n  Average |r| (off-diagonal): {avg_corr:.3f}")
    print(f"  Maximum |r|: {max_corr:.3f}")

    if avg_corr < 0.3:
        print("  ✅ Traits are largely independent (avg |r| < 0.3)")
    elif avg_corr < 0.5:
        print("  ⚠️ Moderate inter-trait correlation (avg |r| < 0.5)")
    else:
        print("  ❌ High inter-trait correlation — some traits may be redundant")

    # Plot
    fig, ax = plt.subplots(figsize=(7, 6), facecolor='#0a0a1a')
    ax.set_facecolor('#0a0a1a')

    im = ax.imshow(corr, cmap='RdBu_r', vmin=-1, vmax=1)
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Pearson r', color='white')
    cbar.ax.tick_params(colors='white')

    labels = ['E\n外向性', 'N\n神经质', 'O\n开放性', 'A\n宜人性', 'C\n尽责性']
    ax.set_xticks(range(5))
    ax.set_xticklabels(labels, color='white', fontsize=9)
    ax.set_yticks(range(5))
    ax.set_yticklabels(labels, color='white', fontsize=9)

    for i in range(5):
        for j in range(5):
            color = 'white' if abs(corr[i,j]) < 0.5 else 'black'
            ax.text(j, i, f'{corr[i,j]:.2f}', ha='center', va='center',
                   color=color, fontsize=11, fontweight='bold')

    ax.set_title('Trait Independence Matrix\n(Rigorous Experiment Data)',
                color='white', fontsize=13, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(output_dir, 'trait_independence.png')
    plt.savefig(path, dpi=200, bbox_inches='tight', facecolor='#0a0a1a')
    plt.close()
    print(f"\n  💾 Independence matrix saved: {path}")

    return {'avg_abs_corr': float(avg_corr), 'max_abs_corr': float(max_corr),
            'correlation_matrix': corr.tolist()}


# ============================================================================
# Main
# ============================================================================

def main():
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║                                                                      ║
║   🔍 TRAIT VALIDITY EXPERIMENT                                      ║
║                                                                      ║
║   1. Discriminant: Not just model size                               ║
║   2. Predictive: Traits → generalization & robustness                ║
║   3. Convergent: Architecture → personality cluster                  ║
║   4. Independence: Low inter-trait correlation                       ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝
    """)

    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')

    # Load rigorous results (if available)
    rig_path = os.path.join(output_dir, 'rigorous_results.json')
    rig_results = []
    if os.path.exists(rig_path):
        with open(rig_path, 'r', encoding='utf-8') as f:
            rig_results = json.load(f)
        print(f"  📦 Loaded {len(rig_results)} rigorous experiment results.")
    else:
        print("  ⚠️ No rigorous results yet — will use pretrained model data only.")

    all_validity = {}

    # 1. Discriminant validity (uses pretrained 8-model data)
    all_validity['discriminant'] = discriminant_validity(rig_results, output_dir)

    # 2. Predictive validity (uses rigorous experiment data)
    all_validity['predictive'] = predictive_validity(rig_results, output_dir)

    # 3. Convergent validity (ANOVA)
    all_validity['convergent'] = convergent_validity(rig_results, output_dir)

    # 4. Trait independence
    all_validity['independence'] = trait_independence(rig_results, output_dir)

    # Save
    path = os.path.join(output_dir, 'validity_results.json')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(all_validity, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n💾 Validity results saved: {path}")

    print(f"""
╔══════════════════════════════════════════════════════════════════════╗
║  ✅ VALIDITY EXPERIMENT COMPLETE                                     ║
╠══════════════════════════════════════════════════════════════════════╣
║  See: predictive_validity.png, trait_independence.png                ║
╚══════════════════════════════════════════════════════════════════════╝
    """)


if __name__ == '__main__':
    main()
