"""
================================================================================
📊 Generate Paper Assets: Auto-fill LaTeX Tables from Experiment Results
================================================================================

Reads all JSON result files and generates:
1. Publication-quality LaTeX tables
2. Statistical summary for paper

Run AFTER completing all experiments.

Author: 王唱晓
================================================================================
"""

import os
import sys
import json
import numpy as np
from collections import defaultdict


def load_json(path):
    if not os.path.exists(path):
        return None
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def generate_accuracy_table(results, output_dir):
    """Generate Table 2: Accuracy comparison across conditions."""
    if not results:
        print("  ⚠️ No rigorous results, skipping accuracy table.")
        return

    # Group by (dataset, condition), average across archs and seeds
    groups = defaultdict(list)
    for r in results:
        key = (r['dataset'], r['condition'])
        groups[key].append(r['final_test_acc'])

    # Also group by (arch, dataset, condition)
    detailed = defaultdict(list)
    for r in results:
        key = (r['arch'], r['dataset'], r['condition'])
        detailed[key].append(r['final_test_acc'])

    conditions = ['baseline', 'dropout', 'label_smooth', 'weight_decay', 'social_dropout']
    cond_labels = {
        'baseline': 'Baseline',
        'dropout': 'Std.~Dropout',
        'label_smooth': 'Label Smooth.',
        'weight_decay': 'Weight Decay',
        'social_dropout': '\\textbf{Social Dropout}',
    }

    # Overall table
    lines = []
    lines.append("\\begin{table}[h]")
    lines.append("\\centering")
    lines.append("\\caption{Test accuracy (\\%) across conditions. Mean $\\pm$ std over all architectures and seeds.}")
    lines.append("\\label{tab:accuracy}")
    lines.append("\\small")
    lines.append("\\begin{tabular}{lcc}")
    lines.append("\\toprule")
    lines.append("Condition & CIFAR-10 & CIFAR-100 \\\\")
    lines.append("\\midrule")

    for cond in conditions:
        c10 = groups.get(('cifar10', cond), [])
        c100 = groups.get(('cifar100', cond), [])
        c10_str = f"${np.mean(c10):.2f} \\pm {np.std(c10):.2f}$" if c10 else "--"
        c100_str = f"${np.mean(c100):.2f} \\pm {np.std(c100):.2f}$" if c100 else "--"
        lines.append(f"{cond_labels[cond]} & {c10_str} & {c100_str} \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")

    table_str = "\n".join(lines)
    path = os.path.join(output_dir, 'table_accuracy.tex')
    with open(path, 'w', encoding='utf-8') as f:
        f.write(table_str)
    print(f"  💾 Accuracy table saved: {path}")

    # Detailed table (per architecture)
    archs = ['SmallResNet', 'TinyVGG', 'CompactDenseNet']
    lines2 = []
    lines2.append("\\begin{table*}[t]")
    lines2.append("\\centering")
    lines2.append("\\caption{Per-architecture test accuracy (\\%) on CIFAR-10 / CIFAR-100. Mean $\\pm$ std over 3 seeds.}")
    lines2.append("\\label{tab:accuracy_detailed}")
    lines2.append("\\small")
    lines2.append("\\begin{tabular}{ll" + "c" * len(conditions) + "}")
    lines2.append("\\toprule")
    header = "Arch & Dataset & " + " & ".join(cond_labels[c] for c in conditions) + " \\\\"
    lines2.append(header)
    lines2.append("\\midrule")

    for arch in archs:
        for ds in ['cifar10', 'cifar100']:
            row = f"{arch} & {ds} "
            for cond in conditions:
                vals = detailed.get((arch, ds, cond), [])
                if vals:
                    row += f"& ${np.mean(vals):.2f}\\pm{np.std(vals):.2f}$ "
                else:
                    row += "& -- "
            row += "\\\\"
            lines2.append(row)
        if arch != archs[-1]:
            lines2.append("\\midrule")

    lines2.append("\\bottomrule")
    lines2.append("\\end{tabular}")
    lines2.append("\\end{table*}")

    path2 = os.path.join(output_dir, 'table_accuracy_detailed.tex')
    with open(path2, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines2))
    print(f"  💾 Detailed accuracy table saved: {path2}")


def generate_ttest_table(results, output_dir):
    """Generate paired t-test results table."""
    if not results:
        return
    from scipy.stats import ttest_rel

    conditions = ['baseline', 'dropout', 'label_smooth', 'weight_decay']
    cond_labels = {
        'baseline': 'Baseline',
        'dropout': 'Std.~Dropout',
        'label_smooth': 'Label Smooth.',
        'weight_decay': 'Weight Decay',
    }

    lines = []
    lines.append("\\begin{table}[h]")
    lines.append("\\centering")
    lines.append("\\caption{Paired $t$-test: Social Dropout vs.~each condition.}")
    lines.append("\\label{tab:ttest}")
    lines.append("\\small")
    lines.append("\\begin{tabular}{lcccl}")
    lines.append("\\toprule")
    lines.append("Comparison & $\\Delta$ Acc & $t$-stat & $p$-value & Sig. \\\\")
    lines.append("\\midrule")

    for cond in conditions:
        # Match by (arch, dataset, seed)
        pairs_a = {}
        pairs_b = {}
        for r in results:
            key = (r['arch'], r['dataset'], r['seed'])
            if r['condition'] == cond:
                pairs_a[key] = r['final_test_acc']
            elif r['condition'] == 'social_dropout':
                pairs_b[key] = r['final_test_acc']

        common = set(pairs_a.keys()) & set(pairs_b.keys())
        if len(common) < 2:
            continue

        a_vals = [pairs_a[k] for k in common]
        b_vals = [pairs_b[k] for k in common]
        t_stat, p_val = ttest_rel(b_vals, a_vals)
        diff = np.mean(b_vals) - np.mean(a_vals)

        sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "n.s."
        lines.append(f"vs.~{cond_labels[cond]} & ${diff:+.2f}$ & ${t_stat:.3f}$ & ${p_val:.4f}$ & {sig} \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")

    path = os.path.join(output_dir, 'table_ttest.tex')
    with open(path, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines))
    print(f"  💾 T-test table saved: {path}")


def generate_ensemble_table(results, output_dir):
    """Generate ensemble comparison table."""
    if not results:
        print("  ⚠️ No ensemble results, skipping.")
        return

    strategies = results.get('strategies', {})
    lines = []
    lines.append("\\begin{table}[h]")
    lines.append("\\centering")
    lines.append("\\caption{Ensemble accuracy (\\%) by selection strategy ($K=3$ from $N=8$).}")
    lines.append("\\label{tab:ensemble}")
    lines.append("\\small")
    lines.append("\\begin{tabular}{lcc}")
    lines.append("\\toprule")
    lines.append("Strategy & Majority Vote & Soft Vote \\\\")
    lines.append("\\midrule")

    labels = {
        'personality_diverse': '\\textbf{Personality-Diverse}',
        'best_individual': 'Best-Individual',
        'random_avg': 'Random (avg)',
        'all_ensemble': 'All 8 Models',
    }

    for key in ['personality_diverse', 'best_individual', 'random_avg', 'all_ensemble']:
        if key in strategies:
            s = strategies[key]
            maj = f"${s['majority']:.2f}$"
            if 'majority_std' in s:
                maj = f"${s['majority']:.2f}\\pm{s['majority_std']:.2f}$"
            soft = f"${s['soft']:.2f}$"
            if 'soft_std' in s:
                soft = f"${s['soft']:.2f}\\pm{s['soft_std']:.2f}$"
            lines.append(f"{labels.get(key, key)} & {maj} & {soft} \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")

    path = os.path.join(output_dir, 'table_ensemble.tex')
    with open(path, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines))
    print(f"  💾 Ensemble table saved: {path}")


def generate_validity_summary(results, output_dir):
    """Generate validity results summary."""
    if not results:
        print("  ⚠️ No validity results, skipping.")
        return

    lines = []
    lines.append("% Validity Summary (auto-generated)")

    if 'convergent' in results:
        anova = results['convergent']
        n_sig = sum(1 for v in anova.values() if isinstance(v, dict) and v.get('significant'))
        lines.append(f"% ANOVA: {n_sig}/5 traits significant (architecture dependence)")

    if 'independence' in results:
        ind = results['independence']
        lines.append(f"% Avg inter-trait |r| = {ind.get('avg_abs_corr', 'N/A')}")

    path = os.path.join(output_dir, 'validity_summary.tex')
    with open(path, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines))
    print(f"  💾 Validity summary saved: {path}")


def main():
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║                                                                      ║
║   📊 GENERATING PAPER ASSETS                                        ║
║                                                                      ║
║   Auto-generating LaTeX tables from experiment results               ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝
    """)

    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')

    # Load all results
    rigorous = load_json(os.path.join(results_dir, 'rigorous_results.json'))
    ensemble = load_json(os.path.join(results_dir, 'ensemble_results.json'))
    validity = load_json(os.path.join(results_dir, 'validity_results.json'))

    print(f"  📦 Rigorous: {'✅' if rigorous else '❌'} "
          f"({len(rigorous) if rigorous else 0} runs)")
    print(f"  📦 Ensemble: {'✅' if ensemble else '❌'}")
    print(f"  📦 Validity: {'✅' if validity else '❌'}")

    # Generate tables
    generate_accuracy_table(rigorous, results_dir)
    generate_ttest_table(rigorous, results_dir)
    generate_ensemble_table(ensemble, results_dir)
    generate_validity_summary(validity, results_dir)

    print(f"""
╔══════════════════════════════════════════════════════════════════════╗
║  ✅ PAPER ASSETS GENERATED                                           ║
║  📁 Tables: table_accuracy.tex, table_ttest.tex, table_ensemble.tex  ║
╚══════════════════════════════════════════════════════════════════════╝
    """)


if __name__ == '__main__':
    main()
