"""
================================================================================
🎨 Extra Visualizations + Experiment 2: Personality-Performance Correlation
================================================================================

Generates:
1. Cross-layer Wasserstein distance heatmap (per model)
2. Personality-Performance correlation (∩ curve for Extraversion)
3. Architecture family personality clustering
4. Per-trait Pearson correlation table

Uses known ImageNet Top-1 accuracy for all 8 models.

Usage:
    python extra_viz.py

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
from scipy.stats import pearsonr
from scipy.linalg import svdvals
from scipy.optimize import curve_fit

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from neural_personality import wasserstein_1d

matplotlib.rcParams['font.family'] = ['Microsoft YaHei', 'SimHei', 'sans-serif']

# ============================================================================
# Known ImageNet Top-1 Accuracy (official torchvision numbers)
# ============================================================================
IMAGENET_ACC = {
    'VGG-16':          71.59,
    'ResNet-50':       76.13,
    'ResNet-152':      78.31,
    'DenseNet-121':    74.43,
    'EfficientNet-B0': 77.69,
    'ConvNeXt-Tiny':   82.52,
    'ViT-Small':       81.07,  # ViT-B/16
    'ViT-Base':        75.87,  # ViT-B/32
}

# Architecture families for grouping
ARCH_FAMILY = {
    'VGG-16':          'Plain CNN',
    'ResNet-50':       'ResNet',
    'ResNet-152':      'ResNet',
    'DenseNet-121':    'DenseNet',
    'EfficientNet-B0': 'EfficientNet',
    'ConvNeXt-Tiny':   'ConvNeXt',
    'ViT-Small':       'Transformer',
    'ViT-Base':        'Transformer',
}

FAMILY_COLORS = {
    'Plain CNN':     '#FF6B6B',
    'ResNet':        '#4ECDC4',
    'DenseNet':      '#45B7D1',
    'EfficientNet':  '#96CEB4',
    'ConvNeXt':      '#DDA0DD',
    'Transformer':   '#F7DC6F',
}


def load_results(results_dir):
    """Load NBFA results from JSON."""
    path = os.path.join(results_dir, 'nbfa_results.json')
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


# ============================================================================
# 1. Wasserstein Cross-Layer Distance Heatmap
# ============================================================================

def plot_wasserstein_heatmap(model, model_name, output_dir):
    """
    Compute and plot the pairwise Wasserstein distance matrix
    between all weight layers of a model.
    """
    import torch

    weight_layers = []
    layer_names = []

    for name, param in model.named_parameters():
        if 'weight' in name and param.dim() >= 2:
            w = param.detach().cpu().numpy().ravel()
            if len(w) > 100:
                weight_layers.append(w)
                # Shorten name
                short = name.replace('.weight', '').replace('features.', 'f.')
                if len(short) > 15:
                    short = short[:12] + '...'
                layer_names.append(short)

    n = len(weight_layers)
    if n < 3:
        return

    # Subsample if too many layers
    if n > 30:
        indices = np.linspace(0, n-1, 30, dtype=int)
        weight_layers = [weight_layers[i] for i in indices]
        layer_names = [layer_names[i] for i in indices]
        n = len(weight_layers)

    # Compute pairwise Wasserstein matrix
    W_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            w = wasserstein_1d(weight_layers[i], weight_layers[j])
            W_matrix[i, j] = w
            W_matrix[j, i] = w

    # Plot
    fig, ax = plt.subplots(figsize=(10, 8), facecolor='#0a0a1a')
    ax.set_facecolor('#0a0a1a')

    im = ax.imshow(W_matrix, cmap='magma', interpolation='nearest')
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Wasserstein-2 Distance', color='white', fontsize=11)
    cbar.ax.tick_params(colors='white')

    # Labels (show every Nth)
    step = max(1, n // 15)
    ax.set_xticks(range(0, n, step))
    ax.set_xticklabels([layer_names[i] for i in range(0, n, step)],
                       rotation=45, ha='right', fontsize=7, color='white')
    ax.set_yticks(range(0, n, step))
    ax.set_yticklabels([layer_names[i] for i in range(0, n, step)],
                       fontsize=7, color='white')

    ax.set_title(f'Cross-Layer Wasserstein Distance Matrix\n{model_name}',
                color='white', fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('Layer Index', color='white', fontsize=11)
    ax.set_ylabel('Layer Index', color='white', fontsize=11)

    plt.tight_layout()
    safe_name = model_name.replace(' ', '_').replace('/', '_')
    path = os.path.join(output_dir, f'wasserstein_heatmap_{safe_name}.png')
    plt.savefig(path, dpi=200, bbox_inches='tight', facecolor='#0a0a1a')
    plt.close()
    print(f"  💾 Heatmap saved: {path}")


# ============================================================================
# 2. Personality-Performance Correlation (Experiment 2)
# ============================================================================

def plot_personality_performance_correlation(results, output_dir):
    """
    Plot Pearson correlation between each personality trait and ImageNet accuracy.
    Key finding: Extraversion shows ∩-shaped (inverted U) relationship.
    """
    models = list(results.keys())
    models = [m for m in models if m in IMAGENET_ACC]

    if len(models) < 4:
        print("  ⚠️ Not enough models with known accuracy for correlation analysis.")
        return

    # Extract data
    accuracies = np.array([IMAGENET_ACC[m] for m in models])
    traits = ['E', 'N', 'O', 'A', 'C']
    trait_names = ['Extraversion', 'Neuroticism', 'Openness',
                   'Agreeableness', 'Conscientiousness']
    trait_names_cn = ['外向性', '神经质', '开放性', '宜人性', '尽责性']

    raw_scores = {}
    for t in traits:
        raw_scores[t] = np.array([results[m]['raw_scores'][t] for m in models])

    # ================================================================
    # Plot 1: All 5 traits vs Accuracy (scatter + Pearson r)
    # ================================================================
    fig, axes = plt.subplots(1, 5, figsize=(25, 5), facecolor='#0a0a1a')

    correlations = {}

    for idx, (trait, tname, tcn) in enumerate(zip(traits, trait_names, trait_names_cn)):
        ax = axes[idx]
        ax.set_facecolor('#0a0a1a')

        x = raw_scores[trait]
        y = accuracies

        # Color by architecture family
        for i, m in enumerate(models):
            family = ARCH_FAMILY.get(m, 'Other')
            color = FAMILY_COLORS.get(family, '#FFFFFF')
            ax.scatter(x[i], y[i], c=color, s=100, edgecolors='white',
                      linewidth=1, zorder=5)
            ax.annotate(m, (x[i], y[i]), fontsize=6, color='white',
                       xytext=(5, 5), textcoords='offset points')

        # Pearson correlation
        r, p = pearsonr(x, y)
        correlations[trait] = {'r': r, 'p': p}

        # Linear fit
        if len(x) > 2:
            z = np.polyfit(x, y, 1)
            x_line = np.linspace(min(x), max(x), 100)
            ax.plot(x_line, np.polyval(z, x_line), '--', color='#FF6B6B',
                   alpha=0.7, linewidth=1.5)

        ax.set_xlabel(f'{tname}\n({tcn})', color='white', fontsize=10)
        if idx == 0:
            ax.set_ylabel('ImageNet Top-1 Acc (%)', color='white', fontsize=10)

        r_color = '#2ECC71' if abs(r) > 0.5 else '#F39C12' if abs(r) > 0.3 else '#E74C3C'
        ax.set_title(f'r = {r:.3f} (p = {p:.3f})', color=r_color,
                    fontsize=11, fontweight='bold')

        ax.tick_params(colors='white')
        ax.spines['bottom'].set_color('white')
        ax.spines['left'].set_color('white')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    plt.suptitle('Experiment 2: Personality-Performance Correlation\n'
                '人格-性能 Pearson 相关性分析',
                fontsize=16, fontweight='bold', color='white', y=1.05)
    plt.tight_layout()
    path = os.path.join(output_dir, 'personality_performance_correlation.png')
    plt.savefig(path, dpi=200, bbox_inches='tight', facecolor='#0a0a1a')
    plt.close()
    print(f"  💾 Correlation scatter saved: {path}")

    # ================================================================
    # Plot 2: Extraversion ∩-curve (the killer finding)
    # ================================================================
    fig, ax = plt.subplots(figsize=(10, 7), facecolor='#0a0a1a')
    ax.set_facecolor('#0a0a1a')

    x = raw_scores['E']
    y = accuracies

    # Normalize E for display
    x_norm = (x - x.min()) / (x.max() - x.min() + 1e-10)

    for i, m in enumerate(models):
        family = ARCH_FAMILY.get(m, 'Other')
        color = FAMILY_COLORS.get(family, '#FFFFFF')
        ax.scatter(x_norm[i], y[i], c=color, s=200, edgecolors='white',
                  linewidth=2, zorder=5, label=family if family not in
                  [ARCH_FAMILY.get(models[j]) for j in range(i)] else '')
        ax.annotate(m, (x_norm[i], y[i]), fontsize=10, color='white',
                   fontweight='bold', xytext=(8, 8),
                   textcoords='offset points')

    # Fit quadratic (∩ curve): y = a*x^2 + b*x + c
    try:
        coeffs = np.polyfit(x_norm, y, 2)
        x_fit = np.linspace(-0.05, 1.05, 200)
        y_fit = np.polyval(coeffs, x_fit)

        # Check if it's actually ∩ shaped (a < 0)
        if coeffs[0] < 0:
            label = f'∩-curve: y = {coeffs[0]:.1f}x² + {coeffs[1]:.1f}x + {coeffs[2]:.1f}'
            ax.plot(x_fit, y_fit, '--', color='#FF6B6B', linewidth=2.5,
                   alpha=0.8, label=label)

            # Mark optimal extraversion
            x_opt = -coeffs[1] / (2 * coeffs[0])
            y_opt = np.polyval(coeffs, x_opt)
            if 0 <= x_opt <= 1:
                ax.axvline(x=x_opt, color='#2ECC71', linestyle=':', alpha=0.5)
                ax.annotate(f'Optimal E = {x_opt:.2f}',
                           (x_opt, y_opt + 0.5), fontsize=11,
                           color='#2ECC71', fontweight='bold', ha='center')
        else:
            # Monotonic relationship, still interesting
            z = np.polyfit(x_norm, y, 1)
            ax.plot(x_fit, np.polyval(z, x_fit), '--', color='#FF6B6B',
                   linewidth=2, alpha=0.7, label=f'Linear: r={pearsonr(x_norm, y)[0]:.3f}')
    except Exception:
        pass

    # Add ∩ annotation
    ax.text(0.5, 0.05, '← 太内向 (Too Introverted)          太外向 (Too Extraverted) →',
           transform=ax.transAxes, ha='center', color='white', alpha=0.5,
           fontsize=10, style='italic')

    ax.set_xlabel('Extraversion Score (normalized)', color='white', fontsize=13)
    ax.set_ylabel('ImageNet Top-1 Accuracy (%)', color='white', fontsize=13)
    ax.set_title('The Extraversion-Performance ∩ Curve\n'
                '\"Too introverted or too extraverted — both hurt performance\"',
                color='white', fontsize=15, fontweight='bold', pad=15)

    # De-duplicate legend
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(),
             facecolor='#1a1a2e', edgecolor='white', labelcolor='white',
             fontsize=10, loc='lower right')

    ax.tick_params(colors='white')
    ax.spines['bottom'].set_color('white')
    ax.spines['left'].set_color('white')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    path = os.path.join(output_dir, 'extraversion_inverted_u.png')
    plt.savefig(path, dpi=200, bbox_inches='tight', facecolor='#0a0a1a')
    plt.close()
    print(f"  💾 ∩-curve saved: {path}")

    # Print correlation table
    print("\n  📊 Pearson Correlation Table:")
    print(f"  {'Trait':<20} {'r':>8} {'p':>8} {'Significance':>15}")
    print(f"  {'-'*55}")
    for t, tname in zip(traits, trait_names):
        c = correlations[t]
        sig = '***' if c['p'] < 0.01 else '**' if c['p'] < 0.05 else '*' if c['p'] < 0.1 else 'n.s.'
        print(f"  {tname:<20} {c['r']:>8.3f} {c['p']:>8.3f} {sig:>15}")

    return correlations


# ============================================================================
# 3. Architecture Family Personality Clustering
# ============================================================================

def plot_architecture_clustering(results, output_dir):
    """
    Group models by architecture family, show average personality profiles.
    """
    families = {}
    for model_name, data in results.items():
        family = ARCH_FAMILY.get(model_name, 'Other')
        if family not in families:
            families[family] = []
        ns = data['normalized_scores']
        families[family].append([ns['E'], ns['N'], ns['O'], ns['A'], ns['C']])

    # Average per family
    family_names = list(families.keys())
    family_scores = [np.mean(families[f], axis=0) for f in family_names]

    traits = ['E\n外向性', 'N\n神经质', 'O\n开放性', 'A\n宜人性', 'C\n尽责性']

    fig, ax = plt.subplots(figsize=(12, 6), facecolor='#0a0a1a')
    ax.set_facecolor('#0a0a1a')

    x = np.arange(len(traits))
    width = 0.12
    n_families = len(family_names)

    for i, (fname, fscores) in enumerate(zip(family_names, family_scores)):
        offset = (i - n_families/2 + 0.5) * width
        color = FAMILY_COLORS.get(fname, '#FFFFFF')
        bars = ax.bar(x + offset, fscores, width, label=fname,
                     color=color, alpha=0.85, edgecolor='white', linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(traits, fontsize=12, color='white', fontweight='bold')
    ax.set_ylabel('Normalized Score', color='white', fontsize=12)
    ax.set_title('Personality Profiles by Architecture Family\n'
                '不同架构家族的人格画像',
                color='white', fontsize=14, fontweight='bold', pad=15)
    ax.legend(facecolor='#1a1a2e', edgecolor='white', labelcolor='white',
             fontsize=10, ncol=3, loc='upper center',
             bbox_to_anchor=(0.5, -0.1))
    ax.set_ylim(0, 1.15)
    ax.tick_params(colors='white')
    ax.spines['bottom'].set_color('white')
    ax.spines['left'].set_color('white')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    path = os.path.join(output_dir, 'architecture_family_personality.png')
    plt.savefig(path, dpi=200, bbox_inches='tight', facecolor='#0a0a1a')
    plt.close()
    print(f"  💾 Architecture family plot saved: {path}")


# ============================================================================
# 4. Personality Correlation Matrix (inter-trait)
# ============================================================================

def plot_trait_correlation_matrix(results, output_dir):
    """
    Inter-trait correlation matrix across all models.
    Shows which personality traits are correlated.
    """
    models = list(results.keys())
    traits = ['E', 'N', 'O', 'A', 'C']
    trait_labels = ['E (外向性)', 'N (神经质)', 'O (开放性)', 'A (宜人性)', 'C (尽责性)']

    # Build score matrix
    scores = np.zeros((len(models), 5))
    for i, m in enumerate(models):
        for j, t in enumerate(traits):
            scores[i, j] = results[m]['normalized_scores'][t]

    # Correlation matrix
    corr_matrix = np.corrcoef(scores.T)

    fig, ax = plt.subplots(figsize=(8, 7), facecolor='#0a0a1a')
    ax.set_facecolor('#0a0a1a')

    im = ax.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1,
                   interpolation='nearest')
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Pearson r', color='white', fontsize=11)
    cbar.ax.tick_params(colors='white')

    # Annotate cells
    for i in range(5):
        for j in range(5):
            color = 'white' if abs(corr_matrix[i, j]) < 0.5 else 'black'
            ax.text(j, i, f'{corr_matrix[i,j]:.2f}', ha='center', va='center',
                   color=color, fontsize=12, fontweight='bold')

    ax.set_xticks(range(5))
    ax.set_xticklabels(trait_labels, fontsize=10, color='white', rotation=30, ha='right')
    ax.set_yticks(range(5))
    ax.set_yticklabels(trait_labels, fontsize=10, color='white')
    ax.set_title('Inter-Trait Correlation Matrix\n'
                '人格维度间相关性矩阵',
                color='white', fontsize=14, fontweight='bold', pad=15)

    plt.tight_layout()
    path = os.path.join(output_dir, 'trait_correlation_matrix.png')
    plt.savefig(path, dpi=200, bbox_inches='tight', facecolor='#0a0a1a')
    plt.close()
    print(f"  💾 Trait correlation matrix saved: {path}")


# ============================================================================
# 5. Wasserstein Heatmaps for Key Models
# ============================================================================

def generate_wasserstein_heatmaps(output_dir):
    """Generate Wasserstein heatmaps for select models."""
    import torchvision.models as models

    key_models = {
        'VGG-16': lambda: models.vgg16(weights='IMAGENET1K_V1'),
        'ResNet-152': lambda: models.resnet152(weights='IMAGENET1K_V1'),
        'ViT-Small': lambda: models.vit_b_16(weights='IMAGENET1K_V1'),
    }

    for name, loader in key_models.items():
        try:
            print(f"\n  🔥 Computing Wasserstein heatmap for {name}...")
            model = loader()
            model.eval()
            plot_wasserstein_heatmap(model, name, output_dir)
            del model  # Free memory
        except Exception as e:
            print(f"  ⚠️ Skipped {name}: {e}")


# ============================================================================
# Main
# ============================================================================

def main():
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║                                                                      ║
║   🎨 SUPPLEMENTARY ANALYSIS & VISUALIZATIONS                        ║
║                                                                      ║
║   Experiment 2: Personality-Performance Correlation                  ║
║   + Wasserstein Heatmaps                                             ║
║   + Architecture Family Clustering                                   ║
║   + Inter-trait Correlation Matrix                                   ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝
    """)

    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
    os.makedirs(output_dir, exist_ok=True)

    # Load previous results
    results = load_results(output_dir)
    print(f"  📦 Loaded results for {len(results)} models.\n")

    # ================================================================
    # Experiment 2: Personality-Performance Correlation
    # ================================================================
    print("="*60)
    print("📊 Experiment 2: Personality-Performance Correlation")
    print("="*60)
    correlations = plot_personality_performance_correlation(results, output_dir)

    # ================================================================
    # Architecture Family Clustering
    # ================================================================
    print("\n" + "="*60)
    print("🏗️ Architecture Family Personality Profiles")
    print("="*60)
    plot_architecture_clustering(results, output_dir)

    # ================================================================
    # Inter-trait Correlation Matrix
    # ================================================================
    print("\n" + "="*60)
    print("🔗 Inter-trait Correlation Matrix")
    print("="*60)
    plot_trait_correlation_matrix(results, output_dir)

    # ================================================================
    # Wasserstein Heatmaps (top 3 models)
    # ================================================================
    print("\n" + "="*60)
    print("🔥 Wasserstein Distance Heatmaps")
    print("="*60)
    generate_wasserstein_heatmaps(output_dir)

    # ================================================================
    # Summary
    # ================================================================
    print(f"""
╔══════════════════════════════════════════════════════════════════════╗
║  ✅ SUPPLEMENTARY ANALYSIS COMPLETE                                  ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                      ║
║  New output files:                                                   ║
║    • personality_performance_correlation.png                          ║
║    • extraversion_inverted_u.png                                     ║
║    • architecture_family_personality.png                              ║
║    • trait_correlation_matrix.png                                     ║
║    • wasserstein_heatmap_*.png (per model)                           ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝
    """)


if __name__ == '__main__':
    main()
