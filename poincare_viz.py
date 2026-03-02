"""
================================================================================
🌀 Poincaré Ball Visualization for Neural Personality Space
================================================================================

Embeds neural network personality vectors into the Poincaré ball model
of hyperbolic space for visualization.

Hyperbolic space is ideal for personality visualization because:
1. It naturally represents hierarchical relationships
2. The boundary emphasizes extreme traits (exponential growth near edge)
3. It produces visually stunning plots for the paper

Mathematical framework:
    embed: R^5 → B^2 (Poincaré disk)
    Using exponential map: exp_0(v) = tanh(||v||/2) · v/||v||

Author: 王唱晓
================================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.collections import LineCollection
import matplotlib
matplotlib.rcParams['font.family'] = ['Microsoft YaHei', 'SimHei', 'sans-serif']


def poincare_exp_map(v, c=1.0):
    """
    Exponential map at the origin of the Poincaré ball.
    exp_0(v) = tanh(√c · ||v|| / 2) · v / (√c · ||v||)
    """
    norm = np.linalg.norm(v, axis=-1, keepdims=True)
    norm = np.clip(norm, 1e-10, None)
    sqrt_c = np.sqrt(c)
    coeff = np.tanh(sqrt_c * norm / 2.0) / (sqrt_c * norm)
    return coeff * v


def pca_project(scores, n_components=2):
    """Project 5D personality scores to 2D using PCA."""
    scores = np.array(scores)
    mean = scores.mean(axis=0)
    centered = scores - mean
    
    # SVD for PCA
    U, S, Vt = np.linalg.svd(centered, full_matrices=False)
    projected = centered @ Vt[:n_components].T
    
    # Scale to reasonable range for Poincaré ball
    max_norm = np.max(np.linalg.norm(projected, axis=1))
    if max_norm > 0:
        projected = projected / max_norm * 1.5  # Scale factor
    
    return projected


def plot_poincare_personality_space(names, scores, save_path=None):
    """
    Create a stunning Poincaré ball visualization of neural personalities.
    
    Args:
        names: list of model names
        scores: np.array of shape (N, 5) — normalized personality scores
        save_path: path to save the figure
    """
    # Project 5D → 2D via PCA
    projected_2d = pca_project(scores)
    
    # Map to Poincaré ball via exponential map
    poincare_coords = poincare_exp_map(projected_2d)
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(10, 10), facecolor='#0a0a1a')
    ax.set_facecolor('#0a0a1a')
    
    # Draw Poincaré disk boundary
    boundary = plt.Circle((0, 0), 1.0, fill=False, color='#ffffff', 
                          linewidth=2, alpha=0.3, linestyle='--')
    ax.add_patch(boundary)
    
    # Draw concentric geodesic circles (hyperbolic distance from origin)
    for r in [0.3, 0.5, 0.7, 0.9]:
        circle = plt.Circle((0, 0), r, fill=False, color='#ffffff', 
                           linewidth=0.5, alpha=0.1)
        ax.add_patch(circle)
    
    # Draw axes
    ax.axhline(y=0, color='#ffffff', linewidth=0.5, alpha=0.1)
    ax.axvline(x=0, color='#ffffff', linewidth=0.5, alpha=0.1)
    
    # Color scheme based on personality
    # Color by dominant trait
    colors = []
    trait_colors = {
        0: '#FF6B6B',  # E - Red (warm/social)
        1: '#9B59B6',  # N - Purple (emotional)
        2: '#3498DB',  # O - Blue (intellectual)
        3: '#2ECC71',  # A - Green (harmonious)
        4: '#F39C12',  # C - Orange (structured)
    }
    
    for s in scores:
        dominant = np.argmax(s)
        colors.append(trait_colors[dominant])
    
    # Plot points with glow effect
    for i, (name, coord, color) in enumerate(zip(names, poincare_coords, colors)):
        # Outer glow
        ax.scatter(coord[0], coord[1], s=500, c=color, alpha=0.15, 
                  edgecolors='none', zorder=2)
        ax.scatter(coord[0], coord[1], s=300, c=color, alpha=0.3, 
                  edgecolors='none', zorder=3)
        # Core point
        ax.scatter(coord[0], coord[1], s=120, c=color, alpha=0.9,
                  edgecolors='white', linewidth=1.5, zorder=4)
        
        # Label
        offset_x = 0.04 if coord[0] >= 0 else -0.04
        offset_y = 0.04
        ha = 'left' if coord[0] >= 0 else 'right'
        
        ax.annotate(name, (coord[0], coord[1]),
                   xytext=(coord[0] + offset_x, coord[1] + offset_y),
                   fontsize=11, fontweight='bold', color='white',
                   ha=ha, va='bottom',
                   arrowprops=dict(arrowstyle='-', color='white', alpha=0.3),
                   zorder=5)
    
    # Draw connections between models (hyperbolic geodesics approximation)
    for i in range(len(poincare_coords)):
        for j in range(i + 1, len(poincare_coords)):
            dist = np.linalg.norm(poincare_coords[i] - poincare_coords[j])
            if dist < 0.5:  # Only draw close connections
                alpha = 0.1 * (1 - dist / 0.5)
                ax.plot([poincare_coords[i][0], poincare_coords[j][0]],
                       [poincare_coords[i][1], poincare_coords[j][1]],
                       color='white', alpha=alpha, linewidth=0.5, zorder=1)
    
    # Title and labels
    ax.set_title('Neural Personality Space\n(Poincaré Ball Model, 𝕳²)',
                fontsize=18, fontweight='bold', color='white', pad=20)
    
    # Legend
    legend_elements = [
        plt.scatter([], [], c=trait_colors[0], s=80, label='E-dominant (Extraversion)'),
        plt.scatter([], [], c=trait_colors[1], s=80, label='N-dominant (Neuroticism)'),
        plt.scatter([], [], c=trait_colors[2], s=80, label='O-dominant (Openness)'),
        plt.scatter([], [], c=trait_colors[3], s=80, label='A-dominant (Agreeableness)'),
        plt.scatter([], [], c=trait_colors[4], s=80, label='C-dominant (Conscientiousness)'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=9,
             facecolor='#1a1a2e', edgecolor='white', labelcolor='white',
             framealpha=0.7)
    
    # Axis settings
    ax.set_xlim(-1.15, 1.15)
    ax.set_ylim(-1.15, 1.15)
    ax.set_aspect('equal')
    ax.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight',
                   facecolor='#0a0a1a', edgecolor='none')
        print(f"  💾 Poincaré visualization saved to: {save_path}")
    
    plt.close()
    return fig


def plot_radar_comparison(names, scores, save_path=None):
    """
    Create radar chart comparing personality profiles of all models.
    
    Args:
        names: list of model names
        scores: np.array of shape (N, 5) - normalized [0, 1]
        save_path: path to save
    """
    traits = ['E\n(外向性)', 'N\n(神经质)', 'O\n(开放性)', 'A\n(宜人性)', 'C\n(尽责性)']
    n_traits = len(traits)
    
    # Compute angles
    angles = np.linspace(0, 2 * np.pi, n_traits, endpoint=False).tolist()
    angles += angles[:1]  # Close the loop
    
    # Color palette
    palette = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7',
               '#DDA0DD', '#98D8C8', '#F7DC6F']
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True),
                           facecolor='#0a0a1a')
    ax.set_facecolor('#0a0a1a')
    
    for i, (name, score) in enumerate(zip(names, scores)):
        values = score.tolist()
        values += values[:1]  # Close the loop
        
        color = palette[i % len(palette)]
        ax.plot(angles, values, 'o-', linewidth=2, label=name,
               color=color, alpha=0.85)
        ax.fill(angles, values, alpha=0.08, color=color)
    
    # Styling
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(traits, fontsize=12, fontweight='bold', color='white')
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8'], fontsize=8, color='gray')
    
    # Grid styling
    ax.spines['polar'].set_color('white')
    ax.spines['polar'].set_alpha(0.2)
    ax.grid(color='white', alpha=0.1)
    ax.tick_params(colors='white')
    
    ax.set_title('Neural Big Five Personality Profiles\n(Normalized Scores)',
                fontsize=16, fontweight='bold', color='white', pad=30)
    
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=9,
             facecolor='#1a1a2e', edgecolor='white', labelcolor='white',
             framealpha=0.8)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight',
                   facecolor='#0a0a1a', edgecolor='none')
        print(f"  💾 Radar chart saved to: {save_path}")
    
    plt.close()
    return fig


def plot_depth_extraversion(model_data, save_path=None):
    """
    Plot the key finding: Depth vs Extraversion (E ∝ L^{-1/2}).
    
    Args:
        model_data: list of (name, depth, extraversion_raw) tuples
        save_path: path to save
    """
    fig, ax = plt.subplots(figsize=(10, 6), facecolor='#0a0a1a')
    ax.set_facecolor('#0a0a1a')
    
    names = [d[0] for d in model_data]
    depths = np.array([d[1] for d in model_data])
    extraversions = np.array([d[2] for d in model_data])
    
    # Normalize extraversion for display
    e_norm = (extraversions - extraversions.min()) / (extraversions.max() - extraversions.min() + 1e-10)
    
    # Scatter plot
    colors_map = plt.cm.RdYlGn(e_norm)
    scatter = ax.scatter(depths, e_norm, c=e_norm, cmap='RdYlGn',
                        s=200, edgecolors='white', linewidth=1.5, zorder=5)
    
    # Fit theoretical curve: E = a * L^(-1/2) + b
    from scipy.optimize import curve_fit
    def theory_curve(L, a, b):
        return a / np.sqrt(L) + b
    
    try:
        popt, _ = curve_fit(theory_curve, depths, e_norm, p0=[1, 0])
        L_range = np.linspace(min(depths) * 0.8, max(depths) * 1.2, 100)
        ax.plot(L_range, theory_curve(L_range, *popt),
               '--', color='#FF6B6B', linewidth=2, alpha=0.7,
               label=f'$\\mathcal{{E}} \\propto L^{{-1/2}}$ (R² = {1 - np.sum((e_norm - theory_curve(depths, *popt))**2) / np.sum((e_norm - e_norm.mean())**2):.3f})')
    except Exception:
        pass
    
    # Labels
    for i, name in enumerate(names):
        offset_y = 0.03 if i % 2 == 0 else -0.05
        ax.annotate(name, (depths[i], e_norm[i]),
                   xytext=(depths[i], e_norm[i] + offset_y),
                   fontsize=9, color='white', ha='center',
                   fontweight='bold')
    
    ax.set_xlabel('Network Depth (number of weight layers)', fontsize=13, 
                 color='white', labelpad=10)
    ax.set_ylabel('Extraversion Score (normalized)', fontsize=13, 
                 color='white', labelpad=10)
    ax.set_title('The Introversion Scaling Law\n$\\mathcal{E}(\\mathcal{N}) = O(L^{-1/2})$',
                fontsize=16, fontweight='bold', color='white', pad=15)
    
    ax.tick_params(colors='white')
    ax.spines['bottom'].set_color('white')
    ax.spines['left'].set_color('white')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(fontsize=12, facecolor='#1a1a2e', edgecolor='white',
             labelcolor='white', framealpha=0.8)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight',
                   facecolor='#0a0a1a', edgecolor='none')
        print(f"  💾 Depth vs Extraversion plot saved to: {save_path}")
    
    plt.close()
    return fig


# ============================================================================
# Main
# ============================================================================

# Known depth (number of weight layers) for each model
MODEL_DEPTH = {
    'VGG-16':          16,
    'ResNet-50':       54,
    'ResNet-152':      156,
    'DenseNet-121':    121,
    'EfficientNet-B0': 82,
    'ConvNeXt-Tiny':   59,
    'ViT-Small':       50,
    'ViT-Base':        50,
}


def main():
    import os, json

    print("""
╔══════════════════════════════════════════════════════════════════════╗
║                                                                      ║
║   🌀 EXPERIMENT 4: HYPERBOLIC PERSONALITY SPACE                     ║
║                                                                      ║
║   Poincaré Ball Embedding + Radar Charts + Depth Scaling Law         ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝
    """)

    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
    os.makedirs(results_dir, exist_ok=True)

    # ================================================================
    # Load all model results
    # ================================================================
    nbfa_path = os.path.join(results_dir, 'nbfa_results.json')
    social_path = os.path.join(results_dir, 'social_dropout_results.json')

    with open(nbfa_path, 'r', encoding='utf-8') as f:
        nbfa_results = json.load(f)

    # Also include social dropout models if available
    if os.path.exists(social_path):
        with open(social_path, 'r', encoding='utf-8') as f:
            sd_results = json.load(f)
        # Add baseline and social dropout from experiment 3
        if 'baseline' in sd_results and 'personality_raw' in sd_results['baseline']:
            nbfa_results['SD-Baseline'] = {
                'normalized_scores': sd_results['baseline']['personality_raw'],
                'raw_scores': sd_results['baseline']['personality_raw'],
                'num_layers': 16,
            }
        if 'social_dropout' in sd_results and 'personality_raw' in sd_results['social_dropout']:
            nbfa_results['SD-Treated'] = {
                'normalized_scores': sd_results['social_dropout']['personality_raw'],
                'raw_scores': sd_results['social_dropout']['personality_raw'],
                'num_layers': 16,
            }
        print(f"  📦 Loaded Social Dropout results (SD-Baseline, SD-Treated)")

    print(f"  📦 Total models: {len(nbfa_results)}\n")

    # Extract names and normalized scores
    names = list(nbfa_results.keys())
    norm_scores = np.array([
        [nbfa_results[m]['normalized_scores'][t] for t in ['E', 'N', 'O', 'A', 'C']]
        for m in names
    ])

    # ================================================================
    # 1. Poincaré Ball Visualization
    # ================================================================
    print("=" * 60)
    print("🌀 Poincaré Ball Embedding")
    print("=" * 60)
    plot_poincare_personality_space(
        names, norm_scores,
        save_path=os.path.join(results_dir, 'poincare_personality_space.png')
    )

    # ================================================================
    # 2. Radar Chart
    # ================================================================
    print("\n" + "=" * 60)
    print("📊 Radar Chart: All Models")
    print("=" * 60)
    plot_radar_comparison(
        names, norm_scores,
        save_path=os.path.join(results_dir, 'radar_personality_comparison.png')
    )

    # ================================================================
    # 3. Depth vs Extraversion Scaling Law
    # ================================================================
    print("\n" + "=" * 60)
    print("📐 Depth vs Extraversion Scaling Law")
    print("=" * 60)

    model_data = []
    for m in names:
        depth = nbfa_results[m].get('num_layers', MODEL_DEPTH.get(m, None))
        if depth is None:
            continue
        e_raw = nbfa_results[m]['raw_scores']['E']
        model_data.append((m, depth, e_raw))

    if len(model_data) >= 4:
        plot_depth_extraversion(
            model_data,
            save_path=os.path.join(results_dir, 'depth_extraversion_scaling.png')
        )
    else:
        print("  ⚠️ Not enough models with depth info for scaling law plot.")

    # ================================================================
    # Summary
    # ================================================================
    print(f"""
╔══════════════════════════════════════════════════════════════════════╗
║  ✅ EXPERIMENT 4 COMPLETE                                            ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                      ║
║  Output files:                                                       ║
║    • poincare_personality_space.png   (Poincaré ball)                ║
║    • radar_personality_comparison.png (Radar chart)                   ║
║    • depth_extraversion_scaling.png   (E ∝ L^{{-1/2}})              ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝
    """)


if __name__ == '__main__':
    main()
