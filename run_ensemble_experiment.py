"""
================================================================================
🎯 Personality-Guided Ensemble Experiment
================================================================================

The "killer application" of Neural Personality Assessment:
Can personality diversity predict ensemble complementarity?

Strategy:
  1. Train N models with different seeds
  2. Assess personality of each
  3. Compare ensemble selection strategies:
     a) Personality-diverse: maximize pairwise personality distance
     b) Random: pick K models randomly
     c) Best-individual: pick top-K by individual accuracy
  4. Evaluate ensemble accuracy (majority voting + soft voting)

If personality-guided ≥ best-individual, the paper has a practical contribution.

Author: 王唱晓
================================================================================
"""

import os
import sys
import json
import time
import copy
import itertools
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from neural_personality import NeuralBigFiveAssessment
from social_dropout import SocialDropout

matplotlib.rcParams['font.family'] = ['Microsoft YaHei', 'SimHei', 'sans-serif']


# ============================================================================
# Model (reuse SmallResNet)
# ============================================================================

class BasicBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride, bias=False),
                nn.BatchNorm2d(out_ch)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)


class SmallResNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.layer1 = self._make_layer(32, 32, 2, stride=1)
        self.layer2 = self._make_layer(32, 64, 2, stride=2)
        self.layer3 = self._make_layer(64, 128, 2, stride=2)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(128, num_classes)

    def _make_layer(self, in_ch, out_ch, n_blocks, stride):
        layers = [BasicBlock(in_ch, out_ch, stride)]
        for _ in range(1, n_blocks):
            layers.append(BasicBlock(out_ch, out_ch, 1))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        return self.fc(out)


# ============================================================================
# Helpers
# ============================================================================

def get_cifar10(batch_size=128):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform_test)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)
    testloader = DataLoader(testset, batch_size=256, shuffle=False, num_workers=0)
    return trainloader, testloader


def train_model(model, trainloader, device, epochs=15, lr=0.05,
                social_dropout=None):
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        for x, y in trainloader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            if social_dropout is not None:
                loss = loss + social_dropout.compute_social_loss()
            loss.backward()
            optimizer.step()
        scheduler.step()


def get_predictions(model, testloader, device):
    """Get all predictions and logits for the test set."""
    model.eval()
    all_preds = []
    all_logits = []
    all_labels = []
    with torch.no_grad():
        for x, y in testloader:
            x = x.to(device)
            logits = model(x)
            preds = logits.argmax(dim=1)
            all_preds.append(preds.cpu())
            all_logits.append(logits.cpu())
            all_labels.append(y)
    return (torch.cat(all_preds),
            torch.cat(all_logits),
            torch.cat(all_labels))


def ensemble_accuracy(predictions_list, labels, method='majority'):
    """Compute ensemble accuracy."""
    if method == 'majority':
        # Majority voting
        stacked = torch.stack(predictions_list, dim=0)  # (K, N)
        votes = torch.mode(stacked, dim=0).values
        return (votes == labels).float().mean().item() * 100

    elif method == 'soft':
        # Soft voting (average logits)
        # predictions_list should be logits in this case
        avg_logits = torch.stack(predictions_list, dim=0).mean(dim=0)
        final_preds = avg_logits.argmax(dim=1)
        return (final_preds == labels).float().mean().item() * 100


def personality_distance(p1, p2):
    """Compute Euclidean distance between two personality profiles."""
    return np.sqrt(sum((p1[t] - p2[t]) ** 2 for t in ['E', 'N', 'O', 'A', 'C']))


def select_diverse_ensemble(personalities, K):
    """Select K models that maximize total pairwise personality distance."""
    n = len(personalities)
    if K >= n:
        return list(range(n))

    best_subset = None
    best_diversity = -1

    for subset in itertools.combinations(range(n), K):
        total_dist = 0
        for i, j in itertools.combinations(subset, 2):
            total_dist += personality_distance(personalities[i], personalities[j])
        if total_dist > best_diversity:
            best_diversity = total_dist
            best_subset = list(subset)

    return best_subset


# ============================================================================
# Visualization
# ============================================================================

def plot_ensemble_comparison(results, output_dir):
    """Bar chart comparing ensemble strategies."""
    strategies = list(results.keys())
    majority_accs = [results[s]['majority'] for s in strategies]
    soft_accs = [results[s]['soft'] for s in strategies]

    fig, ax = plt.subplots(figsize=(10, 6), facecolor='#0a0a1a')
    ax.set_facecolor('#0a0a1a')

    x = np.arange(len(strategies))
    width = 0.35

    bars1 = ax.bar(x - width/2, majority_accs, width, label='Majority Voting',
                   color='#4ECDC4', alpha=0.85, edgecolor='white', linewidth=0.5)
    bars2 = ax.bar(x + width/2, soft_accs, width, label='Soft Voting',
                   color='#FF6B6B', alpha=0.85, edgecolor='white', linewidth=0.5)

    for bar in list(bars1) + list(bars2):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., h + 0.05,
               f'{h:.2f}%', ha='center', va='bottom', color='white', fontsize=10)

    ax.set_xticks(x)
    labels_display = {
        'personality_diverse': 'Personality\nDiverse',
        'random_avg': 'Random\n(avg of 10)',
        'best_individual': 'Best\nIndividual',
        'all_ensemble': 'All Models\nEnsemble',
    }
    ax.set_xticklabels([labels_display.get(s, s) for s in strategies],
                       fontsize=12, color='white')
    ax.set_ylabel('Test Accuracy (%)', color='white', fontsize=12)
    ax.set_title('Ensemble Strategy Comparison\n'
                '集成选择策略对比',
                color='white', fontsize=14, fontweight='bold')
    ax.legend(facecolor='#1a1a2e', edgecolor='white', labelcolor='white', fontsize=11)
    ax.tick_params(colors='white')
    ax.spines['bottom'].set_color('white')
    ax.spines['left'].set_color('white')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    path = os.path.join(output_dir, 'ensemble_comparison.png')
    plt.savefig(path, dpi=200, bbox_inches='tight', facecolor='#0a0a1a')
    plt.close()
    print(f"  💾 Ensemble comparison saved: {path}")


# ============================================================================
# Main
# ============================================================================

def main():
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║                                                                      ║
║   🎯 ENSEMBLE EXPERIMENT: Personality-Guided Model Selection        ║
║                                                                      ║
║   "Can personality diversity predict ensemble complementarity?"      ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝
    """)

    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    N_MODELS = 8        # Total models to train
    K_ENSEMBLE = 3      # Ensemble size
    EPOCHS = 15
    SEEDS = list(range(100, 100 + N_MODELS))

    # ================================================================
    # Step 1: Train N models with different seeds
    # ================================================================
    print(f"📦 Training {N_MODELS} models with different seeds...")
    trainloader, testloader = get_cifar10()

    models = []
    personalities = []
    individual_accs = []
    all_preds = []
    all_logits = []
    labels = None

    for i, seed in enumerate(SEEDS):
        print(f"\n  [{i+1}/{N_MODELS}] Seed={seed}", end=" ")
        torch.manual_seed(seed)
        np.random.seed(seed)

        model = SmallResNet(num_classes=10).to(device)
        t0 = time.time()
        train_model(model, trainloader, device, epochs=EPOCHS)
        elapsed = time.time() - t0

        preds, logits, lbls = get_predictions(model, testloader, device)
        if labels is None:
            labels = lbls
        acc = (preds == labels).float().mean().item() * 100

        # Personality
        nbfa = NeuralBigFiveAssessment()
        nbfa.assess(model, f"model_{seed}")
        raw = nbfa.results[f'model_{seed}']['raw_scores']
        personality = {k: float(v) for k, v in raw.items()}

        models.append(model)
        personalities.append(personality)
        individual_accs.append(acc)
        all_preds.append(preds)
        all_logits.append(logits)

        print(f"→ Acc: {acc:.2f}% | E={personality['E']:.1f} "
              f"N={personality['N']:.1f} O={personality['O']:.2f} | {elapsed:.0f}s")

    # ================================================================
    # Step 2: Ensemble Selection Strategies
    # ================================================================
    print(f"\n{'='*60}")
    print(f"🎯 Comparing Ensemble Strategies (K={K_ENSEMBLE})")
    print(f"{'='*60}")

    results = {}

    # Strategy A: Personality-diverse
    diverse_idx = select_diverse_ensemble(personalities, K_ENSEMBLE)
    div_preds = [all_preds[i] for i in diverse_idx]
    div_logits = [all_logits[i] for i in diverse_idx]
    results['personality_diverse'] = {
        'majority': ensemble_accuracy(div_preds, labels, 'majority'),
        'soft': ensemble_accuracy(div_logits, labels, 'soft'),
        'indices': diverse_idx,
        'individual_accs': [individual_accs[i] for i in diverse_idx],
    }
    print(f"\n  🧠 Personality-Diverse (indices={diverse_idx}):")
    print(f"     Individual accs: {[f'{individual_accs[i]:.2f}%' for i in diverse_idx]}")
    print(f"     Majority: {results['personality_diverse']['majority']:.2f}%  "
          f"Soft: {results['personality_diverse']['soft']:.2f}%")

    # Strategy B: Random (average over 10 random selections)
    random_majority_accs = []
    random_soft_accs = []
    for trial in range(10):
        np.random.seed(trial + 999)
        rand_idx = np.random.choice(N_MODELS, K_ENSEMBLE, replace=False).tolist()
        rand_preds = [all_preds[i] for i in rand_idx]
        rand_logits = [all_logits[i] for i in rand_idx]
        random_majority_accs.append(ensemble_accuracy(rand_preds, labels, 'majority'))
        random_soft_accs.append(ensemble_accuracy(rand_logits, labels, 'soft'))

    results['random_avg'] = {
        'majority': float(np.mean(random_majority_accs)),
        'soft': float(np.mean(random_soft_accs)),
        'majority_std': float(np.std(random_majority_accs)),
        'soft_std': float(np.std(random_soft_accs)),
    }
    print(f"\n  🎲 Random (avg of 10 trials):")
    print(f"     Majority: {results['random_avg']['majority']:.2f}±{results['random_avg']['majority_std']:.2f}%  "
          f"Soft: {results['random_avg']['soft']:.2f}±{results['random_avg']['soft_std']:.2f}%")

    # Strategy C: Best-individual top-K
    sorted_idx = sorted(range(N_MODELS), key=lambda i: individual_accs[i], reverse=True)
    best_idx = sorted_idx[:K_ENSEMBLE]
    best_preds = [all_preds[i] for i in best_idx]
    best_logits = [all_logits[i] for i in best_idx]
    results['best_individual'] = {
        'majority': ensemble_accuracy(best_preds, labels, 'majority'),
        'soft': ensemble_accuracy(best_logits, labels, 'soft'),
        'indices': best_idx,
        'individual_accs': [individual_accs[i] for i in best_idx],
    }
    print(f"\n  🏆 Best-Individual Top-{K_ENSEMBLE} (indices={best_idx}):")
    print(f"     Individual accs: {[f'{individual_accs[i]:.2f}%' for i in best_idx]}")
    print(f"     Majority: {results['best_individual']['majority']:.2f}%  "
          f"Soft: {results['best_individual']['soft']:.2f}%")

    # Strategy D: All models ensemble
    results['all_ensemble'] = {
        'majority': ensemble_accuracy(all_preds, labels, 'majority'),
        'soft': ensemble_accuracy(all_logits, labels, 'soft'),
    }
    print(f"\n  📊 All {N_MODELS} Models Ensemble:")
    print(f"     Majority: {results['all_ensemble']['majority']:.2f}%  "
          f"Soft: {results['all_ensemble']['soft']:.2f}%")

    # ================================================================
    # Step 3: Personality Diversity Table
    # ================================================================
    print(f"\n{'='*60}")
    print(f"📊 Model Personality Profiles")
    print(f"{'='*60}")
    print(f"  {'Seed':<8} {'Acc':>7}  {'E':>8} {'N':>8} {'O':>8} {'A':>8} {'C':>8}")
    print(f"  {'-'*55}")
    for i, seed in enumerate(SEEDS):
        p = personalities[i]
        marker = ' ← diverse' if i in diverse_idx else ''
        print(f"  {seed:<8} {individual_accs[i]:6.2f}%  "
              f"{p['E']:8.2f} {p['N']:8.2f} {p['O']:8.4f} "
              f"{p['A']:8.4f} {p['C']:8.4f}{marker}")

    # ================================================================
    # Visualize
    # ================================================================
    plot_ensemble_comparison(results, output_dir)

    # Save results
    save_results = {
        'strategies': results,
        'individual_accs': individual_accs,
        'personalities': personalities,
        'seeds': SEEDS,
        'n_models': N_MODELS,
        'k_ensemble': K_ENSEMBLE,
    }
    path = os.path.join(output_dir, 'ensemble_results.json')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(save_results, f, indent=2, ensure_ascii=False)

    # ================================================================
    # Conclusion
    # ================================================================
    div_soft = results['personality_diverse']['soft']
    best_soft = results['best_individual']['soft']
    rand_soft = results['random_avg']['soft']

    print(f"""
╔══════════════════════════════════════════════════════════════════════╗
║  ✅ ENSEMBLE EXPERIMENT COMPLETE                                     ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                      ║
║  Best Individual Acc:        {max(individual_accs):.2f}%                           
║  Personality-Diverse Soft:   {div_soft:.2f}%                           
║  Best-Individual Soft:       {best_soft:.2f}%                           
║  Random Ensemble Soft:       {rand_soft:.2f}%                           
║                                                                      ║""")

    if div_soft >= best_soft:
        print("║  🎯 Personality diversity WINS over best-individual!            ║")
        print("║  → Paper claim: personality profiles have predictive utility    ║")
    elif div_soft >= rand_soft:
        print("║  🎯 Personality diversity beats random, close to best-individual║")
        print("║  → Paper claim: personality is a useful signal for selection    ║")
    else:
        print("║  🤔 Personality diversity underperforms — needs investigation   ║")
        print("║  → Paper claim: further work needed on selection criteria      ║")

    print("║                                                                      ║")
    print("╚══════════════════════════════════════════════════════════════════════╝")


if __name__ == '__main__':
    main()
