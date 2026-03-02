"""
================================================================================
🔬 Rigorous Multi-Seed, Multi-Dataset, Multi-Architecture Experiment
================================================================================

Runs the full experimental matrix:
  - 3 Architectures: SmallResNet, TinyVGG, CompactDenseNet
  - 2 Datasets: CIFAR-10, CIFAR-100
  - 5 Training Conditions: Baseline, Dropout, LabelSmooth, WeightDecay, SocialDropout
  - 3 Random Seeds

Total: 3×2×5×3 = 90 training runs

Output: results/rigorous_results.json with full metrics for paper tables.

Author: 王唱晓
================================================================================
"""

import os
import sys
import time
import json
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from neural_personality import NeuralBigFiveAssessment
from social_dropout import SocialDropout


# ============================================================================
# Model Architectures (all compact for CPU training)
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
    """ResNet-style model (~700K params)."""
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


class DropoutWrapper(nn.Module):
    """Wraps a model to insert dropout before the final FC layer."""
    def __init__(self, model, p=0.3):
        super().__init__()
        self.model = model
        self.dropout = nn.Dropout(p=p)
        # Monkey-patch: intercept the fc call
        self._orig_fc = model.fc
        model.fc = nn.Sequential(self.dropout, self._orig_fc)


class TinyVGG(nn.Module):
    """VGG-style model (~600K params)."""
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, 1, 1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, 1, 1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, 1, 1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, 1, 1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

    @property
    def fc(self):
        return self.classifier[-1]

    @fc.setter
    def fc(self, value):
        self.classifier[-1] = value


class DenseBlock(nn.Module):
    def __init__(self, in_ch, growth_rate, n_layers):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(n_layers):
            self.layers.append(nn.Sequential(
                nn.BatchNorm2d(in_ch + i * growth_rate),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_ch + i * growth_rate, growth_rate, 3, 1, 1, bias=False),
            ))

    def forward(self, x):
        features = [x]
        for layer in self.layers:
            out = layer(torch.cat(features, 1))
            features.append(out)
        return torch.cat(features, 1)


class Transition(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.bn = nn.BatchNorm2d(in_ch)
        self.conv = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.pool = nn.AvgPool2d(2, 2)

    def forward(self, x):
        return self.pool(self.conv(F.relu(self.bn(x))))


class CompactDenseNet(nn.Module):
    """DenseNet-style model (~400K params)."""
    def __init__(self, num_classes=10, growth_rate=12, block_layers=(6, 6, 6)):
        super().__init__()
        n_ch = 2 * growth_rate  # 24
        self.conv1 = nn.Conv2d(3, n_ch, 3, 1, 1, bias=False)

        self.blocks = nn.ModuleList()
        self.transitions = nn.ModuleList()
        for i, n_layers in enumerate(block_layers):
            self.blocks.append(DenseBlock(n_ch, growth_rate, n_layers))
            n_ch = n_ch + n_layers * growth_rate
            if i < len(block_layers) - 1:
                out_ch = n_ch // 2
                self.transitions.append(Transition(n_ch, out_ch))
                n_ch = out_ch

        self.bn_final = nn.BatchNorm2d(n_ch)
        self.fc = nn.Linear(n_ch, num_classes)

    def forward(self, x):
        out = self.conv1(x)
        for i, block in enumerate(self.blocks):
            out = block(out)
            if i < len(self.transitions):
                out = self.transitions[i](out)
        out = F.relu(self.bn_final(out))
        out = F.adaptive_avg_pool2d(out, 1)
        out = out.view(out.size(0), -1)
        return self.fc(out)


ARCHITECTURES = {
    'SmallResNet': SmallResNet,
    'TinyVGG': TinyVGG,
    'CompactDenseNet': CompactDenseNet,
}


# ============================================================================
# Data Loading
# ============================================================================

def get_dataset(name='cifar10', batch_size=128):
    """Load CIFAR-10 or CIFAR-100."""
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

    if name == 'cifar10':
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                                download=True, transform=transform_train)
        testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                               download=True, transform=transform_test)
    elif name == 'cifar100':
        trainset = torchvision.datasets.CIFAR100(root='./data', train=True,
                                                 download=True, transform=transform_train)
        testset = torchvision.datasets.CIFAR100(root='./data', train=False,
                                                download=True, transform=transform_test)
    else:
        raise ValueError(f"Unknown dataset: {name}")

    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True,
                            num_workers=0, pin_memory=True)
    testloader = DataLoader(testset, batch_size=256, shuffle=False,
                           num_workers=0, pin_memory=True)
    return trainloader, testloader


# ============================================================================
# Training with different conditions
# ============================================================================

def evaluate(model, testloader, device):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in testloader:
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            _, predicted = outputs.max(1)
            total += y.size(0)
            correct += predicted.eq(y).sum().item()
    return 100.0 * correct / total


def train_accuracy(model, trainloader, device, max_batches=50):
    """Estimate training accuracy on subset for generalization gap."""
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for i, (x, y) in enumerate(trainloader):
            if i >= max_batches:
                break
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            _, predicted = outputs.max(1)
            total += y.size(0)
            correct += predicted.eq(y).sum().item()
    return 100.0 * correct / total


def evaluate_noisy(model, testloader, device, noise_std=0.1):
    """Evaluate accuracy under Gaussian input noise (robustness test)."""
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in testloader:
            x, y = x.to(device), y.to(device)
            x = x + torch.randn_like(x) * noise_std
            outputs = model(x)
            _, predicted = outputs.max(1)
            total += y.size(0)
            correct += predicted.eq(y).sum().item()
    return 100.0 * correct / total


def train_one_run(arch_name, dataset_name, condition, seed, epochs=15,
                  lr=0.05, device='cpu'):
    """
    Train one model under one condition and return full metrics.

    Args:
        arch_name: 'SmallResNet' | 'TinyVGG' | 'CompactDenseNet'
        dataset_name: 'cifar10' | 'cifar100'
        condition: 'baseline' | 'dropout' | 'label_smooth' | 'weight_decay' | 'social_dropout'
        seed: random seed
        epochs: number of training epochs
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    num_classes = 10 if dataset_name == 'cifar10' else 100
    model = ARCHITECTURES[arch_name](num_classes=num_classes).to(device)
    trainloader, testloader = get_dataset(dataset_name, batch_size=128)

    # Configure condition
    wd = 5e-4
    social_dropout = None
    criterion = nn.CrossEntropyLoss()

    if condition == 'baseline':
        wd = 0  # no weight decay
    elif condition == 'dropout':
        # Insert dropout before final FC layer
        DropoutWrapper(model, p=0.3)
        wd = 0
    elif condition == 'label_smooth':
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        wd = 0
    elif condition == 'weight_decay':
        wd = 5e-4  # standard weight decay
    elif condition == 'social_dropout':
        wd = 0
        social_dropout = SocialDropout(social_rate=0.05, method='wasserstein')
        social_dropout.register_hooks(model)

    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=wd)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    history = {'train_loss': [], 'test_acc': [], 'social_loss': []}

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        running_social = 0.0
        n_batches = 0

        for x, y in trainloader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()

            outputs = model(x)

            loss = criterion(outputs, y)

            s_loss = 0.0
            if social_dropout is not None:
                s_loss_tensor = social_dropout.compute_social_loss()
                loss = loss + s_loss_tensor
                s_loss = s_loss_tensor.item()

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            running_social += s_loss
            n_batches += 1

        scheduler.step()
        avg_loss = running_loss / n_batches
        avg_social = running_social / n_batches
        test_acc = evaluate(model, testloader, device)

        history['train_loss'].append(avg_loss)
        history['test_acc'].append(test_acc)
        history['social_loss'].append(avg_social)

    # Final metrics
    final_test_acc = evaluate(model, testloader, device)
    final_train_acc = train_accuracy(model, trainloader, device)
    noisy_acc = evaluate_noisy(model, testloader, device, noise_std=0.1)
    gen_gap = final_train_acc - final_test_acc

    # Remove hooks
    if social_dropout is not None:
        social_dropout.remove_hooks()

    # Personality assessment
    nbfa = NeuralBigFiveAssessment()
    nbfa.assess(model, "model")
    raw_scores = nbfa.results['model']['raw_scores']
    personality = {k: float(v) for k, v in raw_scores.items()}

    result = {
        'arch': arch_name,
        'dataset': dataset_name,
        'condition': condition,
        'seed': seed,
        'final_test_acc': final_test_acc,
        'best_test_acc': max(history['test_acc']),
        'final_train_acc': final_train_acc,
        'gen_gap': gen_gap,
        'noisy_acc': noisy_acc,
        'robustness_drop': final_test_acc - noisy_acc,
        'personality': personality,
        'history': {k: [float(v) for v in vs] for k, vs in history.items()},
    }

    return result, model


# ============================================================================
# Statistical Analysis
# ============================================================================

def compute_statistics(results_list):
    """Compute mean ± std across seeds for each (arch, dataset, condition)."""
    from collections import defaultdict

    groups = defaultdict(list)
    for r in results_list:
        key = (r['arch'], r['dataset'], r['condition'])
        groups[key].append(r)

    stats = {}
    for key, runs in groups.items():
        accs = [r['final_test_acc'] for r in runs]
        gaps = [r['gen_gap'] for r in runs]
        robust = [r['noisy_acc'] for r in runs]

        traits = {}
        for t in ['E', 'N', 'O', 'A', 'C']:
            vals = [r['personality'][t] for r in runs]
            traits[t] = {'mean': float(np.mean(vals)), 'std': float(np.std(vals))}

        stats[key] = {
            'arch': key[0], 'dataset': key[1], 'condition': key[2],
            'n_runs': len(runs),
            'acc_mean': float(np.mean(accs)),
            'acc_std': float(np.std(accs)),
            'gap_mean': float(np.mean(gaps)),
            'gap_std': float(np.std(gaps)),
            'robust_mean': float(np.mean(robust)),
            'robust_std': float(np.std(robust)),
            'personality': traits,
        }
    return stats


def paired_ttest(results_list, condition_a='baseline', condition_b='social_dropout'):
    """Paired t-test between two conditions (same arch, dataset, seed)."""
    from scipy.stats import ttest_rel
    from collections import defaultdict

    pairs = defaultdict(lambda: {'a': [], 'b': []})
    for r in results_list:
        key = (r['arch'], r['dataset'], r['seed'])
        if r['condition'] == condition_a:
            pairs[key]['a'].append(r['final_test_acc'])
        elif r['condition'] == condition_b:
            pairs[key]['b'].append(r['final_test_acc'])

    a_vals, b_vals = [], []
    for key, p in pairs.items():
        if p['a'] and p['b']:
            a_vals.append(p['a'][0])
            b_vals.append(p['b'][0])

    if len(a_vals) < 2:
        return {'t_stat': 0, 'p_value': 1.0, 'n_pairs': 0,
                'mean_diff': 0, 'significant': False}

    t_stat, p_value = ttest_rel(b_vals, a_vals)
    return {
        't_stat': float(t_stat),
        'p_value': float(p_value),
        'n_pairs': len(a_vals),
        'mean_diff': float(np.mean(b_vals) - np.mean(a_vals)),
        'significant_005': p_value < 0.05,
        'significant_001': p_value < 0.01,
    }


# ============================================================================
# Main
# ============================================================================

def main():
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║                                                                      ║
║   🔬 RIGOROUS EXPERIMENT: Multi-Seed × Multi-Arch × Multi-Dataset   ║
║                                                                      ║
║   3 Architectures × 2 Datasets × 5 Conditions × 3 Seeds = 90 runs   ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝
    """)

    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  Device: {device}")

    ARCHS = ['SmallResNet', 'TinyVGG', 'CompactDenseNet']
    DATASETS = ['cifar10', 'cifar100']
    CONDITIONS = ['baseline', 'dropout', 'label_smooth', 'weight_decay', 'social_dropout']
    SEEDS = [42, 123, 456]
    EPOCHS = 15

    total_runs = len(ARCHS) * len(DATASETS) * len(CONDITIONS) * len(SEEDS)
    all_results = []
    run_count = 0

    # Check for checkpoint (resume support)
    checkpoint_path = os.path.join(output_dir, 'rigorous_checkpoint.json')
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, 'r') as f:
            all_results = json.load(f)
        completed = {(r['arch'], r['dataset'], r['condition'], r['seed'])
                     for r in all_results}
        run_count = len(all_results)
        print(f"  📂 Resuming from checkpoint: {run_count}/{total_runs} completed")
    else:
        completed = set()

    print(f"\n  📋 Total runs: {total_runs}")
    print(f"  ⏱️  Estimated time: ~{total_runs * 5:.0f} min ({total_runs * 5 / 60:.1f} hours)\n")

    for arch in ARCHS:
        for dataset in DATASETS:
            for condition in CONDITIONS:
                for seed in SEEDS:
                    key = (arch, dataset, condition, seed)
                    if key in completed:
                        continue

                    run_count += 1
                    print(f"\n{'='*70}")
                    print(f"  [{run_count}/{total_runs}] {arch} | {dataset} | "
                          f"{condition} | seed={seed}")
                    print(f"{'='*70}")

                    t0 = time.time()
                    try:
                        result, _ = train_one_run(
                            arch, dataset, condition, seed,
                            epochs=EPOCHS, device=device
                        )
                        elapsed = time.time() - t0
                        result['time_seconds'] = elapsed

                        print(f"  ✅ Acc: {result['final_test_acc']:.2f}% | "
                              f"Gap: {result['gen_gap']:.2f}% | "
                              f"Robust: {result['noisy_acc']:.2f}% | "
                              f"Time: {elapsed:.1f}s")
                        print(f"     Personality: E={result['personality']['E']:.2f} "
                              f"N={result['personality']['N']:.2f} "
                              f"O={result['personality']['O']:.2f} "
                              f"A={result['personality']['A']:.2f} "
                              f"C={result['personality']['C']:.2f}")

                        all_results.append(result)

                        # Save checkpoint every 5 runs
                        if len(all_results) % 5 == 0:
                            with open(checkpoint_path, 'w') as f:
                                json.dump(all_results, f, indent=2, ensure_ascii=False)
                            print(f"  💾 Checkpoint saved ({len(all_results)}/{total_runs})")

                    except Exception as e:
                        print(f"  ❌ FAILED: {e}")
                        import traceback
                        traceback.print_exc()

    # ================================================================
    # Save final results
    # ================================================================
    results_path = os.path.join(output_dir, 'rigorous_results.json')
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\n💾 All results saved: {results_path}")

    # ================================================================
    # Statistical Analysis
    # ================================================================
    print("\n" + "="*70)
    print("📊 STATISTICAL ANALYSIS")
    print("="*70)

    stats = compute_statistics(all_results)

    # Print summary table
    print(f"\n{'Arch':<18} {'Dataset':<10} {'Condition':<16} "
          f"{'Acc (mean±std)':<18} {'Gen Gap':<12} {'Robust':<12}")
    print("-" * 86)
    for key in sorted(stats.keys()):
        s = stats[key]
        print(f"{s['arch']:<18} {s['dataset']:<10} {s['condition']:<16} "
              f"{s['acc_mean']:.2f}±{s['acc_std']:.2f}       "
              f"{s['gap_mean']:.2f}±{s['gap_std']:.2f}  "
              f"{s['robust_mean']:.2f}±{s['robust_std']:.2f}")

    # Paired t-tests: Social Dropout vs each baseline
    print("\n" + "="*70)
    print("📊 PAIRED T-TESTS: Social Dropout vs Others")
    print("="*70)

    for cond in ['baseline', 'dropout', 'label_smooth', 'weight_decay']:
        tt = paired_ttest(all_results, cond, 'social_dropout')
        sig = '***' if tt['significant_001'] else ('**' if tt['significant_005'] else 'n.s.')
        print(f"  Social Dropout vs {cond:<15}: "
              f"Δ={tt['mean_diff']:+.2f}%  t={tt['t_stat']:.3f}  "
              f"p={tt['p_value']:.4f}  {sig}  (n={tt['n_pairs']})")

    # Save stats
    stats_serializable = {f"{k[0]}_{k[1]}_{k[2]}": v for k, v in stats.items()}
    stats_path = os.path.join(output_dir, 'rigorous_stats.json')
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(stats_serializable, f, indent=2, ensure_ascii=False)

    # Clean up checkpoint
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)

    print(f"""
╔══════════════════════════════════════════════════════════════════════╗
║  ✅ RIGOROUS EXPERIMENT COMPLETE                                     ║
║  📊 {len(all_results)} runs completed, results saved.                ║
║  📁 Files: rigorous_results.json, rigorous_stats.json                ║
╚══════════════════════════════════════════════════════════════════════╝
    """)


if __name__ == '__main__':
    main()
