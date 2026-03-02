"""
================================================================================
🔬 Social Dropout Experiment: CIFAR-10 Training
================================================================================

Trains ResNet-18 on CIFAR-10 with two configurations:
  1. Baseline: Standard training with regular Dropout
  2. Social Dropout: Training with Wasserstein cross-layer regularization

Compares:
  - Test accuracy
  - Personality trait changes (before vs after Social Dropout)
  - Training dynamics

Usage:
    python run_social_dropout_experiment.py

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
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from neural_personality import NeuralBigFiveAssessment
from social_dropout import SocialDropout

matplotlib.rcParams['font.family'] = ['Microsoft YaHei', 'SimHei', 'sans-serif']


# ============================================================================
# Simple ResNet-18 for CIFAR-10
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
    """Compact ResNet for CIFAR-10 (fast training on CPU)."""
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
# Training Loop
# ============================================================================

def get_cifar10(batch_size=128):
    """Load CIFAR-10 dataset."""
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

    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, 
                            num_workers=0, pin_memory=True)
    testloader = DataLoader(testset, batch_size=256, shuffle=False, 
                           num_workers=0, pin_memory=True)
    return trainloader, testloader


def evaluate(model, testloader, device):
    """Evaluate accuracy on test set."""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in testloader:
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            _, predicted = outputs.max(1)
            total += y.size(0)
            correct += predicted.eq(y).sum().item()
    return 100.0 * correct / total


def train_model(model, trainloader, testloader, device, epochs=15,
                lr=0.05, social_dropout=None, label="Baseline"):
    """
    Train model and return history.
    """
    model = model.to(device)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    history = {
        'train_loss': [], 'test_acc': [],
        'social_loss': [], 'epoch_time': []
    }

    print(f"\n{'='*60}")
    print(f"🏋️ Training: {label}")
    print(f"{'='*60}")

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        running_social = 0.0
        n_batches = 0
        t0 = time.time()

        for x, y in trainloader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()

            outputs = model(x)
            loss = criterion(outputs, y)

            # Social Dropout regularization
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
        epoch_time = time.time() - t0

        avg_loss = running_loss / n_batches
        avg_social = running_social / n_batches
        test_acc = evaluate(model, testloader, device)

        history['train_loss'].append(avg_loss)
        history['test_acc'].append(test_acc)
        history['social_loss'].append(avg_social)
        history['epoch_time'].append(epoch_time)

        print(f"  Epoch {epoch+1:2d}/{epochs} | Loss: {avg_loss:.4f} | "
              f"Social: {avg_social:.4f} | Acc: {test_acc:.2f}% | "
              f"Time: {epoch_time:.1f}s")

    return history


# ============================================================================
# Visualization
# ============================================================================

def plot_training_comparison(hist_base, hist_social, output_dir):
    """Plot training curves comparison."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), facecolor='#0a0a1a')

    for ax in axes:
        ax.set_facecolor('#0a0a1a')
        ax.tick_params(colors='white')
        ax.spines['bottom'].set_color('white')
        ax.spines['left'].set_color('white')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    epochs = range(1, len(hist_base['train_loss']) + 1)

    # Loss
    axes[0].plot(epochs, hist_base['train_loss'], '-o', color='#FF6B6B',
                label='Baseline', markersize=4)
    axes[0].plot(epochs, hist_social['train_loss'], '-o', color='#4ECDC4',
                label='Social Dropout', markersize=4)
    axes[0].set_xlabel('Epoch', color='white')
    axes[0].set_ylabel('Training Loss', color='white')
    axes[0].set_title('Training Loss', color='white', fontweight='bold')
    axes[0].legend(facecolor='#1a1a2e', edgecolor='white', labelcolor='white')

    # Accuracy
    axes[1].plot(epochs, hist_base['test_acc'], '-o', color='#FF6B6B',
                label='Baseline', markersize=4)
    axes[1].plot(epochs, hist_social['test_acc'], '-o', color='#4ECDC4',
                label='Social Dropout', markersize=4)
    axes[1].set_xlabel('Epoch', color='white')
    axes[1].set_ylabel('Test Accuracy (%)', color='white')
    axes[1].set_title('Test Accuracy', color='white', fontweight='bold')
    axes[1].legend(facecolor='#1a1a2e', edgecolor='white', labelcolor='white')

    # Social Loss
    axes[2].plot(epochs, hist_social['social_loss'], '-o', color='#9B59B6',
                label='W₂ Regularization', markersize=4)
    axes[2].axhline(y=0, color='white', alpha=0.2, linestyle='--')
    axes[2].set_xlabel('Epoch', color='white')
    axes[2].set_ylabel('Social Loss (W₂)', color='white')
    axes[2].set_title('Social Dropout Loss\n(Cross-layer W₂ distance)',
                     color='white', fontweight='bold')
    axes[2].legend(facecolor='#1a1a2e', edgecolor='white', labelcolor='white')

    plt.suptitle('Social Dropout vs Baseline: Training Dynamics',
                fontsize=16, fontweight='bold', color='white', y=1.02)
    plt.tight_layout()
    path = os.path.join(output_dir, 'social_dropout_training.png')
    plt.savefig(path, dpi=200, bbox_inches='tight', facecolor='#0a0a1a')
    plt.close()
    print(f"  💾 Training comparison saved to: {path}")


def plot_personality_change(scores_before, scores_after, output_dir):
    """Plot personality change: before vs after Social Dropout."""
    traits = ['E\n(外向性)', 'N\n(神经质)', 'O\n(开放性)', 'A\n(宜人性)', 'C\n(尽责性)']
    x = np.arange(len(traits))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6), facecolor='#0a0a1a')
    ax.set_facecolor('#0a0a1a')

    bars1 = ax.bar(x - width/2, scores_before, width, label='Baseline',
                   color='#FF6B6B', alpha=0.85, edgecolor='white', linewidth=0.5)
    bars2 = ax.bar(x + width/2, scores_after, width, label='Social Dropout',
                   color='#4ECDC4', alpha=0.85, edgecolor='white', linewidth=0.5)

    # Add value labels
    for bar in bars1:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., h + 0.01,
               f'{h:.2f}', ha='center', va='bottom', color='white', fontsize=9)
    for bar in bars2:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., h + 0.01,
               f'{h:.2f}', ha='center', va='bottom', color='white', fontsize=9)

    # Arrows for changes
    for i in range(len(traits)):
        diff = scores_after[i] - scores_before[i]
        color = '#2ECC71' if diff > 0 else '#E74C3C'
        symbol = '↑' if diff > 0 else '↓'
        ax.text(x[i], max(scores_before[i], scores_after[i]) + 0.06,
               f'{symbol}{abs(diff):.2f}', ha='center', color=color,
               fontsize=11, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(traits, fontsize=12, color='white')
    ax.set_ylabel('Raw Score', color='white', fontsize=12)
    ax.set_title('Personality Change After Social Dropout Therapy\n'
                '心理干预前后人格变化', color='white', fontsize=14, fontweight='bold')
    ax.legend(facecolor='#1a1a2e', edgecolor='white', labelcolor='white', fontsize=11)
    ax.tick_params(colors='white')
    ax.spines['bottom'].set_color('white')
    ax.spines['left'].set_color('white')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    path = os.path.join(output_dir, 'personality_change.png')
    plt.savefig(path, dpi=200, bbox_inches='tight', facecolor='#0a0a1a')
    plt.close()
    print(f"  💾 Personality change plot saved to: {path}")


# ============================================================================
# Main
# ============================================================================

def main():
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║                                                                      ║
║   💊 SOCIAL DROPOUT EXPERIMENT                                      ║
║                                                                      ║
║   "Can psychological intervention improve a network's personality    ║
║    AND performance?"                                                 ║
║                                                                      ║
║   Setup: ResNet on CIFAR-10, Baseline vs Social Dropout              ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝
    """)

    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  Device: {device}")

    EPOCHS = 15
    SOCIAL_RATE = 0.05

    # ================================================================
    # Step 1: Load data
    # ================================================================
    print("\n📦 Loading CIFAR-10...")
    trainloader, testloader = get_cifar10(batch_size=128)
    print("  ✅ CIFAR-10 loaded.")

    # ================================================================
    # Step 2: Create two models with IDENTICAL initialization
    # ================================================================
    torch.manual_seed(42)
    model_baseline = SmallResNet(num_classes=10)
    model_social = SmallResNet(num_classes=10)
    model_social.load_state_dict(copy.deepcopy(model_baseline.state_dict()))
    print("  ✅ Both models initialized with same weights (seed=42)")

    # ================================================================
    # Step 3: Train Baseline
    # ================================================================
    hist_baseline = train_model(
        model_baseline, trainloader, testloader, device,
        epochs=EPOCHS, label="Baseline (Standard Training)"
    )

    # ================================================================
    # Step 4: Setup Social Dropout hooks THEN train
    # ================================================================
    social_dropout = SocialDropout(social_rate=SOCIAL_RATE, method='wasserstein')
    social_dropout.register_hooks(model_social)

    hist_social = train_model(
        model_social, trainloader, testloader, device,
        epochs=EPOCHS, social_dropout=social_dropout,
        label=f"Social Dropout (λ={SOCIAL_RATE})"
    )

    social_dropout.remove_hooks()

    # ================================================================
    # Step 5: Compare personalities
    # ================================================================
    print("\n" + "="*60)
    print("🧠 Personality Assessment: Before vs After Therapy")
    print("="*60)

    nbfa = NeuralBigFiveAssessment()
    nbfa.assess(model_baseline, "Baseline")
    nbfa.assess(model_social, "Social-Dropout")
    nbfa.normalize_across_models()
    nbfa.print_comparison_table()

    # Get raw scores for plotting
    base_raw = nbfa.results['Baseline']['raw_scores']
    social_raw = nbfa.results['Social-Dropout']['raw_scores']

    scores_before = [base_raw['E'], base_raw['N'], base_raw['O'],
                     base_raw['A'], base_raw['C']]
    scores_after = [social_raw['E'], social_raw['N'], social_raw['O'],
                    social_raw['A'], social_raw['C']]

    # Print diagnoses
    for name in ['Baseline', 'Social-Dropout']:
        print(nbfa.generate_diagnosis(name))

    # ================================================================
    # Step 6: Visualizations
    # ================================================================
    print("\n" + "="*60)
    print("🎨 Generating visualizations")
    print("="*60)

    plot_training_comparison(hist_baseline, hist_social, output_dir)
    plot_personality_change(scores_before, scores_after, output_dir)

    # ================================================================
    # Step 7: Save results
    # ================================================================
    acc_diff = hist_social['test_acc'][-1] - hist_baseline['test_acc'][-1]
    e_change = float(social_raw['E'] - base_raw['E'])

    results = {
        'baseline': {
            'final_acc': hist_baseline['test_acc'][-1],
            'best_acc': max(hist_baseline['test_acc']),
            'personality_raw': {k: float(v) for k, v in base_raw.items()},
            'mbti': nbfa.get_mbti('Baseline'),
            'history': {k: [float(x) for x in v] for k, v in hist_baseline.items()},
        },
        'social_dropout': {
            'social_rate': SOCIAL_RATE,
            'final_acc': hist_social['test_acc'][-1],
            'best_acc': max(hist_social['test_acc']),
            'personality_raw': {k: float(v) for k, v in social_raw.items()},
            'mbti': nbfa.get_mbti('Social-Dropout'),
            'history': {k: [float(x) for x in v] for k, v in hist_social.items()},
        },
        'improvement': {
            'acc_diff': acc_diff,
            'best_acc_diff': max(hist_social['test_acc']) - max(hist_baseline['test_acc']),
            'extraversion_change': e_change,
        }
    }

    results_path = os.path.join(output_dir, 'social_dropout_results.json')
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # ================================================================
    # Summary
    # ================================================================
    print(f"""
╔══════════════════════════════════════════════════════════════════════╗
║  ✅ SOCIAL DROPOUT EXPERIMENT COMPLETE                               ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                      ║
║  Baseline Accuracy:       {hist_baseline['test_acc'][-1]:6.2f}%                            
║  Social Dropout Accuracy: {hist_social['test_acc'][-1]:6.2f}%                            
║  Difference:              {acc_diff:+6.2f}%                            
║                                                                      ║
║  Baseline MBTI:           {nbfa.get_mbti('Baseline'):<8s}                              
║  After Therapy MBTI:      {nbfa.get_mbti('Social-Dropout'):<8s}                              
║  Extraversion Change:     {e_change:+.4f}                            
║                                                                      ║
║  📝 Conclusion:                                                      ║
""")

    if acc_diff > 0 and e_change > 0:
        print("║  ✅ 心理干预成功！Social Dropout 同时提升了性能和外向性。   ║")
        print("║  → 证明网络的社交困难确实是性能瓶颈之一。                    ║")
    elif acc_diff > 0:
        print("║  ✅ 性能提升但人格未变——说明心理治疗提升了自信而非性格。     ║")
    elif e_change > 0:
        print("║  ✅ 外向性提升但性能持平——证明内向不影响智力，令人欣慰。     ║")
    else:
        print("║  🤔 两项均无显著变化——该网络可能天生就是这个性格。           ║")
        print("║  → 建议论文结论：'Some networks are just introverted,       ║")
        print("║    and that's okay.' — 内向不是病，别强迫它社交。           ║")

    print("║                                                                      ║")
    print("╚══════════════════════════════════════════════════════════════════════╝")


if __name__ == '__main__':
    main()
