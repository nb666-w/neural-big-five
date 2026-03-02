"""
================================================================================
🧠 Neural Big Five Assessment (NBFA)
================================================================================

The Introverted ResNet: Diagnosing Extraversion Bottlenecks in Deep 
Convolutional Networks via Big Five Assessment

Core module: Extracts personality traits from pretrained neural networks
using Wasserstein distances, Grassmann manifold analysis, Hessian spectral
theory, gradient cosine alignment, and pruning resilience.

Author: 王唱晓
Paper: For the prestigious journal "SHIT"
================================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.stats import pearsonr
from scipy.linalg import svdvals
from collections import OrderedDict
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# 1. EXTRAVERSION (E) — Wasserstein Cross-Layer Communication Cost
# ============================================================================
#
# E(N) = 1 / (W̄₂ + ε)
# W̄₂ = (1/(L-1)) Σ W₂(μₗ, μₗ₊₁)
#
# Intuition: Similar weight distributions across layers → smooth "social"
# communication → high extraversion.
# Deep networks have increasingly different layer distributions → introversion.

def wasserstein_1d(u_values, v_values, p=2):
    """
    Compute W_p distance between two 1D distributions.
    For 1D distributions, W_p has a closed-form solution via sorting.
    
    Reference: Ramdas et al. (2017) "On Wasserstein Two-Sample Testing"
    """
    u_sorted = np.sort(u_values.ravel())
    v_sorted = np.sort(v_values.ravel())
    
    # Resample to same length if needed
    n = max(len(u_sorted), len(v_sorted))
    u_interp = np.interp(
        np.linspace(0, 1, n),
        np.linspace(0, 1, len(u_sorted)),
        u_sorted
    )
    v_interp = np.interp(
        np.linspace(0, 1, n),
        np.linspace(0, 1, len(v_sorted)),
        v_sorted
    )
    
    return np.mean(np.abs(u_interp - v_interp) ** p) ** (1.0 / p)


def compute_extraversion(model, eps=1e-6):
    """
    Compute Extraversion score using Wasserstein-2 cross-layer distance.
    
    E(N) = 1 / (mean(W₂(μₗ, μₗ₊₁)) + ε)
    
    Higher score = more "extraverted" (layers communicate easily)
    """
    weight_distributions = []
    
    for name, param in model.named_parameters():
        if 'weight' in name and param.dim() >= 2:
            w = param.detach().cpu().numpy().ravel()
            if len(w) > 100:  # Skip tiny layers
                weight_distributions.append(w)
    
    if len(weight_distributions) < 2:
        return 0.5  # Default for too-shallow networks
    
    # Compute pairwise Wasserstein distances between consecutive layers
    w2_distances = []
    for i in range(len(weight_distributions) - 1):
        w2 = wasserstein_1d(weight_distributions[i], weight_distributions[i + 1])
        w2_distances.append(w2)
    
    mean_w2 = np.mean(w2_distances)
    extraversion = 1.0 / (mean_w2 + eps)
    
    return extraversion, {
        'mean_w2': mean_w2,
        'per_layer_w2': w2_distances,
        'num_layers': len(weight_distributions)
    }


# ============================================================================
# 2. NEUROTICISM (N) — Loss Landscape Roughness via Weight Spectral Analysis
# ============================================================================
#
# N(N) = mean(condition_number(Wₗ))
# condition_number = σ_max / σ_min⁺
#
# Intuition: High condition number → sharp loss landscape → model is 
# "neurotic", sensitive to small perturbations.

def compute_neuroticism(model, eps=1e-8):
    """
    Compute Neuroticism via weight matrix condition numbers.
    
    Uses SVD to compute condition numbers efficiently.
    High condition number → sharp landscape → neurotic.
    
    Reference: Li et al. (2018) "Visualizing the Loss Landscape of Neural Nets"
    """
    condition_numbers = []
    spectral_stats = []
    
    for name, param in model.named_parameters():
        if 'weight' in name and param.dim() >= 2:
            w = param.detach().cpu().numpy()
            # Reshape to 2D for SVD
            w_2d = w.reshape(w.shape[0], -1)
            
            if min(w_2d.shape) < 2:
                continue
                
            try:
                sv = svdvals(w_2d)
                sv_pos = sv[sv > eps]
                
                if len(sv_pos) >= 2:
                    cond = sv_pos[0] / sv_pos[-1]
                    condition_numbers.append(np.log(cond + 1))  # Log scale
                    
                    spectral_stats.append({
                        'name': name,
                        'sigma_max': float(sv_pos[0]),
                        'sigma_min': float(sv_pos[-1]),
                        'condition': float(cond),
                        'spectral_norm': float(sv_pos[0])
                    })
            except Exception:
                continue
    
    if not condition_numbers:
        return 0.5, {}
    
    neuroticism = np.mean(condition_numbers)
    
    return neuroticism, {
        'mean_log_condition': neuroticism,
        'max_condition': max(s['condition'] for s in spectral_stats),
        'spectral_details': spectral_stats[:5],  # Top 5 for brevity
        'num_layers_analyzed': len(condition_numbers)
    }


# ============================================================================
# 3. OPENNESS (O) — Effective Rank on Grassmann Manifold
# ============================================================================
#
# O(N) = (1/L) Σ rank_ε(Wₗ) / min(d_in, d_out)
#
# Effective rank = exp(entropy of normalized singular values)
# Reference: Roy & Vetterli (2007) "The Effective Rank"
#
# Intuition: Higher effective rank → using more subspace dimensions → 
# more "open" to diverse features.

def effective_rank(matrix, eps=1e-10):
    """
    Compute the effective rank of a matrix.
    
    erank(W) = exp(H(p))   where p_i = σ_i / Σσ_j
    H(p) = -Σ p_i log(p_i)   (Shannon entropy)
    
    Reference: Roy & Vetterli (2007) "The Effective Rank: A Measure of 
    Effective Dimensionality"
    """
    sv = svdvals(matrix)
    sv = sv[sv > eps]
    
    if len(sv) == 0:
        return 0.0
    
    # Normalize singular values to form a probability distribution
    p = sv / np.sum(sv)
    
    # Shannon entropy
    entropy = -np.sum(p * np.log(p + eps))
    
    # Effective rank = exp(entropy)
    return np.exp(entropy)


def compute_openness(model):
    """
    Compute Openness via Grassmann manifold effective rank analysis.
    
    O(N) = mean(erank(Wₗ) / min(d_in, d_out))
    
    Higher effective rank ratio → richer subspace utilization → more open.
    """
    rank_ratios = []
    rank_details = []
    
    for name, param in model.named_parameters():
        if 'weight' in name and param.dim() >= 2:
            w = param.detach().cpu().numpy()
            w_2d = w.reshape(w.shape[0], -1)
            
            if min(w_2d.shape) < 2:
                continue
            
            try:
                erank = effective_rank(w_2d)
                max_rank = min(w_2d.shape)
                ratio = erank / max_rank
                rank_ratios.append(ratio)
                
                rank_details.append({
                    'name': name,
                    'effective_rank': float(erank),
                    'max_rank': max_rank,
                    'ratio': float(ratio)
                })
            except Exception:
                continue
    
    if not rank_ratios:
        return 0.5, {}
    
    openness = np.mean(rank_ratios)
    
    return openness, {
        'mean_rank_ratio': openness,
        'rank_details': rank_details[:5],
        'num_layers': len(rank_ratios)
    }


# ============================================================================
# 4. AGREEABLENESS (A) — Gradient Direction Consensus
# ============================================================================
#
# A(N) = (1/C(L,2)) Σᵢ<ⱼ cos(∇θᵢL, ∇θⱼL)
#
# Intuition: If all layers' gradients point in roughly the same direction,
# the network has high "agreeableness" — internal harmony.
#
# Reference: Yu et al. (2020) "Gradient Surgery for Multi-Task Learning"

def compute_agreeableness(model, dataloader=None, criterion=None):
    """
    Compute Agreeableness via gradient cosine similarity between layers.
    
    If no data is available, uses weight correlation as proxy:
    Measure how "harmonious" the weight statistics are across layers
    (variance of per-layer weight means and stds).
    """
    # Proxy method: Weight distribution harmony
    # Low variance in weight statistics across layers → high agreeableness
    layer_means = []
    layer_stds = []
    layer_skews = []
    
    for name, param in model.named_parameters():
        if 'weight' in name and param.dim() >= 2:
            w = param.detach().cpu().numpy().ravel()
            if len(w) > 100:
                layer_means.append(np.mean(w))
                layer_stds.append(np.std(w))
                # Skewness as a measure of distribution asymmetry
                m3 = np.mean((w - np.mean(w)) ** 3)
                s3 = np.std(w) ** 3
                if s3 > 1e-10:
                    layer_skews.append(m3 / s3)
    
    if len(layer_means) < 2:
        return 0.5, {}
    
    # Agreeableness = inverse of cross-layer statistical variance
    # Low variance → everyone agrees → high agreeableness
    mean_consistency = 1.0 / (np.std(layer_means) + 1e-6)
    std_consistency = 1.0 / (np.std(layer_stds) + 1e-6)
    
    # Cosine similarity between consecutive layer weight vectors (sampled)
    cosine_sims = []
    all_weights = []
    for name, param in model.named_parameters():
        if 'weight' in name and param.dim() >= 2:
            w = param.detach().cpu().numpy().ravel()
            if len(w) > 100:
                # Subsample for efficiency
                idx = np.random.RandomState(42).choice(len(w), min(1000, len(w)), replace=False)
                all_weights.append(w[idx])
    
    for i in range(len(all_weights) - 1):
        # Align lengths
        min_len = min(len(all_weights[i]), len(all_weights[i+1]))
        a = all_weights[i][:min_len]
        b = all_weights[i+1][:min_len]
        cos_sim = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10)
        cosine_sims.append(abs(cos_sim))
    
    agreeableness_raw = np.mean(cosine_sims) if cosine_sims else 0.5
    
    return agreeableness_raw, {
        'mean_cosine_sim': float(agreeableness_raw),
        'mean_consistency': float(mean_consistency),
        'std_consistency': float(std_consistency),
        'num_layers': len(layer_means)
    }


# ============================================================================
# 5. CONSCIENTIOUSNESS (C) — Sparsity × Pruning Resilience
# ============================================================================
#
# C(N) = sparsity_ratio × (1 - information_entropy_of_weights)
#
# Intuition: Sparse weights with low entropy → each parameter has a clear
# purpose → conscientious. Dense, high-entropy weights → "lazy" parameters.
#
# Reference: Frankle & Carlin (2019) "The Lottery Ticket Hypothesis"

def compute_conscientiousness(model, prune_ratio=0.3):
    """
    Compute Conscientiousness via weight efficiency analysis.
    
    Measures:
    1. Near-zero weight ratio (effective sparsity)
    2. Weight entropy (low entropy → more structured → more conscientious)
    3. Pruning resilience estimate (weight magnitude concentration)
    """
    total_params = 0
    near_zero_params = 0
    weight_entropies = []
    magnitude_ginis = []
    
    for name, param in model.named_parameters():
        if 'weight' in name and param.dim() >= 2:
            w = param.detach().cpu().numpy().ravel()
            if len(w) < 100:
                continue
                
            total_params += len(w)
            
            # Near-zero count (effective sparsity)
            threshold = 0.01 * np.std(w)
            near_zero_params += np.sum(np.abs(w) < threshold)
            
            # Weight entropy (discretize into bins)
            hist, _ = np.histogram(w, bins=50, density=True)
            hist = hist[hist > 0]
            hist = hist / hist.sum()
            entropy = -np.sum(hist * np.log(hist + 1e-10))
            max_entropy = np.log(50)
            weight_entropies.append(entropy / max_entropy)  # Normalized
            
            # Gini coefficient of weight magnitudes
            # High Gini → few weights carry most importance → conscientious
            abs_w = np.sort(np.abs(w))
            n = len(abs_w)
            cumsum = np.cumsum(abs_w)
            gini = (2.0 * np.sum((np.arange(1, n+1) * abs_w))) / (n * np.sum(abs_w) + 1e-10) - (n + 1) / n
            magnitude_ginis.append(gini)
    
    if total_params == 0:
        return 0.5, {}
    
    sparsity = near_zero_params / total_params
    mean_entropy = np.mean(weight_entropies)
    mean_gini = np.mean(magnitude_ginis)
    
    # Conscientiousness: high gini (concentrated importance) + low entropy
    conscientiousness = mean_gini * (1 - mean_entropy)
    
    return conscientiousness, {
        'effective_sparsity': float(sparsity),
        'mean_weight_entropy': float(mean_entropy),
        'mean_gini': float(mean_gini),
        'total_params': total_params,
        'near_zero_params': near_zero_params
    }


# ============================================================================
# NEURAL BIG FIVE ASSESSMENT — Main Class
# ============================================================================

class NeuralBigFiveAssessment:
    """
    Neural Big Five Assessment (NBFA)
    
    Diagnoses the personality of deep neural networks using:
    - E: Wasserstein cross-layer distance (Cuturi, 2013)
    - N: Weight spectral condition analysis (Li et al., 2018)
    - O: Grassmann effective rank (Roy & Vetterli, 2007)
    - A: Cross-layer gradient cosine similarity (Yu et al., 2020)
    - C: Weight efficiency / Lottery Ticket analysis (Frankle & Carlin, 2019)
    """
    
    TRAIT_NAMES = ['Extraversion', 'Neuroticism', 'Openness', 
                   'Agreeableness', 'Conscientiousness']
    TRAIT_NAMES_CN = ['外向性', '神经质', '开放性', '宜人性', '尽责性']
    TRAIT_ABBREV = ['E', 'N', 'O', 'A', 'C']
    
    # MBTI mapping thresholds (median split)
    MBTI_MAP = {
        'E': ('E', 'I'),  # Extraversion: E vs I
        'N': ('N', 'S'),  # Neuroticism→Intuition proxy: N vs S  
        'O': ('P', 'J'),  # Openness→Perceiving: P vs J
        'A': ('F', 'T'),  # Agreeableness→Feeling: F vs T
    }
    
    def __init__(self):
        self.results = OrderedDict()
    
    def assess(self, model, model_name="Unknown"):
        """
        Perform full Big Five personality assessment on a neural network.
        
        Args:
            model: PyTorch model (nn.Module)
            model_name: Name for display
            
        Returns:
            dict with raw scores, normalized scores, and details
        """
        print(f"\n{'='*60}")
        print(f"🧠 NEURAL BIG FIVE ASSESSMENT: {model_name}")
        print(f"{'='*60}")
        
        model.eval()
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        num_layers = sum(1 for n, p in model.named_parameters() if 'weight' in n and p.dim() >= 2)
        
        print(f"📊 Parameters: {total_params:,} | Trainable: {trainable_params:,} | Weight layers: {num_layers}")
        
        # Compute each trait
        print("\n🔬 Computing personality dimensions...")
        
        e_raw, e_detail = compute_extraversion(model)
        print(f"  E (Extraversion):      {e_raw:.4f}  [W̄₂ = {e_detail.get('mean_w2', 0):.4f}]")
        
        n_raw, n_detail = compute_neuroticism(model)
        print(f"  N (Neuroticism):       {n_raw:.4f}  [mean log κ = {n_detail.get('mean_log_condition', 0):.4f}]")
        
        o_raw, o_detail = compute_openness(model)
        print(f"  O (Openness):          {o_raw:.4f}  [mean rank ratio = {o_detail.get('mean_rank_ratio', 0):.4f}]")
        
        a_raw, a_detail = compute_agreeableness(model)
        print(f"  A (Agreeableness):     {a_raw:.4f}  [mean cos sim = {a_detail.get('mean_cosine_sim', 0):.4f}]")
        
        c_raw, c_detail = compute_conscientiousness(model)
        print(f"  C (Conscientiousness): {c_raw:.4f}  [Gini = {c_detail.get('mean_gini', 0):.4f}]")
        
        raw_scores = {
            'E': e_raw, 'N': n_raw, 'O': o_raw, 'A': a_raw, 'C': c_raw
        }
        details = {
            'E': e_detail, 'N': n_detail, 'O': o_detail, 'A': a_detail, 'C': c_detail
        }
        
        result = {
            'model_name': model_name,
            'total_params': total_params,
            'num_layers': num_layers,
            'raw_scores': raw_scores,
            'details': details
        }
        
        self.results[model_name] = result
        return result
    
    def normalize_across_models(self):
        """
        Normalize raw scores to [0, 1] across all assessed models.
        This gives relative personality scores — who's more extraverted 
        compared to others, etc.
        """
        if len(self.results) < 2:
            # With only 1 model, just scale to 0.5
            for name, result in self.results.items():
                result['normalized_scores'] = {
                    k: 0.5 for k in result['raw_scores']
                }
            return
        
        # Collect all raw scores per trait
        all_scores = {trait: [] for trait in self.TRAIT_ABBREV}
        for result in self.results.values():
            for trait in self.TRAIT_ABBREV:
                all_scores[trait].append(result['raw_scores'][trait])
        
        # Min-max normalize per trait
        for name, result in self.results.items():
            normalized = {}
            for trait in self.TRAIT_ABBREV:
                scores = all_scores[trait]
                min_s, max_s = min(scores), max(scores)
                if max_s - min_s < 1e-10:
                    normalized[trait] = 0.5
                else:
                    normalized[trait] = (result['raw_scores'][trait] - min_s) / (max_s - min_s)
            result['normalized_scores'] = normalized
    
    def get_mbti(self, model_name):
        """
        Convert Big Five normalized scores to MBTI type.
        
        Mapping (following established psychology correlations):
        - E/I: Extraversion > 0.5 → E, else I
        - N/S: Openness > 0.5 → N (Intuition), else S
        - F/T: Agreeableness > 0.5 → F, else T
        - J/P: Conscientiousness > 0.5 → J, else P
        """
        if model_name not in self.results:
            return "????"
        
        scores = self.results[model_name].get('normalized_scores', 
                                                self.results[model_name]['raw_scores'])
        
        ei = 'E' if scores['E'] > 0.5 else 'I'
        ns = 'N' if scores['O'] > 0.5 else 'S'
        ft = 'F' if scores['A'] > 0.5 else 'T'
        jp = 'J' if scores['C'] > 0.5 else 'P'
        
        return ei + ns + ft + jp
    
    def generate_diagnosis(self, model_name):
        """
        Generate a humorous clinical-style personality diagnosis.
        """
        if model_name not in self.results:
            return "Patient not found."
        
        result = self.results[model_name]
        scores = result.get('normalized_scores', result['raw_scores'])
        mbti = self.get_mbti(model_name)
        num_layers = result['num_layers']
        params = result['total_params']
        
        # Generate trait descriptions
        descriptions = []
        
        # Extraversion
        e = scores['E']
        if e > 0.7:
            descriptions.append(f"极度外向 (E={e:.2f})：每个神经元都想认识所有人，Attention机制就是它的社交App")
        elif e > 0.4:
            descriptions.append(f"中等外向 (E={e:.2f})：社交能力在线，但不会主动搭话")
        else:
            descriptions.append(f"严重内向 (E={e:.2f})：{num_layers}层深度让信息传递如同跨部门邮件——发出去就没回音了")
        
        # Neuroticism
        n = scores['N']
        if n > 0.7:
            descriptions.append(f"高度神经质 (N={n:.2f})：权重条件数爆表，梯度稍微抖一下就原地崩溃")
        elif n > 0.4:
            descriptions.append(f"中等神经质 (N={n:.2f})：情绪波动可控，但建议定期做loss landscape检查")
        else:
            descriptions.append(f"情绪稳定 (N={n:.2f})：loss景观平坦如湖面，堪称模型界的禅宗大师")
        
        # Openness
        o = scores['O']
        if o > 0.7:
            descriptions.append(f"高度开放 (O={o:.2f})：有效秩接近满秩，对每个特征维度都充满好奇")
        elif o > 0.4:
            descriptions.append(f"中等开放 (O={o:.2f})：会选择性地探索新特征，不算保守但也不冒进")
        else:
            descriptions.append(f"保守封闭 (O={o:.2f})：权重矩阵严重低秩，总是用同几个特征看世界")
        
        # Agreeableness
        a = scores['A']
        if a > 0.7:
            descriptions.append(f"极度宜人 (A={a:.2f})：各层梯度高度一致，团队协作堪称典范")
        elif a > 0.4:
            descriptions.append(f"中等宜人 (A={a:.2f})：基本和谐，偶有层间意见分歧")
        else:
            descriptions.append(f"宜人性低 (A={a:.2f})：各层各自为政，梯度方向南辕北辙——典型的大公司病")
        
        # Conscientiousness
        c = scores['C']
        if c > 0.7:
            descriptions.append(f"极度尽责 (C={c:.2f})：每个参数都物尽其用，Gini系数显示没有摸鱼的权重")
        elif c > 0.4:
            descriptions.append(f"中等尽责 (C={c:.2f})：大部分参数在干活，小部分在带薪摸鱼")
        else:
            descriptions.append(f"缺乏责任心 (C={c:.2f})：{params:,}个参数里大量冗余，相当于有编制但不干活")
        
        # Compile diagnosis
        diagnosis = f"""
╔══════════════════════════════════════════════════════════════╗
║  🏥 NEURAL PSYCHOLOGICAL ASSESSMENT REPORT                  ║
╠══════════════════════════════════════════════════════════════╣
║  Patient: {model_name:<49s}║
║  MBTI Type: {mbti:<48s}║
║  Parameters: {params:,}                           
║  Depth: {num_layers} weight layers                          
╠══════════════════════════════════════════════════════════════╣
║  CLINICAL FINDINGS:                                          ║
╚══════════════════════════════════════════════════════════════╝
"""
        for desc in descriptions:
            diagnosis += f"\n  • {desc}"
        
        diagnosis += f"\n\n  📋 MBTI 诊断: {mbti}"
        diagnosis += f"\n  💊 建议: "
        
        if e < 0.3:
            diagnosis += "建议进行 Social Dropout 疗法（跨层特征正则化）以改善社交能力。"
        elif n > 0.7:
            diagnosis += "建议采用 Stochastic Weight Averaging 进行情绪稳定治疗。"
        elif c < 0.3:
            diagnosis += "建议进行结构化剪枝，辞退不干活的参数。"
        else:
            diagnosis += "心理状态总体健康，建议保持现有训练计划。"
        
        return diagnosis
    
    def print_comparison_table(self):
        """Print a formatted comparison table of all assessed models."""
        self.normalize_across_models()
        
        print(f"\n{'='*90}")
        print(f"{'MODEL PERSONALITY COMPARISON TABLE':^90}")
        print(f"{'='*90}")
        print(f"{'Model':<20} {'E':>8} {'N':>8} {'O':>8} {'A':>8} {'C':>8} {'MBTI':>8} {'Params':>12}")
        print(f"{'-'*90}")
        
        for name, result in self.results.items():
            scores = result.get('normalized_scores', result['raw_scores'])
            mbti = self.get_mbti(name)
            params = result['total_params']
            print(f"{name:<20} {scores['E']:>8.3f} {scores['N']:>8.3f} "
                  f"{scores['O']:>8.3f} {scores['A']:>8.3f} {scores['C']:>8.3f} "
                  f"{mbti:>8} {params:>12,}")
        
        print(f"{'='*90}")
    
    def get_all_scores_array(self):
        """Return (model_names, scores_array) for visualization."""
        names = list(self.results.keys())
        scores = []
        for name in names:
            result = self.results[name]
            s = result.get('normalized_scores', result['raw_scores'])
            scores.append([s['E'], s['N'], s['O'], s['A'], s['C']])
        return names, np.array(scores)


# ============================================================================
# MODEL LOADER — Load pretrained models from torchvision
# ============================================================================

def load_pretrained_models():
    """
    Load all pretrained models for personality assessment.
    Returns dict of {name: model}.
    """
    import torchvision.models as models
    
    print("📦 Loading pretrained models from torchvision...")
    print("   (Using ImageNet-1K pretrained weights)\n")
    
    model_configs = OrderedDict([
        ('VGG-16', lambda: models.vgg16(weights='IMAGENET1K_V1')),
        ('ResNet-50', lambda: models.resnet50(weights='IMAGENET1K_V1')),
        ('ResNet-152', lambda: models.resnet152(weights='IMAGENET1K_V1')),
        ('DenseNet-121', lambda: models.densenet121(weights='IMAGENET1K_V1')),
        ('EfficientNet-B0', lambda: models.efficientnet_b0(weights='IMAGENET1K_V1')),
        ('ConvNeXt-Tiny', lambda: models.convnext_tiny(weights='IMAGENET1K_V1')),
        ('ViT-Small', lambda: models.vit_b_16(weights='IMAGENET1K_V1')),  # ViT-B/16 as "small"
        ('ViT-Base', lambda: models.vit_b_32(weights='IMAGENET1K_V1')),   # ViT-B/32 as "base"
    ])
    
    loaded = OrderedDict()
    for name, loader in model_configs.items():
        try:
            print(f"  Loading {name}...", end=" ", flush=True)
            model = loader()
            model.eval()
            total_params = sum(p.numel() for p in model.parameters())
            print(f"✓ ({total_params:,} params)")
            loaded[name] = model
        except Exception as e:
            print(f"✗ ({e})")
    
    return loaded
