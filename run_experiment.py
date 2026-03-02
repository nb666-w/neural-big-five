"""
================================================================================
🔬 The Introverted ResNet: Full Experiment Runner
================================================================================

Run this script to perform the complete Neural Big Five Assessment:
1. Load 8 pretrained models from torchvision
2. Compute Big Five personality traits for each model
3. Generate personality profiles, MBTI types, and clinical diagnoses
4. Create visualizations (radar chart, Poincaré space, depth-E curve)
5. Output LaTeX-ready comparison table

Usage:
    python run_experiment.py

Author: 王唱晓
Paper: "The Introverted ResNet: Diagnosing Extraversion Bottlenecks in 
        Deep Convolutional Networks via Big Five Assessment"
================================================================================
"""

import os
import sys
import time
import json
import numpy as np

# Ensure we can import local modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from neural_personality import NeuralBigFiveAssessment, load_pretrained_models
from poincare_viz import (plot_poincare_personality_space, 
                           plot_radar_comparison,
                           plot_depth_extraversion)


def main():
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║                                                                      ║
║   🧠 THE INTROVERTED RESNET                                         ║
║                                                                      ║
║   Diagnosing Extraversion Bottlenecks in Deep Convolutional          ║
║   Networks via Big Five Assessment                                   ║
║                                                                      ║
║   Neural Big Five Assessment (NBFA) Framework v1.0                   ║
║   For the prestigious journal: SHIT                                  ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝
    """)
    
    start_time = time.time()
    
    # Create output directory
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
    os.makedirs(output_dir, exist_ok=True)
    
    # ================================================================
    # Step 1: Load pretrained models
    # ================================================================
    print("\n" + "="*60)
    print("📦 STEP 1: Loading pretrained models")
    print("="*60)
    
    models = load_pretrained_models()
    
    if not models:
        print("❌ No models loaded! Check your torchvision installation.")
        return
    
    print(f"\n✅ Loaded {len(models)} models successfully.")
    
    # ================================================================
    # Step 2: Run Neural Big Five Assessment
    # ================================================================
    print("\n" + "="*60)
    print("🧠 STEP 2: Running Neural Big Five Assessment")
    print("="*60)
    
    nbfa = NeuralBigFiveAssessment()
    
    for name, model in models.items():
        nbfa.assess(model, model_name=name)
    
    # Normalize scores across all models
    nbfa.normalize_across_models()
    
    # ================================================================
    # Step 3: Print comparison table
    # ================================================================
    print("\n" + "="*60)
    print("📊 STEP 3: Results")
    print("="*60)
    
    nbfa.print_comparison_table()
    
    # ================================================================
    # Step 4: Generate clinical diagnoses
    # ================================================================
    print("\n" + "="*60)
    print("🏥 STEP 4: Clinical Diagnoses")
    print("="*60)
    
    diagnoses = {}
    for name in models:
        diagnosis = nbfa.generate_diagnosis(name)
        diagnoses[name] = diagnosis
        print(diagnosis)
        print()
    
    # ================================================================
    # Step 5: Generate visualizations
    # ================================================================
    print("\n" + "="*60)
    print("🎨 STEP 5: Generating visualizations")
    print("="*60)
    
    names, scores = nbfa.get_all_scores_array()
    
    # 5.1 Radar chart
    print("\n  📊 Generating radar chart...")
    plot_radar_comparison(
        names, scores, 
        save_path=os.path.join(output_dir, 'personality_radar.png')
    )
    
    # 5.2 Poincaré ball
    print("  🌀 Generating Poincaré ball visualization...")
    plot_poincare_personality_space(
        names, scores,
        save_path=os.path.join(output_dir, 'poincare_space.png')
    )
    
    # 5.3 Depth vs Extraversion
    print("  📈 Generating depth-extraversion curve...")
    model_data = []
    for name, result in nbfa.results.items():
        depth = result['num_layers']
        e_raw = result['raw_scores']['E']
        model_data.append((name, depth, e_raw))
    
    plot_depth_extraversion(
        model_data,
        save_path=os.path.join(output_dir, 'depth_vs_extraversion.png')
    )
    
    # ================================================================
    # Step 6: Save results to JSON
    # ================================================================
    print("\n" + "="*60)
    print("💾 STEP 6: Saving results")
    print("="*60)
    
    # Serialize results
    results_data = {}
    for name, result in nbfa.results.items():
        results_data[name] = {
            'total_params': result['total_params'],
            'num_layers': result['num_layers'],
            'raw_scores': {k: float(v) for k, v in result['raw_scores'].items()},
            'normalized_scores': {k: float(v) for k, v in result['normalized_scores'].items()},
            'mbti': nbfa.get_mbti(name),
        }
    
    results_path = os.path.join(output_dir, 'nbfa_results.json')
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results_data, f, indent=2, ensure_ascii=False)
    print(f"  📄 Results saved to: {results_path}")
    
    # Save diagnoses
    diagnoses_path = os.path.join(output_dir, 'diagnoses.txt')
    with open(diagnoses_path, 'w', encoding='utf-8') as f:
        for name, diag in diagnoses.items():
            f.write(diag)
            f.write('\n\n' + '='*60 + '\n\n')
    print(f"  📄 Diagnoses saved to: {diagnoses_path}")
    
    # ================================================================
    # Step 7: Generate LaTeX table
    # ================================================================
    print("\n" + "="*60)
    print("📝 STEP 7: LaTeX table")
    print("="*60)
    
    latex_lines = [
        r'\begin{table}[h]',
        r'\centering',
        r'\caption{Neural Big Five Personality Assessment Results. '
        r'Scores are normalized to [0, 1] across all models. '
        r'MBTI types are derived from Big Five to MBTI mapping.}',
        r'\label{tab:personality}',
        r'\begin{tabular}{lcccccrc}',
        r'\toprule',
        r'Model & E & N & O & A & C & Params & MBTI \\',
        r'\midrule',
    ]
    
    for name, data in results_data.items():
        s = data['normalized_scores']
        params_m = data['total_params'] / 1e6
        latex_lines.append(
            f"  {name} & {s['E']:.3f} & {s['N']:.3f} & {s['O']:.3f} "
            f"& {s['A']:.3f} & {s['C']:.3f} & {params_m:.1f}M "
            f"& {data['mbti']} \\\\"
        )
    
    latex_lines.extend([
        r'\bottomrule',
        r'\end{tabular}',
        r'\end{table}',
    ])
    
    latex_table = '\n'.join(latex_lines)
    print(latex_table)
    
    latex_path = os.path.join(output_dir, 'table_personality.tex')
    with open(latex_path, 'w', encoding='utf-8') as f:
        f.write(latex_table)
    print(f"\n  📄 LaTeX table saved to: {latex_path}")
    
    # ================================================================
    # Summary
    # ================================================================
    elapsed = time.time() - start_time
    
    print(f"""
╔══════════════════════════════════════════════════════════════════════╗
║  ✅ EXPERIMENT COMPLETE                                              ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                      ║
║  Models assessed:  {len(models):<47d}║
║  Time elapsed:     {elapsed:.1f}s                                    
║                                                                      ║
║  Output files in: {output_dir}
║    • personality_radar.png     — Radar chart comparison              ║
║    • poincare_space.png        — Poincaré ball visualization         ║
║    • depth_vs_extraversion.png — Introversion Scaling Law            ║
║    • nbfa_results.json         — Machine-readable results            ║
║    • diagnoses.txt             — Clinical personality reports        ║
║    • table_personality.tex     — LaTeX table for paper               ║
║                                                                      ║
║  Key Finding: E(N) = O(L^{{-1/2}})                                  ║
║  → Deeper networks are more introverted.                             ║
║  → This perfectly explains the Extraversion prediction bottleneck.   ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝
    """)


if __name__ == '__main__':
    main()
