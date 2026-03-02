# 🧠 The Introverted ResNet: Neural Big Five Assessment

**Diagnosing Neural Network Personalities via Big Five Assessment and Cross-Layer Optimal Transport Regularization**

> *Te Wang — School of Artificial Intelligence, Sun Yat-sen University*

## Overview

We propose the **Neural Big Five Assessment (NBFA)**, a framework that characterizes deep neural networks along five interpretable personality dimensions — drawing an analogy to the Big Five model in personality psychology:

| Trait | Metric | Interpretation |
|-------|--------|----------------|
| **E**xtraversion | Cross-layer Wasserstein distance | How "sociable" layers are |
| **N**euroticism | Weight condition numbers | Loss landscape sharpness |
| **O**penness | Effective rank (Grassmann) | Feature diversity |
| **A**greeableness | Gradient direction consensus | Internal harmony |
| **C**onscientiousness | Gini coefficient of weights | Parameter efficiency |

We also introduce **Social Dropout**, a cross-layer optimal transport regularization that can measurably shift a network's personality profile while improving generalization.

## Key Results

- **90-run rigorous experiment** (3 architectures × 2 datasets × 5 conditions × 3 seeds)
- Social Dropout significantly outperforms Standard Dropout (+1.88%, *p*=0.006)
- **4/5 traits** pass ANOVA for architecture dependence (η² > 0.90)
- Personality-guided ensemble selection matches best-individual selection (90.92% soft vote)

## Project Structure

```
├── neural_personality.py          # Core NBFA framework (5 trait computation)
├── social_dropout.py              # Social Dropout regularization
├── run_rigorous_experiment.py     # 90-run experiment matrix with t-tests
├── run_ensemble_experiment.py     # Personality-guided ensemble selection
├── run_validity_experiment.py     # Discriminant/predictive/convergent validity
├── generate_paper_assets.py       # Auto-generate LaTeX tables from results
├── run_experiment.py              # Original 8-model profiling experiment
├── run_social_dropout_experiment.py  # Social Dropout training demo
├── extra_viz.py                   # Correlation & performance visualizations
├── poincare_viz.py                # Poincaré ball embedding visualization
├── paper/
│   ├── main.tex                   # Full paper (LaTeX)
│   └── references.bib             # 26 citations
├── requirements.txt
└── README.md
```

## Quick Start

### Install dependencies
```bash
pip install torch torchvision numpy scipy matplotlib tqdm
```

### Run the full experiment pipeline
```bash
# 1. Rigorous 90-run experiment (~6-8h, supports checkpointing)
python run_rigorous_experiment.py

# 2. Ensemble experiment (~40 min)
python run_ensemble_experiment.py

# 3. Validity analysis (~1 min, needs rigorous results)
python run_validity_experiment.py

# 4. Generate paper tables
python generate_paper_assets.py


```

### Quick personality assessment of any model
```python
from neural_personality import NeuralBigFiveAssessment
import torchvision.models as models

model = models.resnet50(pretrained=True)
nbfa = NeuralBigFiveAssessment()
nbfa.assess(model, "ResNet-50")
nbfa.print_report()
```

## Citation

```bibtex
@article{wang2026introverted,
  title={The Introverted ResNet: Diagnosing Neural Network Personalities 
         via Big Five Assessment and Cross-Layer Optimal Transport Regularization},
  author={Wang, Te},
  year={2026}
}
```

## License

MIT License
