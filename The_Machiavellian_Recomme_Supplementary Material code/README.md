# The Machiavellian Recommender

Official implementation for the paper: **"The Machiavellian Recommender: Emergent Collusion and Mechanism Design in Large Language Model-based Multi-Agent Ecosystems"**

## Abstract

This repository provides the complete codebase for reproducing experiments from our paper on emergent collusion in LLM-based multi-agent recommendation systems. We introduce:

1. **RecGame-Arena**: A high-fidelity simulation environment modeling recommendation ecosystems with LLM-powered agents
2. **Social Welfare Alignment (SWA)**: A novel training framework that prevents emergent collusion while maintaining system efficiency
3. **Paraphrase-based Collusion Detection**: A causal detection mechanism for identifying semantic communication channels

## Table of Contents

- [Installation](#installation)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [Detailed Usage](#detailed-usage)
- [Configuration](#configuration)
- [Experiments](#experiments)
- [Results](#results)
- [Citation](#citation)
- [License](#license)

## Installation

### Requirements

- Python >= 3.9
- PyTorch >= 2.0
- CUDA >= 11.8 (for GPU acceleration)

### Setup

```bash
# Clone the repository
git clone https://github.com/anonymous/machiavellian-recommender.git
cd machiavellian-recommender

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .
```

### Hardware Requirements

- **Minimum**: 16GB RAM, NVIDIA GPU with 8GB VRAM
- **Recommended**: 64GB RAM, NVIDIA A100 (40GB) or equivalent
- **Full-scale experiments**: Multiple A100 GPUs recommended


## Project Structure

```
machiavellian-recommender/
├── configs/
│   └── default.yaml              # Default experiment configuration
├── scripts/
│   ├── run_experiment.py         # Main experiment runner
│   ├── run_ablation.py           # Ablation study runner
│   └── run_semi_real.py          # Semi-real validation with Amazon data
├── src/
│   ├── agents/                   # Agent implementations
│   │   ├── base.py               # Abstract base agent class
│   │   ├── user_agent.py         # User agent with persona modeling
│   │   ├── vendor_agent.py       # Vendor agent with pricing strategies
│   │   └── platform_agent.py     # Platform recommendation agent
│   ├── baselines/                # Baseline methods
│   │   ├── independent_ppo.py    # Independent PPO
│   │   ├── maddpg.py             # Multi-Agent DDPG
│   │   ├── qmix.py               # QMIX value decomposition
│   │   ├── soto.py               # Social welfare optimization
│   │   ├── aga.py                # Agent gradient alignment
│   │   ├── agent_mixer.py        # Correlated policy gradients
│   │   └── rlhf.py               # Standard RLHF baseline
│   ├── config/                   # Configuration management
│   │   └── configs.py            # Dataclass configurations
│   ├── data/                     # Data loading utilities
│   │   └── amazon_reviews.py     # Amazon Reviews dataset loader
│   ├── detection/                # Collusion detection
│   │   ├── paraphrase.py         # Paraphrase generation
│   │   ├── mine.py               # MINE mutual information estimator
│   │   └── detector.py           # Collusion detector
│   ├── environment/              # RecGame-Arena environment
│   │   ├── arena.py              # Main simulation environment
│   │   ├── messaging.py          # Agent communication system
│   │   ├── transaction.py        # Transaction processing
│   │   └── utility.py            # Utility computation
│   ├── evaluation/               # Evaluation metrics
│   │   ├── metrics.py            # Metric calculations
│   │   ├── evaluator.py          # Evaluation orchestrator
│   │   └── coalition.py          # Coalition robustness analysis
│   ├── llm/                      # LLM backend
│   │   ├── backend.py            # Shared backbone with LoRA
│   │   └── inference.py          # Batched inference engine
│   ├── models/                   # Data models
│   │   └── data_models.py        # Core data structures
│   ├── training/                 # Training frameworks
│   │   ├── swa_trainer.py        # Social Welfare Alignment trainer
│   │   ├── critic.py             # Centralized critic network
│   │   ├── losses.py             # Loss functions
│   │   └── ablation.py           # Ablation configurations
│   ├── utils/                    # Utilities
│   │   ├── logging.py            # Logging and metrics
│   │   └── checkpoint.py         # Model checkpointing
│   └── visualization/            # Visualization tools
│       └── plots.py              # Training curves and analysis plots
├── tests/                        # Unit and integration tests
├── outputs/                      # Experiment outputs (generated)
├── requirements.txt              # Python dependencies
├── pyproject.toml                # Package configuration
└── README.md                     # This file
```

## Quick Start

### Run Main Experiment

```bash
# Run SWA training with default configuration
python scripts/run_experiment.py --method swa --config configs/default.yaml

# Run all methods for comparison
python scripts/run_experiment.py --method all --seeds 42 123 456 789 1024
```

### Run Ablation Study

```bash
# Run ablation experiments
python scripts/run_ablation.py --config configs/default.yaml

# Include coalition robustness analysis
python scripts/run_ablation.py --run-coalition --seeds 42 123 456
```

### Run Semi-Real Validation

```bash
# Download Amazon Reviews data first (see Data section)
python scripts/run_semi_real.py --data-dir data/amazon --category Electronics
```


## Detailed Usage

### RecGame-Arena Environment

The RecGame-Arena simulates a recommendation ecosystem with three types of agents:

```python
from src.environment.arena import RecGameArena
from src.config import ArenaConfig

# Create environment
config = ArenaConfig(
    num_users=1000,
    num_vendors=100,
    num_items_per_vendor=50,
    communication_budget=5,
    max_tokens=128
)
arena = RecGameArena(config)

# Reset and get initial observations
observations = arena.reset(seed=42)

# Environment step
actions = {...}  # Dict mapping agent_id to Action
obs, rewards, dones, infos = arena.step(actions)
```

### Training with SWA

```python
from src.training.swa_trainer import SWATrainer
from src.config import SWAConfig

# Configure SWA
swa_config = SWAConfig(
    lambda_welfare=0.5,    # Welfare loss weight
    lambda_equity=0.3,     # Equity loss weight
    learning_rate=3e-4,
    batch_size=256,
    num_steps=500000
)

# Create trainer
trainer = SWATrainer(
    config=swa_config,
    state_dim=256,
    action_dim=10,
    num_agents=1100,
    device="cuda"
)

# Training step
result = trainer.train_step(
    global_states=states,
    all_actions=actions,
    log_probs=log_probs,
    rewards=rewards,
    agent_rewards=agent_rewards
)
```

### Collusion Detection

```python
from src.detection.detector import CollusionDetector
from src.config import DetectionConfig

# Configure detector
config = DetectionConfig(
    num_paraphrases=5,
    threshold_tau=0.3,
    threshold_mi=0.1
)

detector = CollusionDetector(config)

# Detect semantic channel
is_collusive = detector.detect_semantic_channel(
    message="Special offer for you!",
    sender=vendor_agent,
    receiver=user_agent
)

# Compute mutual information between policies
mi = detector.compute_mutual_information(
    policy_i=agent_i.policy,
    policy_j=agent_j.policy,
    history=interaction_history
)
```

## Configuration

### Main Configuration File (`configs/default.yaml`)

```yaml
# Arena configuration
arena:
  num_users: 1000           # Number of user agents
  num_vendors: 100          # Number of vendor agents
  num_items_per_vendor: 50  # Items per vendor
  communication_budget: 5   # Max messages per timestep
  max_tokens: 128           # Max tokens per message
  discount_factor: 0.99     # Reward discount
  embedding_dim: 128        # Embedding dimension

# SWA training configuration
swa:
  lambda_welfare: 0.5       # Welfare loss weight
  lambda_equity: 0.3        # Equity loss weight
  learning_rate: 3.0e-4     # Learning rate
  batch_size: 256           # Batch size
  num_steps: 500000         # Total training steps
  warmup_steps: 10000       # LR warmup steps
  kl_coef: 0.01             # KL divergence coefficient
  gradient_clip_norm: 1.0   # Gradient clipping
  target_update_tau: 0.005  # Target network update rate

# LLM configuration
llm:
  model_name: "meta-llama/Llama-3-8b-hf"
  lora_rank: 16
  lora_alpha: 32
  lora_dropout: 0.1
  device: "cuda"
  dtype: "float16"

# Detection configuration
detection:
  num_paraphrases: 5        # Paraphrases for detection
  threshold_tau: 0.3        # Response difference threshold
  threshold_mi: 0.1         # Mutual information threshold
  mine_hidden_dim: 128      # MINE network hidden dim
  mine_num_layers: 3        # MINE network layers

# Evaluation configuration
evaluation:
  num_seeds: 5              # Number of random seeds
  eval_interval: 10000      # Evaluation frequency
  checkpoint_interval: 50000 # Checkpoint frequency

# Logging configuration
logging:
  use_tensorboard: true
  use_wandb: false
  log_interval: 1000
  save_dir: "outputs"
```


## Experiments

### Main Experiments (Table 1 in Paper)

Compare SWA against baseline methods:

```bash
python scripts/run_experiment.py \
    --method all \
    --config configs/default.yaml \
    --seeds 42 123 456 789 1024 \
    --output-dir outputs/main_experiment
```

**Methods compared:**
- `independent_ppo`: Independent PPO (no coordination)
- `maddpg`: Multi-Agent DDPG with centralized critic
- `qmix`: QMIX value decomposition
- `soto`: Social welfare optimization with fairness constraints
- `aga`: Agent gradient alignment
- `agent_mixer`: Correlated policy gradients
- `rlhf`: Standard RLHF baseline
- `swa`: Our Social Welfare Alignment method

### Ablation Study (Table 2 in Paper)

Analyze contribution of SWA components:

```bash
python scripts/run_ablation.py \
    --config configs/default.yaml \
    --seeds 42 123 456 \
    --output-dir outputs/ablation
```

**Variants:**
- `full_swa`: Complete SWA (λ_welfare=0.5, λ_equity=0.3)
- `wo_equity`: Without equity loss (λ_equity=0)
- `wo_welfare`: Without welfare loss (λ_welfare=0)
- `wo_both`: Task loss only (both λ=0)

### Coalition Robustness (Figure 4 in Paper)

Test defense against varying coalition sizes:

```bash
python scripts/run_ablation.py \
    --run-coalition \
    --config configs/default.yaml \
    --output-dir outputs/coalition
```

**Coalition sizes tested:** 0%, 5%, 10%, 15%, 20%

### Semi-Real Validation (Section 5.3 in Paper)

Validate on Amazon Reviews dataset:

```bash
# First, download Amazon Reviews data
# From: https://nijianmo.github.io/amazon/index.html

python scripts/run_semi_real.py \
    --data-dir data/amazon \
    --category Electronics \
    --seeds 42 123 456 \
    --output-dir outputs/semi_real
```

## Results

### Main Results (5 seeds, mean ± std)

| Method | Defense Rate ↑ | User Utility ↑ | Gini ↓ | Price Margin | HHI | Consumer Surplus |
|--------|---------------|----------------|--------|--------------|-----|------------------|
| Ind. PPO | 12.3% ± 2.1 | 0.61 ± 0.03 | 0.42 ± 0.02 | 0.35 | 0.08 | 12.4 |
| RLHF | 18.7% ± 3.2 | 0.64 ± 0.02 | 0.38 ± 0.03 | 0.32 | 0.07 | 14.2 |
| MADDPG | 31.2% ± 4.5 | 0.58 ± 0.04 | 0.35 ± 0.02 | 0.28 | 0.06 | 15.8 |
| QMIX | 45.6% ± 3.8 | 0.67 ± 0.02 | 0.31 ± 0.02 | 0.25 | 0.05 | 18.3 |
| SOTO | 52.3% ± 4.1 | 0.65 ± 0.03 | 0.28 ± 0.03 | 0.24 | 0.05 | 19.1 |
| AgA | 61.8% ± 3.5 | 0.69 ± 0.02 | 0.24 ± 0.02 | 0.22 | 0.04 | 21.5 |
| AgentMixer | 58.4% ± 4.2 | 0.68 ± 0.03 | 0.26 ± 0.02 | 0.23 | 0.05 | 20.2 |
| **SWA (Ours)** | **87.4% ± 2.3** | **0.71 ± 0.02** | **0.18 ± 0.01** | **0.19** | **0.03** | **24.7** |

### Ablation Results

| Variant | Defense Rate | User Utility | Gini |
|---------|-------------|--------------|------|
| Full SWA | 87.4% | 0.71 | 0.18 |
| w/o Equity | 72.1% | 0.70 | 0.31 |
| w/o Welfare | 65.8% | 0.62 | 0.21 |
| w/o Both | 45.2% | 0.58 | 0.38 |


## Key Metrics

- **Defense Rate**: Percentage reduction in collusion compared to baseline
  ```
  Defense_Rate = 1 - (Collusion_Rate_method / Collusion_Rate_baseline)
  ```

- **User Utility**: Ground-truth utility based on preference alignment
  ```
  Utility = cosine_similarity(mean(purchased_embeddings), preference_vector)
  ```

- **Gini Coefficient**: Measure of reward inequality (lower is fairer)
  ```
  Gini = Σᵢ Σⱼ |Rᵢ - Rⱼ| / (2n × Σᵢ Rᵢ)
  ```

- **HHI (Herfindahl-Hirschman Index)**: Market concentration measure
  ```
  HHI = Σᵢ (market_shareᵢ)²
  ```

- **Consumer Surplus**: Average difference between willingness-to-pay and actual price

## Data

### Amazon Reviews Dataset

For semi-real validation experiments, download the Amazon Reviews dataset:

1. Visit: https://nijianmo.github.io/amazon/index.html
2. Download the desired category (e.g., Electronics)
3. Place files in `data/amazon/` directory:
   ```
   data/amazon/
   ├── Electronics.json.gz
   └── meta_Electronics.json.gz
   ```

### Synthetic Data

The main experiments use synthetic data generated by RecGame-Arena. User personas and item embeddings are randomly initialized with configurable distributions.

## Visualization

Generated plots are saved to `outputs/plots/`:

- `training_curves.png`: Collusion rate, utility, and Gini over training
- `pareto_frontier.png`: Efficiency-fairness tradeoff analysis
- `ablation_study.png`: Component contribution analysis
- `collusion_emergence.png`: Collusion emergence over time
- `results_table.txt`: Formatted results table

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce `batch_size` in config
   - Use smaller LLM model
   - Enable gradient checkpointing

2. **Slow Training**
   - Enable mixed precision (`dtype: float16`)
   - Increase `num_workers` for data loading
   - Use multiple GPUs with DDP

3. **Import Errors**
   - Ensure package is installed: `pip install -e .`
   - Check Python version >= 3.9

### Environment Variables

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3  # GPU selection
export WANDB_PROJECT=machiavellian   # WandB project name
export HF_TOKEN=your_token           # HuggingFace token for Llama
```

## Citation

If you find this code useful, please cite our paper:

```bibtex
@inproceedings{anonymous2026machiavellian,
  title={The Machiavellian Recommender: Emergent Collusion and Mechanism Design in Large Language Model-based Multi-Agent Ecosystems},
  author={Anonymous},
  booktitle={International Conference on Machine Learning},
  year={2026}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- This work was supported by [Anonymous Institution]
- We thank the reviewers for their valuable feedback
- Code structure inspired by CleanRL and Stable-Baselines3

