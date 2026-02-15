# SAM NEURAL ARCHITECTURE & C-CORE INTEGRATION

## Overview

This document describes the complete architecture integrating:
- **C Core**: Heavy computation (neural networks, SAM AGI, MuZero)
- **Python Bridge**: Interface layer (`sam_c_bridge.py`)
- **Automation Builder**: Orchestration (`src/python/automation/core.py`)

## Directory Structure

```
NN_C/
├── ORGANIZED/                    # C CORE (1.9M lines, restored from git)
│   ├── UTILS/utils/NN/           # Neural network implementations
│   │   ├── GNN/                  # Graph Neural Networks
│   │   ├── RNN/                  # Recurrent Neural Networks
│   │   ├── CNN/                  # Convolutional Neural Networks
│   │   ├── GAN/                  # Generative Adversarial Networks
│   │   ├── KAN/                  # Kolmogorov-Arnold Networks
│   │   ├── SNN/                  # Spiking Neural Networks
│   │   ├── NEAT/                 # NeuroEvolution
│   │   ├── TRANSFORMER/          # Attention mechanisms
│   │   └── MUZE/                 # MuZero implementation
│   │       ├── mcts.c            # Monte Carlo Tree Search
│   │       ├── ewc.c             # Elastic Weight Consolidation
│   │       ├── growth.c          # Morphogenetic growth
│   │       └── muzero_model.c    # World model
│   ├── UTILS/SAM/SAM/            # SAM AGI Core
│   │   ├── SAM.c                 # Main AGI system
│   │   ├── sam_morphogenesis.c   # Growth primitives
│   │   └── SAM_MUZE_BRIDGE/      # SAM-MUZE integration
│   ├── MODELS/                   # Training stages
│   │   ├── STAGE1/               # Basic word prediction
│   │   ├── STAGE2/               # Context awareness
│   │   ├── STAGE3/               # Phrase extraction
│   │   ├── STAGE4/               # Hybrid actions
│   │   └── STAGE5/               # MCTS integration
│   └── PROJECTS/                 # Applications
│       ├── RL_AGENT/             # Reinforcement learning
│       └── GAME/                 # Game AI
│
├── automation_framework/         # RUST ORCHESTRATION TOOL
│   └── src/                      # Rust source
│       ├── governance.rs         # Tri-cameral governance
│       ├── completeness.rs       # Completeness verification
│       └── python_bindings.rs    # PyO3 bindings
│
└── src/python/                   # PYTHON COORDINATION
    ├── automation/
    │   └── core.py               # Main automation (MODIFIED)
    ├── sam_c_bridge.py           # C-Python bridge (NEW)
    ├── sam_neural_integration.py # SAM components (NEW)
    └── complete_sam_unified.py   # Full SAM system
```

## Architecture Principles

### 1. C Core = Heavy Computation
**Location**: `ORGANIZED/`
- **Neural Networks**: 555 C/H files implementing all major architectures
- **SAM AGI**: Self-modifying AI with morphogenesis, identity preservation
- **MUZE**: MuZero with MCTS, world models, self-play
- **Optimizers**: Adam, SGD, RMSprop, Natural GD, BFGS, Newton

**Why C?**
- Speed: Bare-metal performance for neural operations
- Memory: Direct control over allocations
- Safety: Explicit resource management
- Deployment: Portable, no dependencies

### 2. Python Bridge = Interface Layer
**Location**: `src/python/sam_c_bridge.py`

**Responsibilities**:
- Load C libraries (.so/.dylib files)
- Convert Python/numpy to C structures
- Provide fallback implementations
- Handle compilation when needed
- Expose clean Python API

**Key Classes**:
```python
NeuralNetworkCore    # Wraps C neural nets
SAMCore             # Wraps SAM AGI state
MUZECore            # Wraps MuZero
COptimizerBridge    # Wraps C optimizers
SAMBuilderBridge    # Main integration point
```

### 3. Rust = Automation Tool Only
**Location**: `automation_framework/`

**Responsibilities**:
- Subagent orchestration
- Resource tracking/billing
- Governance (CIC/AEE/CSF)
- Workflow management
- **NOT**: Neural computation (that's C)

### 4. Python Builder = Orchestration
**Location**: `src/python/automation/core.py`

**Responsibilities**:
- File processing pipeline
- Iteration management
- Completeness tracking
- Use SAM components via `sam_c_bridge`

## Key Components

### Neural Networks in C

Each network type has:
- `*_create()`: Initialize network
- `*_forward()`: Forward pass
- `*_backward()`: Backpropagation
- `*_train()`: Training loop
- `*_save()`/`*_load()`: Persistence

**Types Available**:
- **GNN**: Graph operations, message passing
- **RNN**: Sequential data, LSTM/GRU
- **CNN**: Convolution, pooling
- **GAN**: Generator/discriminator
- **KAN**: Spline-based universal approximation
- **SNN**: Spiking, temporal dynamics
- **NEAT**: Evolutionary topology
- **Transformer**: Attention mechanisms

### SAM AGI State

From `SAM.h`:
```c
typedef struct {
    double *S;          // Latent state (morphogenetic)
    double *theta;      // Parameters
    double *Sigma;      // Identity manifold
    double U;           // Unsolvability budget
    double lr;          // Learning rate
    int timestamp;
} SAMState;
```

**Operations**:
- `SAM_step()`: Single timestep with all invariants
- `SAM_morphogenesis()`: Expand dimensions
- `SAM_identity_check()`: Preserve self
- `SAM_uncertainty_decay()`: Epistemic humility

### MUZE (MuZero)

Components:
- **MCTS**: `mcts.c` - Monte Carlo planning
- **World Model**: `muzero_model.c` - Dynamics + reward
- **EWC**: `ewc.c` - Catastrophic forgetting prevention
- **Replay**: `replay_buffer.c` - Experience storage
- **Self-Play**: `self_play.c` - Data generation

## Integration with Automation Builder

### Step 1: Import Bridge
```python
from sam_c_bridge import SAMBuilderBridge, get_bridge

# Create bridge
bridge = get_bridge()
```

### Step 2: Initialize
```python
bridge.initialize(
    latent_dim=64,
    optimizer_type='adam',
    network_type='MLP'
)
```

### Step 3: Process Chunks with SAM
```python
def process_chunk_with_sam(chunk_data, iteration):
    """Called from automation core for each chunk"""
    result = bridge.process_chunk_with_sam(
        chunk_data,
        iteration
    )
    
    return {
        'sam_state': result['sam_state'],
        'identity_overlap': result['identity_overlap'],
        'unsolvability': result['unsolvability'],
        'quality_score': compute_quality(result),
    }
```

### Step 4: Neural Optimization
```python
# During building phase
grad = compute_gradient(chunk)
optimized_params = bridge.optimizer.step(
    current_params,
    grad
)
```

## Compilation

### Compile Individual Networks
```bash
cd ORGANIZED/UTILS/utils/NN/GNN
make

# Creates: libgnn.so
```

### Compile SAM
```bash
cd ORGANIZED/UTILS/SAM/SAM
make -f Makefile_sam_test

# Creates: libsam.so
```

### Compile MUZE
```bash
cd ORGANIZED/UTILS/utils/NN/MUZE
make

# Creates: libmuze.so
```

### Auto-Compilation from Python
```python
from sam_c_bridge import NeuralNetworkCore

nn = NeuralNetworkCore("GNN")
nn.compile_if_needed()  # Auto-detects Makefile and compiles
```

## Fallback Behavior

When C libraries not available:
1. **Python implementations** are used automatically
2. **Performance**: ~10-100x slower but functionally equivalent
3. **Compilation**: Can be triggered manually or auto-detected
4. **Warnings**: Printed to console about missing libs

## Hard Invariants (Never Violated)

From `sam_c_bridge.py`:
```python
HardInvariant.check_all(state)
```

**Invariants**:
1. **Self-Preservation**: Σ manifold must not vanish
2. **Epistemic Rank**: Minimum covariance preserved
3. **Uncertainty**: U > 0 (always acknowledge limits)
4. **Identity**: Overlap with canonical self > threshold

## Growth Primitives

**Available Operations**:
- `EXPAND_LATENT`: Add dimensions to S
- `COMPRESS`: Reduce via distillation
- `BRANCH`: Create expert submodel
- `MERGE`: Combine similar experts
- `FORGET`: Prune low-importance weights
- `REPLAY`: Trigger experience replay
- `DISTILL`: Knowledge transfer
- `CURRICULUM`: Adjust difficulty

**Selection Policy**:
Based on pressure signals from meta-controller

## Usage Example

```python
#!/usr/bin/env python3
"""
Complete example integrating SAM with automation
"""

from sam_c_bridge import get_bridge
import numpy as np

# Initialize
bridge = get_bridge()
bridge.initialize(latent_dim=64)

# Process file chunks
for iteration in range(5):
    for chunk_id, chunk_data in enumerate(chunks):
        # SAM processing
        result = bridge.process_chunk_with_sam(
            chunk_data,
            iteration
        )
        
        # Check invariants
        if result['identity_overlap'] < 0.7:
            print(f"⚠️  Identity drift at chunk {chunk_id}")
        
        if result['unsolvability'] < 0.1:
            print(f"⚠️  High epistemic risk")
        
        # Extract features
        sam_state = result['sam_state']
        features = sam_state['S']
        
        # Store results...

# Get final status
print(bridge.get_status())
```

## Performance

**C Core**:
- Neural forward: ~1-10 μs
- SAM step: ~10-100 μs
- MCTS rollout: ~1-10 ms

**Python Bridge**:
- Overhead: ~10-100 μs per call
- Fallback: ~1-10 ms per operation

**Rust Tool**:
- Subagent spawn: ~1-10 ms
- Governance vote: ~1-100 μs

## Next Steps

1. **Compile C libraries** for your platform
2. **Test integration** with `sam_c_bridge.py`
3. **Modify core.py** to use bridge
4. **Add neural optimization** to builder
5. **Implement SAM state tracking** across iterations

## Files to Review

- `ORGANIZED/UTILS/utils/NN/MUZE/mcts.c` - MCTS implementation
- `ORGANIZED/UTILS/SAM/SAM/SAM.c` - SAM AGI core
- `ORGANIZED/UTILS/utils/NN/GNN/GNN.c` - Graph networks
- `src/python/sam_c_bridge.py` - Bridge implementation
- `src/python/sam_neural_integration.py` - Pure Python fallback

## References

- **God Equation**: See ChatGPT logs for full derivation
- **Morphogenesis**: See `sam_morphogenesis.c`
- **MuZero**: See Silver et al. 2019 + MUZE implementations
- **EWC**: See Kirkpatrick et al. 2017 + `ewc.c`
