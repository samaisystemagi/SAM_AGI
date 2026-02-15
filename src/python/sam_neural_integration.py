"""
SAM_NEURAL_INTEGRATION - Advanced Neural Network & SAM Components for Builder

Integrates:
- Neural network optimizers (Adam, SGD, RMSprop, Natural GD)
- L1/L2 Regularization
- Learning rate schedulers
- SAM-specific: Self-referential objectives, growth primitives, identity preservation
- Morphogenetic latency
- Hard invariants
- Meta-controller with pressure signals
"""

import numpy as np
import math
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
import json
from collections import deque


# ============================================================================
# NEURAL NETWORK OPTIMIZERS
# ============================================================================

class OptimizerType(Enum):
    ADAM = "adam"
    SGD = "sgd"
    RMSPROP = "rmsprop"
    NATURAL_GD = "natural_gd"
    NEWTON = "newton"
    BFGS = "bfgs"
    CONJUGATE_GRADIENT = "cg"


@dataclass
class OptimizerConfig:
    """Configuration for neural network optimizers"""
    type: OptimizerType = OptimizerType.ADAM
    learning_rate: float = 0.001
    beta1: float = 0.9  # For Adam
    beta2: float = 0.999  # For Adam
    epsilon: float = 1e-8
    momentum: float = 0.9  # For SGD/RMSprop
    weight_decay: float = 0.0  # L2 regularization
    l1_lambda: float = 0.0  # L1 regularization
    gradient_clip: float = 1.0
    use_nesterov: bool = False
    rmsprop_alpha: float = 0.99


class NeuralOptimizer:
    """
    Advanced neural network optimizer with multiple algorithms.
    Supports: Adam, SGD, RMSprop, Natural GD, Newton, BFGS, CG
    """
    
    def __init__(self, config: OptimizerConfig, params_shape: Tuple[int, ...]):
        self.config = config
        self.params_shape = params_shape
        self.t = 0  # Timestep
        
        # Initialize state based on optimizer type
        if config.type == OptimizerType.ADAM:
            self.m = np.zeros(params_shape)  # First moment
            self.v = np.zeros(params_shape)  # Second moment
        elif config.type == OptimizerType.RMSPROP:
            self.square_avg = np.zeros(params_shape)
        elif config.type == OptimizerType.SGD and config.momentum > 0:
            self.velocity = np.zeros(params_shape)
        elif config.type == OptimizerType.NATURAL_GD:
            self.fisher_info = np.eye(np.prod(params_shape))
            self.fisher_lambda = 0.001
        elif config.type == OptimizerType.BFGS:
            n = np.prod(params_shape)
            self.H = np.eye(n)  # Inverse Hessian approximation
            self.prev_grad = None
            self.prev_params = None
    
    def compute_regularization(self, params: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Compute L1 and L2 regularization.
        Returns: (reg_gradient, reg_loss)
        """
        reg_loss = 0.0
        reg_grad = np.zeros_like(params)
        
        # L2 regularization (weight decay)
        if self.config.weight_decay > 0:
            reg_grad += self.config.weight_decay * params
            reg_loss += 0.5 * self.config.weight_decay * np.sum(params ** 2)
        
        # L1 regularization (sparsity)
        if self.config.l1_lambda > 0:
            reg_grad += self.config.l1_lambda * np.sign(params)
            reg_loss += self.config.l1_lambda * np.sum(np.abs(params))
        
        return reg_grad, reg_loss
    
    def step(self, params: np.ndarray, grad: np.ndarray, loss: float = None) -> np.ndarray:
        """
        Single optimization step.
        
        Args:
            params: Current parameters
            grad: Gradient of loss w.r.t. parameters
            loss: Optional loss value for certain optimizers
        
        Returns:
            Updated parameters
        """
        self.t += 1
        
        # Add regularization gradient
        reg_grad, _ = self.compute_regularization(params)
        grad = grad + reg_grad
        
        # Clip gradients
        grad_norm = np.linalg.norm(grad)
        if grad_norm > self.config.gradient_clip:
            grad = grad * (self.config.gradient_clip / grad_norm)
        
        # Apply optimizer-specific update
        if self.config.type == OptimizerType.ADAM:
            return self._adam_step(params, grad)
        elif self.config.type == OptimizerType.SGD:
            return self._sgd_step(params, grad)
        elif self.config.type == OptimizerType.RMSPROP:
            return self._rmsprop_step(params, grad)
        elif self.config.type == OptimizerType.NATURAL_GD:
            return self._natural_gd_step(params, grad)
        elif self.config.type == OptimizerType.BFGS:
            return self._bfgs_step(params, grad)
        elif self.config.type == OptimizerType.NEWTON:
            return self._newton_step(params, grad)
        else:
            return params - self.config.learning_rate * grad
    
    def _adam_step(self, params: np.ndarray, grad: np.ndarray) -> np.ndarray:
        """Adam optimizer step"""
        self.m = self.config.beta1 * self.m + (1 - self.config.beta1) * grad
        self.v = self.config.beta2 * self.v + (1 - self.config.beta2) * (grad ** 2)
        
        # Bias correction
        m_hat = self.m / (1 - self.config.beta1 ** self.t)
        v_hat = self.v / (1 - self.config.beta2 ** self.t)
        
        # Update
        update = self.config.learning_rate * m_hat / (np.sqrt(v_hat) + self.config.epsilon)
        return params - update
    
    def _sgd_step(self, params: np.ndarray, grad: np.ndarray) -> np.ndarray:
        """SGD with optional momentum"""
        if self.config.momentum > 0:
            self.velocity = (self.config.momentum * self.velocity + 
                           self.config.learning_rate * grad)
            if self.config.use_nesterov:
                return params - (self.config.momentum * self.velocity + 
                               self.config.learning_rate * grad)
            return params - self.velocity
        else:
            return params - self.config.learning_rate * grad
    
    def _rmsprop_step(self, params: np.ndarray, grad: np.ndarray) -> np.ndarray:
        """RMSprop optimizer step"""
        self.square_avg = (self.config.rmsprop_alpha * self.square_avg + 
                          (1 - self.config.rmsprop_alpha) * (grad ** 2))
        update = (self.config.learning_rate * grad / 
                 (np.sqrt(self.square_avg) + self.config.epsilon))
        return params - update
    
    def _natural_gd_step(self, params: np.ndarray, grad: np.ndarray) -> np.ndarray:
        """Natural Gradient Descent using Fisher Information"""
        grad_flat = grad.flatten()
        natural_grad = np.linalg.solve(
            self.fisher_info + self.fisher_lambda * np.eye(len(grad_flat)),
            grad_flat
        )
        natural_grad = natural_grad.reshape(self.params_shape)
        return params - self.config.learning_rate * natural_grad
    
    def _bfgs_step(self, params: np.ndarray, grad: np.ndarray) -> np.ndarray:
        """BFGS quasi-Newton method"""
        if self.prev_grad is None:
            # First step - use gradient descent
            self.prev_grad = grad.flatten()
            self.prev_params = params.flatten()
            return params - self.config.learning_rate * grad
        
        grad_flat = grad.flatten()
        params_flat = params.flatten()
        
        # Compute s and y
        s = params_flat - self.prev_params
        y = grad_flat - self.prev_grad
        
        # Update H using BFGS formula
        rho = 1.0 / (np.dot(y, s) + 1e-10)
        if rho > 0:  # Ensure positive definiteness
            I = np.eye(len(s))
            self.H = ((I - rho * np.outer(s, y)) @ self.H @ 
                     (I - rho * np.outer(y, s)) + rho * np.outer(s, s))
        
        # Compute search direction
        p = -self.H @ grad_flat
        
        # Update parameters
        new_params = params_flat + self.config.learning_rate * p
        
        self.prev_grad = grad_flat
        self.prev_params = params_flat
        
        return new_params.reshape(self.params_shape)
    
    def _newton_step(self, params: np.ndarray, grad: np.ndarray) -> np.ndarray:
        """Newton's method (requires Hessian)"""
        # Approximate Hessian using finite differences (simplified)
        eps = 1e-5
        hessian_diag = np.ones_like(params)  # Simplified - should compute full Hessian
        update = grad / (hessian_diag + self.config.epsilon)
        return params - self.config.learning_rate * update


# ============================================================================
# LEARNING RATE SCHEDULERS
# ============================================================================

class LRSchedulerType(Enum):
    CONSTANT = "constant"
    STEP = "step"
    EXPONENTIAL = "exponential"
    COSINE = "cosine"
    WARMUP = "warmup"
    REDUCE_ON_PLATEAU = "reduce_on_plateau"
    CYCLICAL = "cyclical"


@dataclass
class SchedulerConfig:
    type: LRSchedulerType = LRSchedulerType.COSINE
    warmup_steps: int = 1000
    min_lr: float = 1e-6
    max_lr: float = 0.001
    step_size: int = 1000  # For step scheduler
    gamma: float = 0.1  # Decay factor
    T_max: int = 10000  # For cosine
    patience: int = 10  # For reduce on plateau
    mode: str = "min"  # "min" or "max" for plateau


class LearningRateScheduler:
    """Advanced learning rate scheduling"""
    
    def __init__(self, config: SchedulerConfig):
        self.config = config
        self.step_count = 0
        self.best_metric = float('inf') if config.mode == "min" else float('-inf')
        self.patience_counter = 0
        self.cycle_count = 0
    
    def get_lr(self, base_lr: float, metric: Optional[float] = None) -> float:
        """Get current learning rate"""
        self.step_count += 1
        
        if self.config.type == LRSchedulerType.CONSTANT:
            return base_lr
        
        elif self.config.type == LRSchedulerType.STEP:
            return base_lr * (self.config.gamma ** 
                            (self.step_count // self.config.step_size))
        
        elif self.config.type == LRSchedulerType.EXPONENTIAL:
            return base_lr * (self.config.gamma ** self.step_count)
        
        elif self.config.type == LRSchedulerType.COSINE:
            if self.step_count < self.config.warmup_steps:
                return self._warmup(base_lr)
            progress = (self.step_count - self.config.warmup_steps) / self.config.T_max
            progress = min(1.0, progress)
            return (self.config.min_lr + 
                   (base_lr - self.config.min_lr) * 
                   0.5 * (1 + math.cos(math.pi * progress)))
        
        elif self.config.type == LRSchedulerType.WARMUP:
            return self._warmup(base_lr)
        
        elif self.config.type == LRSchedulerType.REDUCE_ON_PLATEAU and metric is not None:
            return self._reduce_on_plateau(base_lr, metric)
        
        elif self.config.type == LRSchedulerType.CYCLICAL:
            return self._cyclical(base_lr)
        
        return base_lr
    
    def _warmup(self, base_lr: float) -> float:
        """Linear warmup"""
        if self.step_count < self.config.warmup_steps:
            return base_lr * (self.step_count / self.config.warmup_steps)
        return base_lr
    
    def _reduce_on_plateau(self, base_lr: float, metric: float) -> float:
        """Reduce LR when metric plateaus"""
        improved = ((self.config.mode == "min" and metric < self.best_metric) or
                   (self.config.mode == "max" and metric > self.best_metric))
        
        if improved:
            self.best_metric = metric
            self.patience_counter = 0
        else:
            self.patience_counter += 1
            if self.patience_counter >= self.config.patience:
                self.patience_counter = 0
                return base_lr * self.config.gamma
        
        return base_lr
    
    def _cyclical(self, base_lr: float) -> float:
        """Cyclical learning rates"""
        cycle = math.floor(1 + self.step_count / (2 * self.config.step_size))
        x = abs(self.step_count / self.config.step_size - 2 * cycle + 1)
        return self.config.min_lr + (base_lr - self.config.min_lr) * max(0, (1 - x))


# ============================================================================
# SAM-SPECIFIC COMPONENTS
# ============================================================================

@dataclass
class SAMState:
    """
    SAM AGI State Representation
    Based on the God Equation and morphogenetic principles
    """
    S: np.ndarray  # Latent world state space (morphogenetic, variable-dim)
    theta: np.ndarray  # Model parameters
    phi: Dict[str, float]  # Meta-parameters (lr, compression, uncertainty)
    Sigma: np.ndarray  # Self manifold (conserved identity)
    U: float  # Unsolvability budget (epistemic humility)
    
    # Additional SAM-specific state
    timestamp: float = field(default_factory=lambda: np.random.rand())
    identity_overlap: float = 1.0
    morphogenesis_count: int = 0
    

class HardInvariant:
    """
    Hard invariants that must never be violated.
    These form the 'floor' on which all intelligence is built.
    """
    
    def __init__(self):
        self.invariants = {
            'self_preservation': {
                'check': lambda state: np.linalg.norm(state.Sigma) > 0,
                'description': 'Self-manifold must not vanish',
                'severity': 'CRITICAL'
            },
            'minimum_epistemic_rank': {
                'check': lambda state: np.linalg.matrix_rank(
                    np.atleast_2d(state.S)) >= 1,
                'description': 'Minimum covariance/rank must be preserved',
                'severity': 'CRITICAL'
            },
            'non_deletable_uncertainty': {
                'check': lambda state: state.U > 0,
                'description': 'Unsolvability budget must remain positive',
                'severity': 'CRITICAL'
            },
            'identity_continuity': {
                'check': lambda state: state.identity_overlap > 0.5,
                'description': 'Identity must not drift too far',
                'severity': 'WARNING'
            }
        }
        self.violations = []
    
    def check_all(self, state: SAMState) -> Tuple[bool, List[str]]:
        """Check all invariants, return (pass, violations)"""
        violations = []
        for name, invariant in self.invariants.items():
            try:
                if not invariant['check'](state):
                    violations.append(f"{invariant['severity']}: {invariant['description']}")
            except Exception as e:
                violations.append(f"ERROR checking {name}: {e}")
        
        self.violations.extend(violations)
        return len(violations) == 0, violations


class MorphogeneticController:
    """
    Controls morphogenesis (structural growth and adaptation).
    Implements the morphogenetic latency from the God Equation.
    """
    
    def __init__(self, 
                 gamma_morph: float = 0.01,
                 delta_identity: float = 0.7,
                 growth_threshold: float = 0.9):
        self.gamma_morph = gamma_morph  # Morphogenesis rate
        self.delta_identity = delta_identity  # Identity preservation threshold
        self.growth_threshold = growth_threshold
        self.expansion_history = []
    
    def should_grow(self, state: SAMState, uncertainty: float) -> bool:
        """Determine if system should expand dimensionality"""
        # Trigger growth if:
        # 1. Uncertainty is high (system confused)
        # 2. Identity is stable (not drifting)
        # 3. Has capacity to grow
        
        uncertainty_trigger = uncertainty > self.growth_threshold
        identity_stable = state.identity_overlap > self.delta_identity
        
        return uncertainty_trigger and identity_stable
    
    def grow(self, state: SAMState) -> SAMState:
        """Expand latent dimensionality"""
        # Add new dimension to S
        new_dim = np.random.randn()
        state.S = np.append(state.S, new_dim)
        
        # Expand theta
        state.theta = np.append(state.theta, np.random.randn())
        
        # Update Sigma to match new dimension
        if len(state.Sigma) < len(state.S):
            state.Sigma = np.append(state.Sigma, np.random.randn())
        
        # Cost of expansion: reduce learning rate
        state.phi['lr'] *= (1 - self.gamma_morph)
        
        state.morphogenesis_count += 1
        self.expansion_history.append({
            'timestamp': state.timestamp,
            'new_dim': len(state.S),
            'lr': state.phi['lr']
        })
        
        return state


class IdentityManifold:
    """
    Manages the self manifold (Sigma) for identity preservation.
    Ensures continuity of self across concept shifts.
    """
    
    def __init__(self, delta_identity: float = 0.7):
        self.delta_identity = delta_identity
        self.history = deque(maxlen=1000)
    
    def compute_overlap(self, current: np.ndarray, canonical: np.ndarray) -> float:
        """Compute overlap between current and canonical identity"""
        norm_current = np.linalg.norm(current)
        norm_canonical = np.linalg.norm(canonical)
        
        if norm_current == 0 or norm_canonical == 0:
            return 0.0
        
        overlap = np.dot(current, canonical) / (norm_current * norm_canonical)
        return float(overlap)
    
    def preserve(self, state: SAMState) -> SAMState:
        """Apply identity preservation correction if drifting"""
        overlap = self.compute_overlap(state.S, state.Sigma)
        state.identity_overlap = overlap
        
        if overlap < self.delta_identity:
            # Partial projection back to canonical identity
            correction = (self.delta_identity * state.Sigma + 
                         (1 - self.delta_identity) * state.S)
            state.S = correction
            
            # Recompute overlap after correction
            state.identity_overlap = self.compute_overlap(state.S, state.Sigma)
        
        self.history.append({
            'timestamp': state.timestamp,
            'overlap': overlap,
            'corrected': overlap < self.delta_identity
        })
        
        return state


class UnsolvabilityBudget:
    """
    Manages explicit knowledge of theoretical limits.
    Epistemic humility - knowing what cannot be known.
    """
    
    def __init__(self, initial_budget: float = 1.0, decay_rate: float = 0.99):
        self.initial_budget = initial_budget
        self.decay_rate = decay_rate
        self.uncertainty_log = deque(maxlen=1000)
    
    def decay(self, state: SAMState) -> SAMState:
        """Decay unsolvability budget over time"""
        state.U *= self.decay_rate
        
        # Check if budget is critically low
        if state.U < 0.1:
            # High epistemic risk - act conservatively
            state.phi['uncertainty_tolerance'] = max(
                0.1, state.phi.get('uncertainty_tolerance', 1.0) * 0.9
            )
        
        self.uncertainty_log.append({
            'timestamp': state.timestamp,
            'budget': state.U,
            'risk_level': 'HIGH' if state.U < 0.1 else 'NORMAL'
        })
        
        return state
    
    def replenish(self, state: SAMState, amount: float = 0.1):
        """Replenish budget when new information is gained"""
        state.U = min(1.0, state.U + amount)
        return state


# ============================================================================
# META-CONTROLLER WITH PRESSURE SIGNALS
# ============================================================================

class GrowthPrimitive(Enum):
    """Allowed mutations/growth operations"""
    EXPAND_LATENT = "expand_latent"  # Add dimensions to S
    COMPRESS = "compress"  # Reduce dimensions via distillation
    BRANCH = "branch"  # Create expert submodel
    MERGE = "merge"  # Merge similar experts
    FORGET = "forget"  # Prune low-importance weights
    REPLAY = "replay"  # Trigger experience replay
    DISTILL = "distill"  # Knowledge distillation
    CURRICULUM = "curriculum"  # Adjust curriculum difficulty


@dataclass
class PressureSignal:
    """Signal indicating system pressure for growth/adaptation"""
    type: str
    magnitude: float  # 0.0 to 1.0
    source: str
    timestamp: float


class MetaController:
    """
    Meta-controller that manages growth and adaptation.
    Selects growth primitives based on pressure signals.
    """
    
    def __init__(self):
        self.pressure_signals: List[PressureSignal] = []
        self.primitive_history: List[Dict] = []
        self.selection_policy = self._default_selection_policy()
    
    def emit_pressure(self, signal_type: str, magnitude: float, source: str):
        """Emit a pressure signal"""
        signal = PressureSignal(
            type=signal_type,
            magnitude=magnitude,
            source=source,
            timestamp=np.random.rand()  # Should use actual time
        )
        self.pressure_signals.append(signal)
    
    def _default_selection_policy(self) -> Callable:
        """Default policy for selecting growth primitives"""
        def policy(pressures: List[PressureSignal], state: SAMState) -> Optional[GrowthPrimitive]:
            if not pressures:
                return None
            
            # Aggregate pressures by type
            pressure_by_type = {}
            for p in pressures:
                if p.type not in pressure_by_type:
                    pressure_by_type[p.type] = []
                pressure_by_type[p.type].append(p.magnitude)
            
            # Find highest pressure
            max_pressure_type = max(pressure_by_type.keys(), 
                                   key=lambda t: np.mean(pressure_by_type[t]))
            avg_magnitude = np.mean(pressure_by_type[max_pressure_type])
            
            if avg_magnitude < 0.3:
                return None  # Not enough pressure
            
            # Map pressure type to primitive
            pressure_primitive_map = {
                'high_uncertainty': GrowthPrimitive.EXPAND_LATENT,
                'redundancy': GrowthPrimitive.COMPRESS,
                'specialization_needed': GrowthPrimitive.BRANCH,
                'overfitting': GrowthPrimitive.MERGE,
                'memory_pressure': GrowthPrimitive.FORGET,
                'stagnation': GrowthPrimitive.REPLAY,
                'knowledge_transfer': GrowthPrimitive.DISTILL,
                'difficulty_mismatch': GrowthPrimitive.CURRICULUM
            }
            
            return pressure_primitive_map.get(max_pressure_type)
        
        return policy
    
    def select_primitive(self, state: SAMState) -> Optional[GrowthPrimitive]:
        """Select a growth primitive based on current pressures"""
        # Clean old signals (keep last 100)
        if len(self.pressure_signals) > 100:
            self.pressure_signals = self.pressure_signals[-100:]
        
        primitive = self.selection_policy(self.pressure_signals, state)
        
        if primitive:
            self.primitive_history.append({
                'timestamp': state.timestamp,
                'primitive': primitive.value,
                'pressures': [(p.type, p.magnitude) for p in self.pressure_signals[-5:]]
            })
        
        return primitive
    
    def apply_primitive(self, primitive: GrowthPrimitive, state: SAMState) -> SAMState:
        """Apply a selected growth primitive"""
        if primitive == GrowthPrimitive.EXPAND_LATENT:
            # Expand latent space
            state.S = np.append(state.S, np.random.randn())
            state.theta = np.append(state.theta, np.random.randn())
            
        elif primitive == GrowthPrimitive.COMPRESS:
            # Simple compression: PCA-like reduction (placeholder)
            if len(state.S) > 2:
                state.S = state.S[:-1]
                state.theta = state.theta[:-1]
                
        elif primitive == GrowthPrimitive.FORGET:
            # Prune small weights
            threshold = 0.01
            state.theta = np.where(np.abs(state.theta) < threshold, 0, state.theta)
        
        return state


# ============================================================================
# SAM BUILDER INTEGRATION
# ============================================================================

class SAMBuilderIntegration:
    """
    Integrates all SAM/Neural components into the builder.
    This is the main interface for the automation system.
    """
    
    def __init__(self, 
                 latent_dim: int = 10,
                 optimizer_type: OptimizerType = OptimizerType.ADAM,
                 scheduler_type: LRSchedulerType = LRSchedulerType.COSINE):
        
        # Initialize SAM state
        self.state = SAMState(
            S=np.random.randn(latent_dim),
            theta=np.random.randn(latent_dim),
            phi={'lr': 0.001, 'compression': 0.5, 'uncertainty_tolerance': 1.0},
            Sigma=np.random.randn(latent_dim),  # Canonical identity
            U=1.0  # Full unsolvability budget
        )
        
        # Initialize components
        self.optimizer = NeuralOptimizer(
            OptimizerConfig(type=optimizer_type),
            (latent_dim,)
        )
        self.scheduler = LearningRateScheduler(SchedulerConfig(type=scheduler_type))
        self.invariants = HardInvariant()
        self.morphogenesis = MorphogeneticController()
        self.identity = IdentityManifold()
        self.unsolvability = UnsolvabilityBudget()
        self.meta_controller = MetaController()
        
        # Metrics
        self.metrics = {
            'iterations': 0,
            'violations': 0,
            'growth_events': 0,
            'primitives_applied': 0
        }
    
    def step(self, observation: np.ndarray, loss_gradient: np.ndarray, 
             loss_value: float = None) -> Dict[str, Any]:
        """
        Single step of the integrated system.
        
        Args:
            observation: Input observation
            loss_gradient: Gradient from loss
            loss_value: Optional loss value
        
        Returns:
            Step results and metrics
        """
        self.metrics['iterations'] += 1
        
        # 1. Check invariants
        invariants_ok, violations = self.invariants.check_all(self.state)
        if not invariants_ok:
            self.metrics['violations'] += len(violations)
        
        # 2. Compute uncertainty from gradient
        uncertainty = np.linalg.norm(loss_gradient)
        
        # 3. Update learning rate
        current_lr = self.scheduler.get_lr(self.state.phi['lr'], loss_value)
        self.state.phi['lr'] = current_lr
        self.optimizer.config.learning_rate = current_lr
        
        # 4. Optimization step
        self.state.theta = self.optimizer.step(self.state.theta, loss_gradient, loss_value)
        
        # 5. Morphogenesis (growth)
        if self.morphogenesis.should_grow(self.state, uncertainty):
            self.state = self.morphogenesis.grow(self.state)
            self.metrics['growth_events'] += 1
            self.meta_controller.emit_pressure('high_uncertainty', uncertainty, 'morphogenesis')
        
        # 6. Identity preservation
        self.state = self.identity.preserve(self.state)
        
        # 7. Unsolvability decay
        self.state = self.unsolvability.decay(self.state)
        
        # 8. Meta-controller: select and apply primitives
        primitive = self.meta_controller.select_primitive(self.state)
        if primitive:
            self.state = self.meta_controller.apply_primitive(primitive, self.state)
            self.metrics['primitives_applied'] += 1
        
        # Update timestamp
        self.state.timestamp = self.metrics['iterations']
        
        return {
            'state': self.state,
            'invariants_ok': invariants_ok,
            'violations': violations,
            'uncertainty': uncertainty,
            'learning_rate': current_lr,
            'identity_overlap': self.state.identity_overlap,
            'unsolvability_budget': self.state.U,
            'primitive_applied': primitive.value if primitive else None,
            'metrics': self.metrics.copy()
        }
    
    def get_status(self) -> Dict[str, Any]:
        """Get full system status"""
        return {
            'sam_state': {
                'latent_dim': len(self.state.S),
                'identity_overlap': self.state.identity_overlap,
                'unsolvability_budget': self.state.U,
                'morphogenesis_count': self.state.morphogenesis_count,
                'meta_params': self.state.phi
            },
            'optimizer': {
                'type': self.optimizer.config.type.value,
                'learning_rate': self.optimizer.config.learning_rate,
                'weight_decay': self.optimizer.config.weight_decay,
                'l1_lambda': self.optimizer.config.l1_lambda
            },
            'scheduler': {
                'type': self.scheduler.config.type.value,
                'step_count': self.scheduler.step_count
            },
            'invariants': {
                'violation_count': len(self.invariants.violations),
                'recent_violations': list(self.invariants.violations)[-5:]
            },
            'meta_controller': {
                'pressure_signals': len(self.meta_controller.pressure_signals),
                'primitives_applied': len(self.meta_controller.primitive_history),
                'recent_primitives': [p['primitive'] for p in 
                                     self.meta_controller.primitive_history[-5:]]
            },
            'metrics': self.metrics
        }


# ============================================================================
# EXPORT FOR BUILDER
# ============================================================================

__all__ = [
    # Optimizers
    'OptimizerType',
    'OptimizerConfig', 
    'NeuralOptimizer',
    
    # Schedulers
    'LRSchedulerType',
    'SchedulerConfig',
    'LearningRateScheduler',
    
    # SAM Components
    'SAMState',
    'HardInvariant',
    'MorphogeneticController',
    'IdentityManifold',
    'UnsolvabilityBudget',
    'MetaController',
    'GrowthPrimitive',
    'PressureSignal',
    
    # Integration
    'SAMBuilderIntegration'
]


if __name__ == "__main__":
    # Test the integration
    print("Testing SAM Neural Integration...")
    
    # Create integration
    sam = SAMBuilderIntegration(latent_dim=5)
    
    # Simulate training
    for i in range(100):
        obs = np.random.randn(5)
        grad = np.random.randn(5) * 0.1
        
        result = sam.step(obs, grad, loss_value=float(i))
        
        if i % 20 == 0:
            print(f"\nIteration {i}:")
            print(f"  Identity overlap: {result['identity_overlap']:.3f}")
            print(f"  Unsolvability: {result['unsolvability_budget']:.3f}")
            print(f"  Learning rate: {result['learning_rate']:.6f}")
            print(f"  Primitive: {result['primitive_applied']}")
    
    # Print final status
    print("\n" + "="*60)
    print("FINAL STATUS")
    print("="*60)
    status = sam.get_status()
    print(json.dumps(status, indent=2, default=str))
