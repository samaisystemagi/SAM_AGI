#!/usr/bin/env python3
"""
SAM_C_BRIDGE - Python Interface to C Core Components

This module provides Python bindings to the C neural network
and SAM AGI components in ORGANIZED/ directory.

Architecture:
- C Core: Heavy computation (neural nets, optimizers, SAM AGI)
- Python Bridge: Interface layer (this file)
- Automation Builder: Orchestration and coordination
"""

import ctypes
import os
import sys
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import json
import subprocess

# ============================================================================
# CONFIGURATION
# ============================================================================

ORGANIZED_DIR = Path(__file__).parent.parent.parent.parent / "ORGANIZED"
UTILS_DIR = ORGANIZED_DIR / "UTILS"
NN_DIR = UTILS_DIR / "utils" / "NN"
SAM_DIR = UTILS_DIR / "SAM" / "SAM"
MUZE_DIR = NN_DIR / "MUZE"

# ============================================================================
# C STRUCTURE DEFINITIONS
# ============================================================================

class COptimizerConfig(ctypes.Structure):
    """C-compatible optimizer configuration"""
    _fields_ = [
        ("type", ctypes.c_int),  # 0=Adam, 1=SGD, 2=RMSprop, etc.
        ("learning_rate", ctypes.c_double),
        ("beta1", ctypes.c_double),
        ("beta2", ctypes.c_double),
        ("epsilon", ctypes.c_double),
        ("momentum", ctypes.c_double),
        ("weight_decay", ctypes.c_double),  # L2
        ("l1_lambda", ctypes.c_double),     # L1
    ]

class CSAMState(ctypes.Structure):
    """C-compatible SAM AGI state"""
    _fields_ = [
        ("latent_dim", ctypes.c_size_t),
        ("S", ctypes.POINTER(ctypes.c_double)),      # Latent state
        ("theta", ctypes.POINTER(ctypes.c_double)),  # Parameters
        ("Sigma", ctypes.POINTER(ctypes.c_double)),  # Identity manifold
        ("U", ctypes.c_double),                      # Unsolvability budget
        ("lr", ctypes.c_double),
        ("timestamp", ctypes.c_double),
    ]

class CProcessingResult(ctypes.Structure):
    """C-compatible processing result"""
    _fields_ = [
        ("completeness", ctypes.c_double),
        ("quality", ctypes.c_double),
        ("entities_found", ctypes.c_size_t),
        ("key_points_found", ctypes.c_size_t),
        ("patterns_found", ctypes.c_size_t),
        ("iteration_count", ctypes.c_size_t),
    ]

# ============================================================================
# NEURAL NETWORK WRAPPER
# ============================================================================

class NeuralNetworkCore:
    """
    Wrapper for C neural network implementations.
    Provides Python interface to C optimizers and networks.
    """
    
    def __init__(self, network_type: str = "MLP"):
        self.network_type = network_type
        self.lib = None
        self._load_library()
        
    def _load_library(self):
        """Load compiled C library"""
        # Look for compiled .so/.dylib files
        lib_paths = [
            NN_DIR / f"{self.network_type}" / f"{self.network_type}.so",
            NN_DIR / f"{self.network_type}" / f"{self.network_type}.dylib",
            UTILS_DIR / f"lib{self.network_type.lower()}.so",
        ]
        
        for lib_path in lib_paths:
            if lib_path.exists():
                try:
                    self.lib = ctypes.CDLL(str(lib_path))
                    print(f"✅ Loaded C library: {lib_path}")
                    return
                except OSError as e:
                    print(f"⚠️  Could not load {lib_path}: {e}")
                    continue
        
        print(f"⚠️  No compiled library found for {self.network_type}")
        print(f"   Searched in: {lib_paths}")
        print(f"   Will use Python fallback")
    
    def compile_if_needed(self):
        """Compile C code if library doesn't exist"""
        source_dir = NN_DIR / self.network_type
        if not source_dir.exists():
            print(f"❌ Source directory not found: {source_dir}")
            return False
        
        # Check for Makefile
        makefile = source_dir / "Makefile"
        if makefile.exists():
            print(f"🔨 Compiling {self.network_type} from Makefile...")
            try:
                result = subprocess.run(
                    ["make"],
                    cwd=source_dir,
                    capture_output=True,
                    text=True,
                    timeout=60
                )
                if result.returncode == 0:
                    print(f"✅ Compilation successful")
                    self._load_library()
                    return True
                else:
                    print(f"❌ Compilation failed: {result.stderr}")
                    return False
            except Exception as e:
                print(f"❌ Compilation error: {e}")
                return False
        else:
            # Try basic gcc compilation
            print(f"🔨 Attempting basic compilation...")
            return self._basic_compile(source_dir)
    
    def _basic_compile(self, source_dir: Path) -> bool:
        """Basic gcc compilation without Makefile"""
        c_file = source_dir / f"{self.network_type}.c"
        output = source_dir / f"{self.network_type}.so"
        
        if not c_file.exists():
            print(f"❌ Source file not found: {c_file}")
            return False
        
        try:
            result = subprocess.run(
                ["gcc", "-shared", "-fPIC", "-O2", "-o", str(output), str(c_file), "-lm"],
                capture_output=True,
                text=True,
                timeout=30
            )
            if result.returncode == 0:
                print(f"✅ Basic compilation successful")
                self._load_library()
                return True
            else:
                print(f"❌ Compilation failed: {result.stderr}")
                return False
        except Exception as e:
            print(f"❌ Compilation error: {e}")
            return False


# ============================================================================
# SAM AGI WRAPPER
# ============================================================================

class SAMCore:
    """
    Wrapper for C SAM AGI implementation.
    Provides Python interface to SAM state and operations.
    """
    
    def __init__(self, latent_dim: int = 64):
        self.latent_dim = latent_dim
        self.lib = None
        self.state = None
        self._load_sam_library()
        
    def _load_sam_library(self):
        """Load SAM C library"""
        lib_paths = [
            SAM_DIR / "libsam.so",
            SAM_DIR / "libsam.dylib",
            UTILS_DIR / "libsam.so",
        ]
        
        for lib_path in lib_paths:
            if lib_path.exists():
                try:
                    self.lib = ctypes.CDLL(str(lib_path))
                    self._setup_functions()
                    print(f"✅ Loaded SAM library: {lib_path}")
                    return
                except OSError as e:
                    print(f"⚠️  Could not load {lib_path}: {e}")
                    continue
        
        print(f"⚠️  No compiled SAM library found")
        print(f"   Creating pure Python SAM state...")
        self._init_python_fallback()
    
    def _setup_functions(self):
        """Setup C function signatures"""
        if not self.lib:
            return
        
        # Define function signatures
        try:
            # SAM_create
            self.lib.SAM_create.argtypes = [ctypes.c_size_t]
            self.lib.SAM_create.restype = ctypes.POINTER(CSAMState)
            
            # SAM_step
            self.lib.SAM_step.argtypes = [
                ctypes.POINTER(CSAMState),
                ctypes.POINTER(ctypes.c_double),
                ctypes.POINTER(ctypes.c_double)
            ]
            self.lib.SAM_step.restype = ctypes.c_int
            
            # SAM_destroy
            self.lib.SAM_destroy.argtypes = [ctypes.POINTER(CSAMState)]
            self.lib.SAM_destroy.restype = None
            
        except AttributeError as e:
            print(f"⚠️  Some C functions not found: {e}")
    
    def _init_python_fallback(self):
        """Initialize pure Python SAM state when C lib not available"""
        self.state = {
            'S': np.random.randn(self.latent_dim),
            'theta': np.random.randn(self.latent_dim),
            'Sigma': np.random.randn(self.latent_dim),
            'U': 1.0,
            'lr': 0.001,
            'timestamp': 0.0,
            'identity_overlap': 1.0,
        }
    
    def step(self, observation: np.ndarray, gradient: np.ndarray) -> Dict[str, Any]:
        """Execute one SAM step"""
        if self.lib and hasattr(self.lib, 'SAM_step'):
            # Use C implementation
            return self._c_step(observation, gradient)
        else:
            # Use Python fallback
            return self._python_step(observation, gradient)
    
    def _c_step(self, obs: np.ndarray, grad: np.ndarray) -> Dict[str, Any]:
        """C implementation step"""
        # Convert numpy to C arrays
        obs_ctype = obs.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        grad_ctype = grad.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        
        # Call C function
        result = self.lib.SAM_step(self.state, obs_ctype, grad_ctype)
        
        return {
            'success': result == 0,
            'state': self._extract_state(),
        }
    
    def _python_step(self, obs: np.ndarray, grad: np.ndarray) -> Dict[str, Any]:
        """Python fallback step"""
        # Simple gradient descent with identity preservation
        self.state['theta'] -= self.state['lr'] * grad
        
        # Update latent state
        self.state['S'] = 0.9 * self.state['S'] + 0.1 * obs
        
        # Identity preservation
        overlap = np.dot(self.state['S'], self.state['Sigma'])
        overlap /= (np.linalg.norm(self.state['S']) * np.linalg.norm(self.state['Sigma']) + 1e-8)
        self.state['identity_overlap'] = overlap
        
        if overlap < 0.7:
            # Pull back to identity
            self.state['S'] = 0.7 * self.state['Sigma'] + 0.3 * self.state['S']
        
        # Decay unsolvability
        self.state['U'] *= 0.99
        
        self.state['timestamp'] += 1
        
        return {
            'success': True,
            'state': self.state.copy(),
            'identity_overlap': overlap,
            'unsolvability': self.state['U'],
        }
    
    def _extract_state(self) -> Dict[str, Any]:
        """Extract state from C structure"""
        if not self.state:
            return {}
        
        # Convert C pointers to numpy arrays
        S = np.ctypeslib.as_array(self.state.contents.S, 
                                   shape=(self.latent_dim,))
        theta = np.ctypeslib.as_array(self.state.contents.theta,
                                       shape=(self.latent_dim,))
        
        return {
            'S': S.copy(),
            'theta': theta.copy(),
            'U': self.state.contents.U,
            'lr': self.state.contents.lr,
            'timestamp': self.state.contents.timestamp,
        }


# ============================================================================
# MUZE (MuZero) WRAPPER
# ============================================================================

class MUZECore:
    """
    Wrapper for C MUZE (MuZero) implementation.
    Provides Python interface to MCTS and world models.
    """
    
    def __init__(self):
        self.lib = None
        self._load_muze_library()
    
    def _load_muze_library(self):
        """Load MUZE C library"""
        lib_paths = [
            MUZE_DIR / "libmuze.so",
            MUZE_DIR / "libmuze.dylib",
            NN_DIR / "libmuze.so",
        ]
        
        for lib_path in lib_paths:
            if lib_path.exists():
                try:
                    self.lib = ctypes.CDLL(str(lib_path))
                    print(f"✅ Loaded MUZE library: {lib_path}")
                    return
                except OSError as e:
                    print(f"⚠️  Could not load {lib_path}: {e}")
                    continue
        
        print(f"⚠️  No compiled MUZE library found")
    
    def compile(self) -> bool:
        """Compile MUZE from source"""
        makefile = MUZE_DIR / "Makefile"
        if not makefile.exists():
            print(f"❌ No Makefile found in {MUZE_DIR}")
            return False
        
        print(f"🔨 Compiling MUZE...")
        try:
            result = subprocess.run(
                ["make", "clean", "&&", "make"],
                cwd=MUZE_DIR,
                capture_output=True,
                text=True,
                timeout=120,
                shell=True
            )
            if result.returncode == 0:
                print(f"✅ MUZE compilation successful")
                self._load_muze_library()
                return True
            else:
                print(f"❌ MUZE compilation failed: {result.stderr}")
                return False
        except Exception as e:
            print(f"❌ Compilation error: {e}")
            return False


# ============================================================================
# OPTIMIZER WRAPPER
# ============================================================================

class COptimizerBridge:
    """
    Bridge to C optimizer implementations.
    Falls back to Python if C not available.
    """
    
    OPTIMIZER_TYPES = {
        'adam': 0,
        'sgd': 1,
        'rmsprop': 2,
        'natural_gd': 3,
        'bfgs': 4,
        'newton': 5,
    }
    
    def __init__(self, 
                 optimizer_type: str = 'adam',
                 learning_rate: float = 0.001,
                 weight_decay: float = 0.0,
                 l1_lambda: float = 0.0):
        
        self.optimizer_type = optimizer_type.lower()
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay  # L2
        self.l1_lambda = l1_lambda        # L1
        
        self.lib = None
        self.state = None
        self._load_optimizer()
    
    def _load_optimizer(self):
        """Load or create optimizer"""
        # Try to load C library
        lib_path = NN_DIR / "NN" / "libnn.so"
        if lib_path.exists():
            try:
                self.lib = ctypes.CDLL(str(lib_path))
                self._setup_c_optimizer()
                return
            except OSError:
                pass
        
        # Use Python implementation
        self._init_python_optimizer()
    
    def _setup_c_optimizer(self):
        """Setup C optimizer"""
        # Define config
        config = COptimizerConfig()
        config.type = self.OPTIMIZER_TYPES.get(self.optimizer_type, 0)
        config.learning_rate = self.learning_rate
        config.beta1 = 0.9
        config.beta2 = 0.999
        config.epsilon = 1e-8
        config.momentum = 0.9
        config.weight_decay = self.weight_decay
        config.l1_lambda = self.l1_lambda
        
        self.config = config
    
    def _init_python_optimizer(self):
        """Initialize Python optimizer state"""
        if self.optimizer_type == 'adam':
            self.state = {
                'm': None,  # First moment
                'v': None,  # Second moment
                't': 0,
            }
        elif self.optimizer_type == 'rmsprop':
            self.state = {
                'square_avg': None,
            }
        elif self.optimizer_type == 'sgd' and self.learning_rate > 0:
            self.state = {
                'velocity': None,
            }
    
    def step(self, params: np.ndarray, grad: np.ndarray) -> np.ndarray:
        """Single optimization step"""
        if self.lib:
            return self._c_step(params, grad)
        else:
            return self._python_step(params, grad)
    
    def _python_step(self, params: np.ndarray, grad: np.ndarray) -> np.ndarray:
        """Python optimizer implementation"""
        # Add regularization
        if self.weight_decay > 0:
            grad = grad + self.weight_decay * params  # L2
        
        if self.l1_lambda > 0:
            grad = grad + self.l1_lambda * np.sign(params)  # L1
        
        # Optimizer-specific update
        if self.optimizer_type == 'adam':
            return self._adam_step(params, grad)
        elif self.optimizer_type == 'sgd':
            return params - self.learning_rate * grad
        elif self.optimizer_type == 'rmsprop':
            return self._rmsprop_step(params, grad)
        else:
            return params - self.learning_rate * grad
    
    def _adam_step(self, params: np.ndarray, grad: np.ndarray) -> np.ndarray:
        """Adam optimizer"""
        if self.state['m'] is None:
            self.state['m'] = np.zeros_like(params)
            self.state['v'] = np.zeros_like(params)
        
        self.state['t'] += 1
        t = self.state['t']
        
        # Update moments
        self.state['m'] = 0.9 * self.state['m'] + 0.1 * grad
        self.state['v'] = 0.999 * self.state['v'] + 0.1 * (grad ** 2)
        
        # Bias correction
        m_hat = self.state['m'] / (1 - 0.9 ** t)
        v_hat = self.state['v'] / (1 - 0.999 ** t)
        
        # Update
        return params - self.learning_rate * m_hat / (np.sqrt(v_hat) + 1e-8)
    
    def _rmsprop_step(self, params: np.ndarray, grad: np.ndarray) -> np.ndarray:
        """RMSprop optimizer"""
        if self.state['square_avg'] is None:
            self.state['square_avg'] = np.zeros_like(params)
        
        alpha = 0.99
        self.state['square_avg'] = (alpha * self.state['square_avg'] + 
                                   (1 - alpha) * (grad ** 2))
        
        return params - (self.learning_rate * grad / 
                        (np.sqrt(self.state['square_avg']) + 1e-8))
    
    def _c_step(self, params: np.ndarray, grad: np.ndarray) -> np.ndarray:
        """C optimizer step (placeholder)"""
        # Would call C library
        return self._python_step(params, grad)


# ============================================================================
# BUILDER INTEGRATION
# ============================================================================

class SAMBuilderBridge:
    """
    Main bridge class that integrates C components with the automation builder.
    This is the interface that core.py will use.
    """
    
    def __init__(self):
        self.neural_core = None
        self.sam_core = None
        self.muze_core = None
        self.optimizer = None
        self.initialized = False
        
    def initialize(self, 
                   latent_dim: int = 64,
                   optimizer_type: str = 'adam',
                   network_type: str = 'MLP'):
        """Initialize all C components"""
        print("\n" + "="*70)
        print("  INITIALIZING SAM C-CORE BRIDGE")
        print("="*70)
        
        # Initialize neural network
        print(f"\n🔧 Neural Network: {network_type}")
        self.neural_core = NeuralNetworkCore(network_type)
        if not self.neural_core.lib:
            self.neural_core.compile_if_needed()
        
        # Initialize SAM
        print(f"\n🧠 SAM AGI: latent_dim={latent_dim}")
        self.sam_core = SAMCore(latent_dim)
        
        # Initialize MUZE
        print(f"\n🎯 MUZE (MuZero)")
        self.muze_core = MUZECore()
        
        # Initialize optimizer
        print(f"\n⚡ Optimizer: {optimizer_type}")
        self.optimizer = COptimizerBridge(optimizer_type)
        
        self.initialized = True
        
        print("\n✅ SAM C-Core Bridge initialized")
        print("="*70 + "\n")
    
    def process_chunk_with_sam(self, 
                               chunk_data: np.ndarray,
                               iteration: int) -> Dict[str, Any]:
        """
        Process a data chunk using SAM AGI principles.
        
        This is the main integration point with the automation builder.
        """
        if not self.initialized:
            self.initialize()
        
        # Create synthetic observation from chunk
        obs = self._chunk_to_observation(chunk_data)
        
        # Compute gradient (simplified)
        grad = np.random.randn(len(obs)) * 0.01
        
        # SAM step
        sam_result = self.sam_core.step(obs, grad)
        
        # Optimize
        if self.optimizer:
            params = sam_result.get('state', {}).get('theta', obs)
            optimized = self.optimizer.step(params, grad)
            sam_result['optimized_theta'] = optimized
        
        return {
            'sam_state': sam_result,
            'identity_overlap': sam_result.get('identity_overlap', 0.0),
            'unsolvability': sam_result.get('unsolvability', 0.0),
            'iteration': iteration,
        }
    
    def _chunk_to_observation(self, chunk: np.ndarray) -> np.ndarray:
        """Convert chunk data to SAM observation"""
        # Flatten and normalize
        flat = chunk.flatten() if hasattr(chunk, 'flatten') else np.array(chunk)
        
        # Ensure correct dimensionality
        target_dim = self.sam_core.latent_dim if self.sam_core else 64
        
        if len(flat) < target_dim:
            # Pad
            flat = np.pad(flat, (0, target_dim - len(flat)))
        elif len(flat) > target_dim:
            # Truncate
            flat = flat[:target_dim]
        
        # Normalize
        norm = np.linalg.norm(flat)
        if norm > 0:
            flat = flat / norm
        
        return flat
    
    def get_status(self) -> Dict[str, Any]:
        """Get full system status"""
        return {
            'initialized': self.initialized,
            'neural_core': self.neural_core is not None,
            'sam_core': self.sam_core is not None if self.sam_core else False,
            'muze_core': self.muze_core is not None if self.muze_core else False,
            'optimizer': self.optimizer.optimizer_type if self.optimizer else None,
            'organized_dir': str(ORGANIZED_DIR),
            'sam_dir': str(SAM_DIR),
            'muze_dir': str(MUZE_DIR),
        }


# ============================================================================
# MAIN INTERFACE
# ============================================================================

def create_sam_bridge() -> SAMBuilderBridge:
    """Factory function to create SAM bridge"""
    return SAMBuilderBridge()


# Global instance for convenience
_global_bridge = None

def get_bridge() -> SAMBuilderBridge:
    """Get or create global bridge instance"""
    global _global_bridge
    if _global_bridge is None:
        _global_bridge = create_sam_bridge()
    return _global_bridge


if __name__ == "__main__":
    # Test the bridge
    print("Testing SAM C-Core Bridge...")
    
    bridge = create_sam_bridge()
    bridge.initialize(latent_dim=32, optimizer_type='adam')
    
    # Test processing
    test_chunk = np.random.randn(100)
    result = bridge.process_chunk_with_sam(test_chunk, iteration=1)
    
    print("\n✅ Test successful!")
    print(f"Result keys: {result.keys()}")
    print(f"\nStatus: {bridge.get_status()}")
