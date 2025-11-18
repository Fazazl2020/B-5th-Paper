"""
configs.py - Clean Configuration File
All paths and settings at the TOP for easy access
"""

import os
from pathlib import Path


# ============================================================
# ==================== USER CONFIGURATION ====================
# ============================================================
# ALL SETTINGS HERE - CHANGE THESE FOR YOUR SETUP
# ============================================================

# ==================== 1. DATASET PATHS ====================
# Where your training/validation/test WAV files are located

DATASET_ROOT = '/gdata/fewahab/data/Voicebank+demand/My_train_valid_test'

# Subdirectories (usually don't need to change these)
TRAIN_CLEAN_SUBDIR = 'Train/train_clean'
TRAIN_NOISY_SUBDIR = 'Train/train_noisy'
VALID_CLEAN_SUBDIR = 'valid/valid_clean'
VALID_NOISY_SUBDIR = 'valid/valid_noisy'
TEST_CLEAN_SUBDIR = 'Test/test_clean'
TEST_NOISY_SUBDIR = 'Test/test_noisy'


# ==================== 2. CHECKPOINT PATHS ====================
# Where to save model checkpoints, logs, and training state

CHECKPOINT_ROOT = '/ghome/fewahab/Sun-Models/Mod-3/T71a1/scripts/ckpt'


# ==================== 3. OUTPUT PATHS ====================
# Where to save enhanced audio files (estimates)

ESTIMATES_ROOT = '/gdata/fewahab/Sun-Models/Mod-3/T71a1/estimates_MA'


# ==================== 4. MODEL CONFIGURATION ====================

MODEL_CONFIG = {
    'in_norm': False,         # Normalization: False = preserve SNR (RECOMMENDED)
    'sample_rate': 16000,     # Sample rate in Hz
    'win_len': 0.020,         # Window length in seconds
    'hop_len': 0.010          # Hop length in seconds
}


# ==================== 5. TRAINING CONFIGURATION ====================

TRAINING_CONFIG = {
    # ----- Hardware -----
    'gpu_ids': '0',           # GPU IDs: '0' = single GPU, '0,1' = multi-GPU, '-1' = CPU
    
    # ----- Data Loading -----
    'unit': 'utt',            # 'utt' = full utterances, 'seg' = segments
    'batch_size': 10,         # Batch size
    'num_workers': 4,         # Parallel workers (4-8 recommended)
    'segment_size': 4.0,      # Segment size in seconds (only for unit='seg')
    'segment_shift': 1.0,     # Segment shift in seconds (only for unit='seg')
    'max_length_seconds': 6.0,  # Maximum utterance length in seconds
    
    # ----- Learning Rate Schedule -----
    'lr': 0.001,              # Initial learning rate
    'plateau_factor': 0.5,    # LR reduction factor (multiply by this when plateau)
    'plateau_patience': 15,   # Epochs to wait before reducing LR
    'plateau_threshold': 0.001,  # Minimum improvement to count as progress
    'plateau_min_lr': 1e-6,   # Minimum learning rate (stop reducing after this)
    
    # ----- Training Duration -----
    'max_n_epochs': 800,      # Maximum number of epochs
    'early_stop_patience': 50,  # Stop if no improvement for this many epochs at min LR
    
    # ----- Optimization -----
    'clip_norm': 1.0,         # Gradient clipping norm (0 = no clipping)
    
    # ----- Logging -----
    'loss_log': 'loss.txt',   # Loss log filename (saved in checkpoint dir)
    'time_log': '',           # Time log filename (empty = print to stdout)
    
    # ----- Resume Training -----
    'resume_model': '',       # Path to checkpoint to resume from (empty = train from scratch)
                              # Example: '/ghome/.../ckpt/models/latest.pt'
}


# ==================== 6. TESTING CONFIGURATION ====================

TESTING_CONFIG = {
    'batch_size': 1,          # Batch size for testing (usually 1)
    'num_workers': 2,         # Parallel workers for testing
    'write_ideal': False,     # Save ideal reconstruction (for debugging)
}


# ============================================================
# ============ END OF USER CONFIGURATION =====================
# ============================================================
# Don't modify below unless you know what you're doing
# ============================================================


# ==================== DERIVED PATHS (AUTO-GENERATED) ====================

# Dataset paths
TRAIN_CLEAN_DIR = os.path.join(DATASET_ROOT, TRAIN_CLEAN_SUBDIR)
TRAIN_NOISY_DIR = os.path.join(DATASET_ROOT, TRAIN_NOISY_SUBDIR)
VALID_CLEAN_DIR = os.path.join(DATASET_ROOT, VALID_CLEAN_SUBDIR)
VALID_NOISY_DIR = os.path.join(DATASET_ROOT, VALID_NOISY_SUBDIR)
TEST_CLEAN_DIR = os.path.join(DATASET_ROOT, TEST_CLEAN_SUBDIR)
TEST_NOISY_DIR = os.path.join(DATASET_ROOT, TEST_NOISY_SUBDIR)

# Checkpoint paths
CHECKPOINT_DIR = CHECKPOINT_ROOT
LOGS_DIR = os.path.join(CHECKPOINT_DIR, 'logs')
MODELS_DIR = os.path.join(CHECKPOINT_DIR, 'models')
CACHE_DIR = os.path.join(CHECKPOINT_DIR, 'cache')

# Output paths
ESTIMATES_DIR = ESTIMATES_ROOT

# Create directories
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(ESTIMATES_DIR, exist_ok=True)


# ==================== LEGACY CONFIG FORMAT (FOR COMPATIBILITY) ====================
# These are used by model.py - don't modify

exp_conf = MODEL_CONFIG

train_conf = {
    'gpu_ids': TRAINING_CONFIG['gpu_ids'],
    'ckpt_dir': CHECKPOINT_DIR,
    'est_path': ESTIMATES_DIR,
    'unit': TRAINING_CONFIG['unit'],
    'batch_size': TRAINING_CONFIG['batch_size'],
    'num_workers': TRAINING_CONFIG['num_workers'],
    'segment_size': TRAINING_CONFIG['segment_size'],
    'segment_shift': TRAINING_CONFIG['segment_shift'],
    'max_length_seconds': TRAINING_CONFIG['max_length_seconds'],
    'lr': TRAINING_CONFIG['lr'],
    'plateau_factor': TRAINING_CONFIG['plateau_factor'],
    'plateau_patience': TRAINING_CONFIG['plateau_patience'],
    'plateau_threshold': TRAINING_CONFIG['plateau_threshold'],
    'plateau_min_lr': TRAINING_CONFIG['plateau_min_lr'],
    'max_n_epochs': TRAINING_CONFIG['max_n_epochs'],
    'early_stop_patience': TRAINING_CONFIG['early_stop_patience'],
    'clip_norm': TRAINING_CONFIG['clip_norm'],
    'loss_log': TRAINING_CONFIG['loss_log'],
    'time_log': TRAINING_CONFIG['time_log'],
    'resume_model': TRAINING_CONFIG['resume_model'],
}

test_conf = {
    'model_file': os.path.join(MODELS_DIR, 'best.pt'),
    'batch_size': TESTING_CONFIG['batch_size'],
    'num_workers': TESTING_CONFIG['num_workers'],
    'write_ideal': TESTING_CONFIG['write_ideal'],
}


# ==================== VALIDATION FUNCTIONS ====================

def validate_path(path, path_type="directory"):
    """Validate that a path exists"""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path_type.capitalize()} not found: {path}\n"
            f"Please check the path exists and is accessible."
        )
    return path


def validate_data_dirs(mode='train'):
    """
    Validate that required directories exist and contain WAV files
    
    Args:
        mode: 'train', 'valid', 'test', or 'all'
    """
    print("\n" + "="*60)
    print("VALIDATING DATA DIRECTORIES")
    print("="*60)
    
    if mode in ['train', 'all']:
        print(f"\nChecking training data...")
        print(f"  Clean: {TRAIN_CLEAN_DIR}")
        print(f"  Noisy: {TRAIN_NOISY_DIR}")
        
        validate_path(TRAIN_CLEAN_DIR, "Training clean directory")
        validate_path(TRAIN_NOISY_DIR, "Training noisy directory")
        
        train_clean_files = [f for f in os.listdir(TRAIN_CLEAN_DIR) if f.endswith('.wav')]
        train_noisy_files = [f for f in os.listdir(TRAIN_NOISY_DIR) if f.endswith('.wav')]
        
        if len(train_clean_files) == 0:
            raise ValueError(f"No WAV files found in {TRAIN_CLEAN_DIR}")
        if len(train_noisy_files) == 0:
            raise ValueError(f"No WAV files found in {TRAIN_NOISY_DIR}")
        
        print(f"  ? Found {len(train_clean_files)} clean files")
        print(f"  ? Found {len(train_noisy_files)} noisy files")
    
    if mode in ['valid', 'train', 'all']:
        print(f"\nChecking validation data...")
        print(f"  Clean: {VALID_CLEAN_DIR}")
        print(f"  Noisy: {VALID_NOISY_DIR}")
        
        validate_path(VALID_CLEAN_DIR, "Validation clean directory")
        validate_path(VALID_NOISY_DIR, "Validation noisy directory")
        
        valid_clean_files = [f for f in os.listdir(VALID_CLEAN_DIR) if f.endswith('.wav')]
        valid_noisy_files = [f for f in os.listdir(VALID_NOISY_DIR) if f.endswith('.wav')]
        
        if len(valid_clean_files) == 0:
            raise ValueError(f"No WAV files found in {VALID_CLEAN_DIR}")
        if len(valid_noisy_files) == 0:
            raise ValueError(f"No WAV files found in {VALID_NOISY_DIR}")
        
        print(f"  ? Found {len(valid_clean_files)} clean files")
        print(f"  ? Found {len(valid_noisy_files)} noisy files")
    
    if mode in ['test', 'all']:
        print(f"\nChecking test data...")
        print(f"  Clean: {TEST_CLEAN_DIR}")
        print(f"  Noisy: {TEST_NOISY_DIR}")
        
        validate_path(TEST_CLEAN_DIR, "Test clean directory")
        validate_path(TEST_NOISY_DIR, "Test noisy directory")
        
        test_clean_files = [f for f in os.listdir(TEST_CLEAN_DIR) if f.endswith('.wav')]
        test_noisy_files = [f for f in os.listdir(TEST_NOISY_DIR) if f.endswith('.wav')]
        
        if len(test_clean_files) == 0:
            raise ValueError(f"No WAV files found in {TEST_CLEAN_DIR}")
        if len(test_noisy_files) == 0:
            raise ValueError(f"No WAV files found in {TEST_NOISY_DIR}")
        
        print(f"  ? Found {len(test_clean_files)} clean files")
        print(f"  ? Found {len(test_noisy_files)} noisy files")
    
    print("\n" + "="*60)
    print("? ALL DATA DIRECTORIES VALIDATED SUCCESSFULLY!")
    print("="*60 + "\n")


def check_pytorch_version():
    """Check PyTorch version for compatibility"""
    try:
        import torch
        version = torch.__version__.split('+')[0]
        major, minor, patch = map(int, version.split('.'))
        
        persistent_workers_supported = (major > 1) or (major == 1 and minor >= 7)
        
        return {
            'version': torch.__version__,
            'persistent_workers': persistent_workers_supported,
            'cuda_available': torch.cuda.is_available()
        }
    except ImportError:
        raise ImportError("PyTorch is not installed!")


def print_config():
    """Print configuration summary"""
    
    pytorch_info = check_pytorch_version()
    
    print("\n" + "="*70)
    print("CONFIGURATION LOADED")
    print("="*70)
    
    print("\n?? PYTORCH INFO:")
    print(f"  Version: {pytorch_info['version']}")
    print(f"  CUDA available: {pytorch_info['cuda_available']}")
    print(f"  Persistent workers: {'? Supported' if pytorch_info['persistent_workers'] else '? Not supported'}")
    
    print("\n?? THREE MAIN DIRECTORIES:")
    print(f"  1. Dataset:     {DATASET_ROOT}")
    print(f"  2. Checkpoints: {CHECKPOINT_ROOT}")
    print(f"  3. Outputs:     {ESTIMATES_ROOT}")
    
    print("\n?? DETAILED PATHS:")
    print(f"  Training clean:  {TRAIN_CLEAN_DIR}")
    print(f"  Training noisy:  {TRAIN_NOISY_DIR}")
    print(f"  Valid clean:     {VALID_CLEAN_DIR}")
    print(f"  Valid noisy:     {VALID_NOISY_DIR}")
    print(f"  Test clean:      {TEST_CLEAN_DIR}")
    print(f"  Test noisy:      {TEST_NOISY_DIR}")
    print(f"  Model checkpoints: {MODELS_DIR}")
    print(f"  Logs:            {LOGS_DIR}")
    print(f"  Estimates:       {ESTIMATES_DIR}")
    
    print("\n??  TRAINING CONFIG:")
    print(f"  GPU: {TRAINING_CONFIG['gpu_ids']}")
    print(f"  Unit: {TRAINING_CONFIG['unit']}")
    print(f"  Batch size: {TRAINING_CONFIG['batch_size']}")
    print(f"  Num workers: {TRAINING_CONFIG['num_workers']}")
    if TRAINING_CONFIG['unit'] == 'seg':
        print(f"  Segment size: {TRAINING_CONFIG['segment_size']}s")
        print(f"  Segment shift: {TRAINING_CONFIG['segment_shift']}s")
    print(f"  Max length: {TRAINING_CONFIG['max_length_seconds']}s")
    print(f"  Initial LR: {TRAINING_CONFIG['lr']}")
    print(f"  Max epochs: {TRAINING_CONFIG['max_n_epochs']}")
    print(f"  Early stop patience: {TRAINING_CONFIG['early_stop_patience']}")
    
    print("\n?? MODEL CONFIG:")
    print(f"  Normalization: {'Disabled (preserves SNR)' if not MODEL_CONFIG['in_norm'] else 'Enabled'}")
    print(f"  Sample rate: {MODEL_CONFIG['sample_rate']} Hz")
    print(f"  Window: {MODEL_CONFIG['win_len']}s, Hop: {MODEL_CONFIG['hop_len']}s")
    
    if TRAINING_CONFIG['resume_model']:
        print("\n?? RESUME TRAINING:")
        print(f"  Checkpoint: {TRAINING_CONFIG['resume_model']}")
    
    print("="*70 + "\n")


# Print config on import
print_config()
