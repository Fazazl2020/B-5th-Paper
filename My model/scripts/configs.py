"""
configs.py - Configuration File
MODIFIED: Added SNR auto-detection for test set
"""

import os
import re
from pathlib import Path

# ============================================================
# ==================== USER CONFIGURATION ====================
# ============================================================

# ==================== 1. DATASET PATHS ====================

DATASET_ROOT = '/gdata/fewahab/data/WSJO-full-wavdataset'

# FIXED SUBDIRECTORIES
TRAIN_CLEAN_SUBDIR = 'train/clean'
TRAIN_NOISY_SUBDIR = 'train/noisy'
VALID_CLEAN_SUBDIR = 'valid/clean'
VALID_NOISY_SUBDIR = 'valid/noisy'
TEST_CLEAN_SUBDIR = 'test/clean'
TEST_NOISY_SUBDIR = 'test/noisy'  # This won't be used directly for SNR-organized tests


# ==================== 2. CHECKPOINT PATHS ====================

CHECKPOINT_ROOT = '/ghome/fewahab/Sun-Models/Mod-3/T71a2/scripts/ckpt'


# ==================== 3. OUTPUT PATHS ====================

ESTIMATES_ROOT = '/gdata/fewahab/Sun-Models/Mod-3/T71a2/estimates_MA'


# ==================== 4. MODEL CONFIGURATION ====================

MODEL_CONFIG = {
    'in_norm': False,
    'sample_rate': 16000,
    'win_len': 0.020,
    'hop_len': 0.010
}


# ==================== 5. TRAINING CONFIGURATION ====================

TRAINING_CONFIG = {
    'gpu_ids': '0',
    'unit': 'utt',
    'batch_size': 10,
    'num_workers': 4,
    'segment_size': 4.0,
    'segment_shift': 1.0,
    'max_length_seconds': 6.0,
    'lr': 0.001,
    'plateau_factor': 0.5,
    'plateau_patience': 15,
    'plateau_threshold': 0.001,
    'plateau_min_lr': 1e-6,
    'max_n_epochs': 800,
    'early_stop_patience': 50,
    'clip_norm': 1.0,
    'loss_log': 'loss.txt',
    'time_log': '',
    'resume_model': '',
}


# ==================== 6. TESTING CONFIGURATION ====================

TESTING_CONFIG = {
    'batch_size': 1,
    'num_workers': 2,
    'write_ideal': False,
}


# ============================================================
# ============ END OF USER CONFIGURATION =====================
# ============================================================

# Derived paths
TRAIN_CLEAN_DIR = os.path.join(DATASET_ROOT, TRAIN_CLEAN_SUBDIR)
TRAIN_NOISY_DIR = os.path.join(DATASET_ROOT, TRAIN_NOISY_SUBDIR)
VALID_CLEAN_DIR = os.path.join(DATASET_ROOT, VALID_CLEAN_SUBDIR)
VALID_NOISY_DIR = os.path.join(DATASET_ROOT, VALID_NOISY_SUBDIR)
TEST_CLEAN_DIR = os.path.join(DATASET_ROOT, TEST_CLEAN_SUBDIR)
TEST_NOISY_DIR = os.path.join(DATASET_ROOT, TEST_NOISY_SUBDIR)

CHECKPOINT_DIR = CHECKPOINT_ROOT
LOGS_DIR = os.path.join(CHECKPOINT_DIR, 'logs')
MODELS_DIR = os.path.join(CHECKPOINT_DIR, 'models')
CACHE_DIR = os.path.join(CHECKPOINT_DIR, 'cache')
ESTIMATES_DIR = ESTIMATES_ROOT

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(ESTIMATES_DIR, exist_ok=True)

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


# ============================================================
# ============ NEW: SNR AUTO-DETECTION FOR TEST ==============
# ============================================================

def get_test_snr_dirs():
    """
    Auto-detect all SNR directories in test folder.
    
    This function scans the test directory and finds all subdirectories
    matching the pattern 'noisy_snr_XXX' where XXX is a signed integer.
    
    Returns:
        List[Tuple[int, str]]: List of (snr_value, directory_path) tuples,
                                sorted by SNR value in ascending order.
                                
    Example:
        [
            (-6, '/gdata/fewahab/data/WSJO-full-wavdataset/test/noisy_snr_-06'),
            (-3, '/gdata/fewahab/data/WSJO-full-wavdataset/test/noisy_snr_-03'),
            (0, '/gdata/fewahab/data/WSJO-full-wavdataset/test/noisy_snr_+00'),
            (3, '/gdata/fewahab/data/WSJO-full-wavdataset/test/noisy_snr_+03'),
            (6, '/gdata/fewahab/data/WSJO-full-wavdataset/test/noisy_snr_+06'),
        ]
    
    Raises:
        FileNotFoundError: If test directory doesn't exist
        ValueError: If no SNR directories are found
    """
    test_root = Path(DATASET_ROOT) / 'test'
    
    if not test_root.exists():
        raise FileNotFoundError(
            f"Test directory not found: {test_root}\n"
            f"Please ensure the dataset has been generated with SNR-organized structure."
        )
    
    snr_dirs = []
    
    # Find all directories matching pattern: noisy_snr_XXX
    for item in sorted(test_root.iterdir()):
        if item.is_dir() and item.name.startswith('noisy_snr_'):
            # Extract SNR value from directory name
            # Patterns: noisy_snr_-06, noisy_snr_+03, noisy_snr_+00, etc.
            match = re.match(r'noisy_snr_([+-]?\d+)', item.name)
            if match:
                snr_value = int(match.group(1))
                snr_dirs.append((snr_value, str(item)))
    
    if len(snr_dirs) == 0:
        raise ValueError(
            f"No SNR directories found in {test_root}\n"
            f"Expected directories like: noisy_snr_-06, noisy_snr_-03, etc.\n"
            f"Please regenerate the test dataset with SNR-organized structure."
        )
    
    # Sort by SNR value (ascending: -6, -3, 0, 3, 6)
    snr_dirs.sort(key=lambda x: x[0])
    
    return snr_dirs


def validate_test_snr_structure():
    """
    Validate that test set has correct SNR-organized structure.
    
    Checks:
    1. test/clean/ directory exists
    2. At least one noisy_snr_XXX directory exists
    3. All SNR directories have matching file counts
    
    Returns:
        Dict: Validation results with structure info
    
    Raises:
        FileNotFoundError: If required directories don't exist
        ValueError: If structure is invalid
    """
    test_root = Path(DATASET_ROOT) / 'test'
    clean_dir = test_root / 'clean'
    
    # Check clean directory
    if not clean_dir.exists():
        raise FileNotFoundError(
            f"Clean directory not found: {clean_dir}\n"
            f"Test set must have: test/clean/ directory"
        )
    
    clean_files = sorted([f for f in clean_dir.glob('*.wav')])
    if len(clean_files) == 0:
        raise ValueError(f"No WAV files found in {clean_dir}")
    
    # Get SNR directories
    try:
        snr_dirs = get_test_snr_dirs()
    except (FileNotFoundError, ValueError) as e:
        raise ValueError(f"Test set validation failed: {e}")
    
    # Validate each SNR directory
    validation_results = {
        'test_root': str(test_root),
        'clean_dir': str(clean_dir),
        'num_clean_files': len(clean_files),
        'snr_levels': [],
        'snr_dirs': {},
        'valid': True,
        'warnings': []
    }
    
    for snr_value, snr_dir in snr_dirs:
        snr_path = Path(snr_dir)
        noisy_files = sorted([f for f in snr_path.glob('*.wav')])
        
        validation_results['snr_levels'].append(snr_value)
        validation_results['snr_dirs'][snr_value] = {
            'path': snr_dir,
            'num_files': len(noisy_files)
        }
        
        # Check if file count matches clean files
        if len(noisy_files) != len(clean_files):
            warning = (
                f"SNR {snr_value:+d} dB: File count mismatch! "
                f"Clean: {len(clean_files)}, Noisy: {len(noisy_files)}"
            )
            validation_results['warnings'].append(warning)
            validation_results['valid'] = False
    
    return validation_results


# ============================================================
# ============ VALIDATION FUNCTIONS (UNCHANGED) ==============
# ============================================================

def validate_path(path, path_type="directory"):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path_type.capitalize()} not found: {path}\n"
            f"Please check the path exists and is accessible."
        )
    return path


def validate_data_dirs(mode='train'):
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
        print(f"\nChecking test data (SNR-organized structure)...")
        print(f"  Clean: {TEST_CLEAN_DIR}")
        
        try:
            # Use new validation function
            validation = validate_test_snr_structure()
            
            print(f"  ? Found {validation['num_clean_files']} clean files")
            print(f"  ? Found {len(validation['snr_levels'])} SNR levels: {validation['snr_levels']}")
            
            for snr in validation['snr_levels']:
                snr_info = validation['snr_dirs'][snr]
                print(f"     SNR {snr:+3d} dB: {snr_info['num_files']} files")
            
            if validation['warnings']:
                print(f"\n  ??  WARNINGS:")
                for warning in validation['warnings']:
                    print(f"     {warning}")
            
            if not validation['valid']:
                raise ValueError("Test set validation failed! Check warnings above.")
                
        except (FileNotFoundError, ValueError) as e:
            raise ValueError(f"Test data validation failed: {e}")
    
    print("\n" + "="*60)
    print("? ALL DATA DIRECTORIES VALIDATED SUCCESSFULLY!")
    print("="*60 + "\n")


def check_pytorch_version():
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
    pytorch_info = check_pytorch_version()
    
    print("\n" + "="*70)
    print("CONFIGURATION LOADED")
    print("="*70)
    print(f"\n?? PYTORCH INFO:")
    print(f"  Version: {pytorch_info['version']}")
    print(f"  CUDA available: {pytorch_info['cuda_available']}")
    print(f"  Persistent workers: {'? Supported' if pytorch_info['persistent_workers'] else '? Not supported'}")
    print(f"\n?? THREE MAIN DIRECTORIES:")
    print(f"  1. Dataset:     {DATASET_ROOT}")
    print(f"  2. Checkpoints: {CHECKPOINT_ROOT}")
    print(f"  3. Outputs:     {ESTIMATES_ROOT}")
    print(f"\n?? DETAILED PATHS:")
    print(f"  Training clean:  {TRAIN_CLEAN_DIR}")
    print(f"  Training noisy:  {TRAIN_NOISY_DIR}")
    print(f"  Valid clean:     {VALID_CLEAN_DIR}")
    print(f"  Valid noisy:     {VALID_NOISY_DIR}")
    print(f"  Test clean:      {TEST_CLEAN_DIR}")
    print(f"  Test structure:  SNR-organized (auto-detected)")
    print(f"  Model checkpoints: {MODELS_DIR}")
    print(f"  Logs:            {LOGS_DIR}")
    print(f"  Estimates:       {ESTIMATES_DIR}")
    print(f"\n??  TRAINING CONFIG:")
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
    print(f"\n?? MODEL CONFIG:")
    print(f"  Normalization: {'Disabled (preserves SNR)' if not MODEL_CONFIG['in_norm'] else 'Enabled'}")
    print(f"  Sample rate: {MODEL_CONFIG['sample_rate']} Hz")
    print(f"  Window: {MODEL_CONFIG['win_len']}s, Hop: {MODEL_CONFIG['hop_len']}s")
    if TRAINING_CONFIG['resume_model']:
        print(f"\n?? RESUME TRAINING:")
        print(f"  Checkpoint: {TRAINING_CONFIG['resume_model']}")
    
    # Show test SNR structure if available
    try:
        snr_dirs = get_test_snr_dirs()
        print(f"\n?? TEST SET (SNR-ORGANIZED):")
        print(f"  Detected {len(snr_dirs)} SNR levels:")
        for snr_val, snr_dir in snr_dirs:
            print(f"    SNR {snr_val:+3d} dB: {Path(snr_dir).name}")
    except (FileNotFoundError, ValueError):
        print(f"\n?? TEST SET:")
        print(f"  ??  SNR-organized structure not detected")
        print(f"  Run dataset generator to create test set")
    
    print("="*70 + "\n")


print_config()