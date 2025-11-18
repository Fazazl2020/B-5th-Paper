"""
configs.py - Configuration for CMGAN-Style Audio Processing
============================================================

MODIFICATIONS FOR CMGAN AUDIO PROCESSING:
1. STFT parameters: 400 window, 100 hop (was 320/160)
2. Power compression: 0.3 exponent
3. Segment size: 2.0 seconds (match CMGAN)
4. Non-overlapping segments for training

COMPATIBILITY:
- Kept old parameters (win_len, hop_len) for reference
- New code uses n_fft and hop_length explicitly
"""

import os

# ==================== EXPERIMENT CONFIGURATION ====================
exp_conf = {
    # CMGAN-STYLE STFT PARAMETERS (PRIMARY)
    'n_fft': 400,                  # FFT size: 400 samples = 25ms @ 16kHz
    'hop_length': 100,             # Hop size: 100 samples = 6.25ms @ 16kHz
    'power_compression': 0.3,      # Power compression exponent (CMGAN uses 0.3)
    
    # Legacy parameters (for reference, not used directly)
    'win_len': 0.025,              # 25ms = 400 samples @ 16kHz
    'hop_len': 0.00625,            # 6.25ms = 100 samples @ 16kHz
    
    # Audio parameters
    'sample_rate': 16000,          # 16 kHz sample rate
    'in_norm': None,               # No normalization in dataloader (done in forward pass)
}

# ==================== TRAINING CONFIGURATION ====================
train_conf = {
    # GPU settings
    'gpu_ids': '0',                # GPU IDs to use (e.g., '0' or '0,1,2,3')
    
    # Checkpoint settings
    'ckpt_dir': './checkpoints',   # Directory to save checkpoints
    'est_path': './estimates_MA',  # Directory to save estimated audio during testing
    'resume_model': '',            # Path to checkpoint to resume from (empty = train from scratch)
    
    # Optimizer settings
    'lr': 1e-3,                    # Initial learning rate
    'clip_norm': 5.0,              # Gradient clipping norm
    
    # Learning rate scheduler (ReduceLROnPlateau)
    'plateau_factor': 0.5,         # Multiply LR by this when plateau
    'plateau_patience': 3,         # Number of epochs with no improvement before reducing LR
    'plateau_threshold': 0.001,    # Threshold for measuring improvement
    'plateau_min_lr': 1e-6,        # Minimum learning rate
    
    # Training duration
    'max_n_epochs': 100,           # Maximum number of epochs
    'early_stop_patience': 10,     # Stop if no improvement for this many epochs at min LR
    
    # Data loading settings
    'batch_size': 4,               # Batch size for training
    'num_workers': 4,              # Number of parallel data loading workers
    
    # Data processing mode
    'unit': 'seg',                 # 'seg' = segment-based, 'utt' = utterance-based
    'segment_size': 2.0,           # Segment duration in seconds (CMGAN uses 2.0)
    'segment_shift': 2.0,          # Segment hop in seconds (2.0 = non-overlapping, like CMGAN)
    'max_length_seconds': 6.0,     # Maximum audio length to process
    
    # Logging
    'loss_log': 'loss.csv',        # CSV file to log training/validation loss
    'time_log': None,              # Path to detailed timing log (None = print to console)
}

# ==================== TEST CONFIGURATION ====================
test_conf = {
    'model_file': './checkpoints/models/best.pt',  # Model checkpoint to use for testing
    'batch_size': 1,               # Batch size for testing (usually 1)
    'num_workers': 2,              # Number of data loading workers
    'write_ideal': False,          # Whether to write ideal (analysis-synthesis) audio
}

# ==================== DATA DIRECTORIES ====================
# IMPORTANT: UPDATE THESE PATHS TO YOUR DATASET!

# Base directory (modify this to your dataset location)
BASE_DIR = '/path/to/your/dataset'

# Training data
TRAIN_CLEAN_DIR = os.path.join(BASE_DIR, 'train', 'clean')
TRAIN_NOISY_DIR = os.path.join(BASE_DIR, 'train', 'noisy')

# Validation data
VALID_CLEAN_DIR = os.path.join(BASE_DIR, 'valid', 'clean')
VALID_NOISY_DIR = os.path.join(BASE_DIR, 'valid', 'noisy')

# Test data
TEST_CLEAN_DIR = os.path.join(BASE_DIR, 'test', 'clean')
TEST_NOISY_DIR = os.path.join(BASE_DIR, 'test', 'noisy')


# ==================== VALIDATION FUNCTIONS ====================

def validate_data_dirs(mode='train'):
    """
    Validate that required data directories exist and contain WAV files.
    
    Args:
        mode: 'train' or 'test'
    
    Raises:
        FileNotFoundError: If directory doesn't exist
        ValueError: If directory is empty
    """
    if mode == 'train':
        dirs_to_check = [
            ('Train Clean', TRAIN_CLEAN_DIR),
            ('Train Noisy', TRAIN_NOISY_DIR),
            ('Valid Clean', VALID_CLEAN_DIR),
            ('Valid Noisy', VALID_NOISY_DIR),
        ]
    elif mode == 'test':
        dirs_to_check = [
            ('Test Clean', TEST_CLEAN_DIR),
            ('Test Noisy', TEST_NOISY_DIR),
        ]
    else:
        raise ValueError(f"Unknown mode: {mode}. Use 'train' or 'test'.")
    
    print("\nValidating data directories...")
    for name, path in dirs_to_check:
        if not os.path.isdir(path):
            raise FileNotFoundError(f"{name} directory not found: {path}")
        
        wav_files = [f for f in os.listdir(path) if f.endswith('.wav')]
        if len(wav_files) == 0:
            raise ValueError(f"{name} directory contains no WAV files: {path}")
        
        print(f"  ✓ {name}: {len(wav_files)} WAV files found")
    
    print("All directories validated successfully!\n")


def get_test_snr_dirs():
    """
    Auto-detect SNR-organized test directories.
    
    Expected structure:
        test/noisy/snr_-06/
        test/noisy/snr_000/
        test/noisy/snr_+06/
    
    Returns:
        list of (snr_value, directory_path) tuples, sorted by SNR
    
    Raises:
        FileNotFoundError: If test noisy directory doesn't exist
        ValueError: If no valid SNR directories or WAV files found
    """
    if not os.path.isdir(TEST_NOISY_DIR):
        raise FileNotFoundError(f"Test noisy directory not found: {TEST_NOISY_DIR}")
    
    # Look for SNR subdirectories
    snr_dirs = []
    for item in os.listdir(TEST_NOISY_DIR):
        item_path = os.path.join(TEST_NOISY_DIR, item)
        
        # Check if it's a directory starting with 'snr_'
        if os.path.isdir(item_path) and item.startswith('snr_'):
            try:
                # Extract SNR value from directory name
                # Expected format: snr_-06, snr_000, snr_+06
                snr_str = item.split('_')[1]
                snr_value = int(snr_str)
                
                # Check if directory contains WAV files
                wav_files = [f for f in os.listdir(item_path) if f.endswith('.wav')]
                if len(wav_files) > 0:
                    snr_dirs.append((snr_value, item_path))
            except (IndexError, ValueError):
                # Invalid directory name format, skip
                continue
    
    if len(snr_dirs) == 0:
        # No SNR subdirectories found, use TEST_NOISY_DIR directly
        wav_files = [f for f in os.listdir(TEST_NOISY_DIR) if f.endswith('.wav')]
        if len(wav_files) == 0:
            raise ValueError(f"No WAV files found in {TEST_NOISY_DIR}")
        return [(0, TEST_NOISY_DIR)]
    
    # Sort by SNR value (ascending)
    snr_dirs.sort(key=lambda x: x[0])
    return snr_dirs


# ==================== CONFIGURATION SUMMARY ====================

def print_config_summary():
    """Print a summary of current configuration."""
    print("\n" + "="*70)
    print("CONFIGURATION SUMMARY")
    print("="*70)
    print("\nAudio Processing (CMGAN-Style):")
    print(f"  STFT window:      {exp_conf['n_fft']} samples ({exp_conf['n_fft']/exp_conf['sample_rate']*1000:.1f} ms)")
    print(f"  STFT hop:         {exp_conf['hop_length']} samples ({exp_conf['hop_length']/exp_conf['sample_rate']*1000:.2f} ms)")
    print(f"  Frequency bins:   {exp_conf['n_fft']//2 + 1}")
    print(f"  Sample rate:      {exp_conf['sample_rate']} Hz")
    print(f"  Power compress:   {exp_conf['power_compression']}")
    
    print("\nTraining Settings:")
    print(f"  Batch size:       {train_conf['batch_size']}")
    print(f"  Learning rate:    {train_conf['lr']}")
    print(f"  Max epochs:       {train_conf['max_n_epochs']}")
    print(f"  Segment size:     {train_conf['segment_size']} seconds")
    print(f"  Segment shift:    {train_conf['segment_shift']} seconds")
    
    print("\nData Directories:")
    print(f"  Train clean:      {TRAIN_CLEAN_DIR}")
    print(f"  Train noisy:      {TRAIN_NOISY_DIR}")
    print(f"  Valid clean:      {VALID_CLEAN_DIR}")
    print(f"  Valid noisy:      {VALID_NOISY_DIR}")
    print(f"  Test clean:       {TEST_CLEAN_DIR}")
    print(f"  Test noisy:       {TEST_NOISY_DIR}")
    print("="*70 + "\n")


# ==================== MAIN ====================

if __name__ == '__main__':
    """Test configuration and validate directories."""
    print_config_summary()
    
    # Try to validate directories
    try:
        validate_data_dirs(mode='train')
    except (FileNotFoundError, ValueError) as e:
        print(f"⚠ Warning: {e}")
        print("\nPlease update the data directory paths in configs.py")
