"""
criteria.py - Loss Function with CMGAN-Style Audio Processing
==============================================================

MODIFICATIONS FOR CMGAN:
1. Updated __init__ to accept CMGAN STFT parameters (n_fft=400, hop=100, power=0.3)
2. Updated __call__ to accept norm_factors parameter
3. Updated resynthesizer initialization with CMGAN parameters

Loss Function: SI-SDR + 10.0 * L1(Real, Imaginary)
"""

import torch
import torch.nn.functional as F
from utils.pipeline_modules import Resynthesizer


def si_sdr(estimated, target, eps=1e-8):
    """
    Compute Scale-Invariant Signal-to-Distortion Ratio (SI-SDR) in dB.
    
    Args:
        estimated: [batch, samples] estimated signal
        target: [batch, samples] target signal
        eps: small constant for numerical stability
    
    Returns:
        si_sdr_value: scalar, SI-SDR in dB
    """
    # Handle dimension
    if estimated.dim() == 3:
        estimated = estimated.squeeze(1)
    if target.dim() == 3:
        target = target.squeeze(1)
    
    # Zero-mean normalization
    estimated = estimated - torch.mean(estimated, dim=1, keepdim=True)
    target = target - torch.mean(target, dim=1, keepdim=True)
    
    # Compute scaling factor
    alpha = (torch.sum(estimated * target, dim=1, keepdim=True) /
            (torch.sum(target ** 2, dim=1, keepdim=True) + eps))
    
    # Scale target
    target_scaled = alpha * target
    
    # Compute noise
    noise = estimated - target_scaled
    
    # Compute SI-SDR
    si_sdr_val = (torch.sum(target_scaled ** 2, dim=1) / 
                  (torch.sum(noise ** 2, dim=1) + eps))
    
    return 10 * torch.log10(si_sdr_val + eps).mean()


def si_sdr_loss(estimated, target):
    """
    SI-SDR loss (negative for minimization).
    
    Args:
        estimated: [batch, samples] estimated signal
        target: [batch, samples] target signal
    
    Returns:
        loss: scalar, negative SI-SDR
    """
    return -si_sdr(estimated, target)


def l1_loss_complex(est, ref):
    """
    L1 loss on real and imaginary components.
    
    Args:
        est: [batch, 2, frames, freq_bins] estimated spectrum
        ref: [batch, 2, frames, freq_bins] reference spectrum
    
    Returns:
        loss: scalar, L1 loss on real + L1 loss on imaginary
    """
    loss_real = F.l1_loss(est[:, 0], ref[:, 0])
    loss_imag = F.l1_loss(est[:, 1], ref[:, 1])
    return loss_real + loss_imag


class LossFunction(object):
    """
    Loss Function with CMGAN-Style Audio Processing.
    
    Loss = SI-SDR + 10.0 * L1(Real, Imaginary)
    
    MODIFIED FOR CMGAN:
    - Uses CMGAN STFT parameters (n_fft=400, hop=100)
    - Uses power compression (0.3)
    - Requires norm_factors for reconstruction
    
    Args:
        device: torch device
        n_fft: FFT size (400 for CMGAN)
        hop_length: Hop size (100 for CMGAN)
        power: Power compression exponent (0.3 for CMGAN)
    """
    
    def __init__(self, device, n_fft=400, hop_length=100, power=0.3):
        """
        Initialize loss function with CMGAN-style parameters.
        
        Args:
            device: torch device ('cuda' or 'cpu')
            n_fft: FFT size (default: 400 for CMGAN)
            hop_length: Hop size (default: 100 for CMGAN)
            power: Power compression exponent (default: 0.3 for CMGAN)
        """
        self.device = device
        
        # Create resynthesizer with CMGAN-style parameters
        self.resynthesizer = Resynthesizer(
            device=device,
            n_fft=n_fft,
            hop_length=hop_length,
            power=power
        )

    def __call__(self, est, lbl, loss_mask, n_frames, mix, n_samples, norm_factors):
        """
        Compute loss with CMGAN-style audio processing.
        
        Args:
            est: [batch, 2, frames, freq_bins] estimated spectrum
            lbl: [batch, 2, frames, freq_bins] target spectrum
            loss_mask: [batch, 2, frames, freq_bins] loss mask
            n_frames: [batch] or list, number of valid frames per sample
            mix: [batch, samples] noisy audio (for reconstruction reference)
            n_samples: [batch] number of valid samples per sample
            norm_factors: dict with keys:
                - 'mix_c': [batch] RMS normalization factors for mix
                - 'sph_c': [batch] RMS normalization factors for clean
                - 'original_length': int, original audio length
        
        Returns:
            loss: scalar tensor, total loss
        """
        # Apply loss mask to focus on valid frames
        est_masked = est * loss_mask
        lbl_masked = lbl * loss_mask
        
        # Time-domain reconstruction (MODIFIED: uses CMGAN-style resynthesizer with norm_factors)
        est_wave = self.resynthesizer(est_masked, mix, norm_factors)
        lbl_wave = self.resynthesizer(lbl_masked, mix, norm_factors)
        
        # Truncate to valid length
        T = n_samples[0].item()
        est_wave = est_wave[:, :T]
        lbl_wave = lbl_wave[:, :T]
        
        # Time-domain loss (SI-SDR)
        loss_sisdr = si_sdr_loss(est_wave, lbl_wave)
        
        # Frequency-domain loss (L1 on real and imaginary)
        loss_mae = l1_loss_complex(est_masked, lbl_masked)
        
        # Combined loss (weight = 10.0 for MAE)
        total_loss = loss_sisdr + 10.0 * loss_mae
        
        return total_loss


# ==================== TESTING CODE ====================

if __name__ == '__main__':
    """
    Test loss function with CMGAN-style processing.
    """
    print("="*70)
    print("Testing Loss Function with CMGAN-Style Processing")
    print("="*70)
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    
    # Create loss function
    criterion = LossFunction(device, n_fft=400, hop_length=100, power=0.3)
    print("✓ Loss function created")
    
    # Create dummy data
    batch_size = 2
    frames = 321  # ~2 seconds with hop=100
    freq_bins = 201  # n_fft=400 -> 201 bins
    audio_length = 32000  # 2 seconds @ 16kHz
    
    print(f"\nTest data:")
    print(f"  Batch size: {batch_size}")
    print(f"  Frames: {frames}")
    print(f"  Freq bins: {freq_bins}")
    print(f"  Audio length: {audio_length}")
    
    # Create dummy tensors
    est = torch.randn(batch_size, 2, frames, freq_bins).to(device)
    lbl = torch.randn(batch_size, 2, frames, freq_bins).to(device)
    loss_mask = torch.ones(batch_size, 2, frames, freq_bins).to(device)
    n_frames = torch.tensor([frames] * batch_size, dtype=torch.int64)
    mix = torch.randn(batch_size, audio_length).to(device)
    n_samples = torch.tensor([audio_length] * batch_size, dtype=torch.int64)
    
    # Create dummy norm_factors
    norm_factors = {
        'mix_c': torch.ones(batch_size).to(device),
        'sph_c': torch.ones(batch_size).to(device),
        'original_length': audio_length
    }
    
    print("\n" + "-"*70)
    print("Computing loss...")
    print("-"*70)
    
    try:
        loss = criterion(est, lbl, loss_mask, n_frames, mix, n_samples, norm_factors)
        print(f"\n✓ Loss computed successfully!")
        print(f"  Loss value: {loss.item():.4f}")
        print(f"  Loss type: {type(loss)}")
        print(f"  Requires grad: {loss.requires_grad}")
        
        # Test backward pass
        print("\n" + "-"*70)
        print("Testing backward pass...")
        print("-"*70)
        
        loss.backward()
        print("✓ Backward pass successful!")
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        raise
    
    print("\n" + "="*70)
    print("✓ ALL TESTS PASSED!")
    print("="*70)
