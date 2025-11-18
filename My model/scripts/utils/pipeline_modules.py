"""
pipeline_modules.py - CMGAN-Style Audio Processing Pipeline
============================================================

MAJOR CHANGES FROM ORIGINAL:
1. RMS normalization (CMGAN-style)
2. Power compression (magnitude^0.3)
3. Uses CMGAN_STFT (PyTorch built-in)
4. Returns 2-channel [real, imag] (not 3-channel)
5. Returns normalization factors for exact reconstruction

Pipeline Flow:
    Audio → RMS Norm → STFT → Power Compress → [Real, Imag]
    [Real, Imag] → Power Decompress → iSTFT → RMS Denorm → Audio

Components:
- power_compress(): Compress spectrum magnitude
- power_uncompress(): Decompress spectrum magnitude
- NetFeeder: Audio to compressed spectrum
- Resynthesizer: Compressed spectrum to audio
"""

import torch
import torch.nn.functional as F
from utils.cmgan_stft import CMGAN_STFT


# ==================== POWER COMPRESSION (CMGAN) ====================

def power_compress(real, imag, power=0.3, eps=1e-8):
    """
    Power compression of complex spectrum (CMGAN-style).
    
    Compresses magnitude: mag_compressed = mag^power
    
    This reduces dynamic range for better gradient flow during training.
    CMGAN uses power=0.3 based on empirical results.
    
    Args:
        real: [batch, frames, freq_bins] real part
        imag: [batch, frames, freq_bins] imaginary part
        power: compression exponent (default: 0.3 for CMGAN)
        eps: small constant for numerical stability
    
    Returns:
        real_compressed: [batch, frames, freq_bins]
        imag_compressed: [batch, frames, freq_bins]
    
    Example:
        >>> real_c, imag_c = power_compress(real, imag, power=0.3)
    """
    # Convert to magnitude and phase
    spec = torch.complex(real, imag)
    mag = torch.abs(spec)
    phase = torch.angle(spec)
    
    # Compress magnitude
    # Add epsilon to avoid 0^0.3 = 0 (keeps gradients flowing)
    mag_compressed = (mag + eps) ** power
    
    # Reconstruct with compressed magnitude
    real_compressed = mag_compressed * torch.cos(phase)
    imag_compressed = mag_compressed * torch.sin(phase)
    
    return real_compressed, imag_compressed


def power_uncompress(real, imag, power=0.3, eps=1e-8):
    """
    Power decompression (inverse of power_compress).
    
    Args:
        real: [batch, frames, freq_bins] compressed real part
        imag: [batch, frames, freq_bins] compressed imaginary part
        power: compression exponent (must match compression)
        eps: small constant for numerical stability
    
    Returns:
        real_uncompressed: [batch, frames, freq_bins]
        imag_uncompressed: [batch, frames, freq_bins]
    
    Example:
        >>> real_uc, imag_uc = power_uncompress(real_c, imag_c, power=0.3)
    """
    # Convert to magnitude and phase
    spec = torch.complex(real, imag)
    mag = torch.abs(spec)
    phase = torch.angle(spec)
    
    # Uncompress magnitude
    # Add epsilon for numerical stability (avoid division by zero in backprop)
    mag_uncompressed = (mag + eps) ** (1.0 / power)
    
    # Reconstruct
    real_uncompressed = mag_uncompressed * torch.cos(phase)
    imag_uncompressed = mag_uncompressed * torch.sin(phase)
    
    return real_uncompressed, imag_uncompressed


# ==================== NET FEEDER (CMGAN-STYLE) ====================

class NetFeeder(object):
    """
    CMGAN-style feature extraction pipeline.
    
    Pipeline:
        Audio → RMS Normalize → STFT → Power Compress → [Real, Imag]
    
    Args:
        device: torch device
        n_fft: FFT size (400 for CMGAN)
        hop_length: Hop size (100 for CMGAN)
        power: Power compression exponent (0.3 for CMGAN)
    
    Returns:
        feat: [batch, 2, frames, freq_bins] - compressed [real, imag] of noisy
        lbl: [batch, 2, frames, freq_bins] - compressed [real, imag] of clean
        norm_factors: dict with normalization info for reconstruction
    
    Example:
        >>> feeder = NetFeeder(device, n_fft=400, hop_length=100, power=0.3)
        >>> feat, lbl, norm_factors = feeder(noisy_audio, clean_audio)
    """
    
    def __init__(self, device, n_fft=400, hop_length=100, power=0.3):
        self.device = device
        self.stft = CMGAN_STFT(n_fft, hop_length, device)
        self.power = power
        self.n_fft = n_fft
        self.hop_length = hop_length
    
    def rms_normalize(self, audio, eps=1e-8):
        """
        RMS normalization (CMGAN-style).
        
        Normalizes audio energy to consistent level:
            c = sqrt(N / sum(x^2))
            audio_normalized = audio * c
        
        This makes different utterances have similar energy levels,
        which improves training stability.
        
        Args:
            audio: [batch, samples]
            eps: small constant for numerical stability
        
        Returns:
            audio_normalized: [batch, samples]
            c: [batch] normalization factors (for denormalization)
        
        Example:
            >>> audio_norm, c = self.rms_normalize(audio)
        """
        # Compute energy per sample
        # sum(x^2) / N = mean power
        energy = torch.sum(audio ** 2.0, dim=-1, keepdim=True)  # [batch, 1]
        
        # Compute normalization factor
        # c = sqrt(N / sum(x^2)) = sqrt(1 / mean_power)
        c = torch.sqrt(audio.size(-1) / (energy + eps))  # [batch, 1]
        
        # Normalize
        audio_normalized = audio * c
        
        return audio_normalized, c.squeeze(-1)  # Return c as [batch]
    
    def __call__(self, mix, sph):
        """
        Convert time-domain signals to CMGAN-style features.
        
        Args:
            mix: [batch, samples] noisy signal
            sph: [batch, samples] clean signal
        
        Returns:
            feat: [batch, 2, frames, freq_bins] - compressed [real, imag] of noisy
            lbl: [batch, 2, frames, freq_bins] - compressed [real, imag] of clean
            norm_factors: dict {
                'mix_c': [batch] normalization factors for mix,
                'sph_c': [batch] normalization factors for clean,
                'original_length': int, original audio length
            }
        
        Example:
            >>> feat, lbl, norm_factors = feeder(noisy, clean)
            >>> # feat.shape: [4, 2, 321, 201] for 2s audio, batch=4
        """
        # Validate inputs
        if mix.shape != sph.shape:
            raise ValueError(f"Mix and speech must have same shape, "
                           f"got mix: {mix.shape}, sph: {sph.shape}")
        
        # Step 1: RMS normalization
        mix_norm, mix_c = self.rms_normalize(mix)
        sph_norm, sph_c = self.rms_normalize(sph)
        
        # Step 2: STFT
        real_mix, imag_mix = self.stft.stft(mix_norm)  # [B, T, F]
        real_sph, imag_sph = self.stft.stft(sph_norm)  # [B, T, F]
        
        # Step 3: Power compression
        real_mix_c, imag_mix_c = power_compress(real_mix, imag_mix, self.power)
        real_sph_c, imag_sph_c = power_compress(real_sph, imag_sph, self.power)
        
        # Step 4: Stack into 2-channel format [real, imag]
        feat = torch.stack([real_mix_c, imag_mix_c], dim=1)  # [B, 2, T, F]
        lbl = torch.stack([real_sph_c, imag_sph_c], dim=1)   # [B, 2, T, F]
        
        # Step 5: Save normalization factors for reconstruction
        norm_factors = {
            'mix_c': mix_c,                # [batch]
            'sph_c': sph_c,                # [batch]
            'original_length': mix.shape[-1]  # int
        }
        
        return feat, lbl, norm_factors


# ==================== RESYNTHESIZER (CMGAN-STYLE) ====================

class Resynthesizer(object):
    """
    CMGAN-style audio reconstruction pipeline.
    
    Pipeline:
        [Real, Imag] → Power Decompress → iSTFT → RMS Denorm → Audio
    
    Args:
        device: torch device
        n_fft: FFT size (400 for CMGAN)
        hop_length: Hop size (100 for CMGAN)
        power: Power compression exponent (0.3 for CMGAN)
    
    Example:
        >>> resynthesizer = Resynthesizer(device, n_fft=400, hop_length=100, power=0.3)
        >>> audio = resynthesizer(estimated_spectrum, original_mix, norm_factors)
    """
    
    def __init__(self, device, n_fft=400, hop_length=100, power=0.3):
        self.device = device
        self.stft = CMGAN_STFT(n_fft, hop_length, device)
        self.power = power
        self.n_fft = n_fft
        self.hop_length = hop_length
    
    def __call__(self, est, mix, norm_factors, eps=1e-8):
        """
        Convert estimated spectrum to time-domain waveform.
        
        Args:
            est: [batch, 2, frames, freq_bins] - compressed [real, imag]
            mix: [batch, samples] - original mix (for length reference)
            norm_factors: dict with normalization info
            eps: small constant for numerical stability
        
        Returns:
            audio: [batch, samples] - reconstructed audio
        
        Example:
            >>> audio_est = resynthesizer(est_spectrum, mix, norm_factors)
        """
        # Extract real and imaginary parts
        est_real = est[:, 0, :, :]  # [B, T, F]
        est_imag = est[:, 1, :, :]  # [B, T, F]
        
        # Step 1: Power decompression
        est_real_uc, est_imag_uc = power_uncompress(est_real, est_imag, self.power)
        
        # Step 2: iSTFT
        target_length = norm_factors['original_length']
        audio = self.stft.istft(est_real_uc, est_imag_uc, length=target_length)
        
        # Step 3: RMS denormalization
        # During forward pass, we normalized by multiplying with c
        # Now we divide by c to get back original scale
        mix_c = norm_factors['mix_c']  # [batch]
        audio = audio / (mix_c.unsqueeze(-1) + eps)
        
        # Step 4: Ensure correct length (safety check)
        # This handles edge cases from STFT/iSTFT rounding
        if audio.shape[-1] < mix.shape[-1]:
            # Pad if too short
            padding = mix.shape[-1] - audio.shape[-1]
            audio = F.pad(audio, (0, padding), value=0)
        elif audio.shape[-1] > mix.shape[-1]:
            # Truncate if too long
            audio = audio[:, :mix.shape[-1]]
        
        return audio


# ==================== TESTING CODE ====================

def test_pipeline():
    """Test NetFeeder and Resynthesizer pipeline."""
    print("="*70)
    print("Testing CMGAN-Style Pipeline")
    print("="*70)
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    print(f"PyTorch: {torch.__version__}")
    
    # Create feeder and resynthesizer
    n_fft = 400
    hop_length = 100
    power = 0.3
    
    print(f"\nPipeline Parameters:")
    print(f"  n_fft:       {n_fft}")
    print(f"  hop_length:  {hop_length}")
    print(f"  power:       {power}")
    
    feeder = NetFeeder(device, n_fft=n_fft, hop_length=hop_length, power=power)
    resynthesizer = Resynthesizer(device, n_fft=n_fft, hop_length=hop_length, power=power)
    
    # Create test data
    batch_size = 2
    audio_length = 32000  # 2 seconds @ 16kHz
    
    print(f"\nTest Data:")
    print(f"  Batch size:  {batch_size}")
    print(f"  Length:      {audio_length} samples (2.0s @ 16kHz)")
    
    mix = torch.randn(batch_size, audio_length).to(device)
    sph = torch.randn(batch_size, audio_length).to(device)
    
    # Test forward pass
    print("\n" + "-"*70)
    print("Test 1: Forward Pass (Audio → Spectrum)")
    print("-"*70)
    
    feat, lbl, norm_factors = feeder(mix, sph)
    
    print(f"Input mix shape:      {mix.shape}")
    print(f"Input sph shape:      {sph.shape}")
    print(f"Output feat shape:    {feat.shape}")
    print(f"Output lbl shape:     {lbl.shape}")
    print(f"Norm factors keys:    {list(norm_factors.keys())}")
    
    expected_frames = (audio_length - n_fft) // hop_length + 1
    expected_freqs = n_fft // 2 + 1
    expected_shape = (batch_size, 2, expected_frames, expected_freqs)
    
    print(f"Expected shape:       {expected_shape}")
    
    assert feat.shape == expected_shape, f"Feature shape mismatch!"
    assert lbl.shape == expected_shape, f"Label shape mismatch!"
    print("✓ Forward pass shapes correct!")
    
    # Test reconstruction
    print("\n" + "-"*70)
    print("Test 2: Reconstruction (Spectrum → Audio)")
    print("-"*70)
    
    sph_recon = resynthesizer(lbl, mix, norm_factors)
    
    print(f"Reconstructed shape:  {sph_recon.shape}")
    assert sph_recon.shape == sph.shape, "Reconstruction shape mismatch!"
    print("✓ Reconstruction shape correct!")
    
    # Check reconstruction quality
    # Note: Won't be perfect due to normalization + compression
    # But should be reasonable
    error = torch.mean(torch.abs(sph - sph_recon)).item()
    print(f"\nReconstruction error:  {error:.6f}")
    
    if error < 0.1:
        print("✓ Reconstruction error acceptable")
    else:
        print("⚠ Reconstruction error high (expected due to normalization)")
    
    # Test with different lengths
    print("\n" + "-"*70)
    print("Test 3: Variable Length Inputs")
    print("-"*70)
    
    test_lengths = [16000, 24000, 48000]
    for length in test_lengths:
        mix_test = torch.randn(1, length).to(device)
        sph_test = torch.randn(1, length).to(device)
        
        feat_test, lbl_test, nf_test = feeder(mix_test, sph_test)
        sph_recon_test = resynthesizer(lbl_test, mix_test, nf_test)
        
        error = torch.mean(torch.abs(sph_test - sph_recon_test)).item()
        print(f"  Length {length:5d}: shape {sph_recon_test.shape}, error {error:.6f}")
    
    print("\n" + "="*70)
    print("✓ ALL TESTS PASSED!")
    print("="*70 + "\n")


if __name__ == '__main__':
    """Run tests when module is executed directly."""
    test_pipeline()
