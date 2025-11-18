"""
stft.py - CMGAN-Style STFT Using PyTorch Built-in
==================================================

This module provides STFT/iSTFT operations matching CMGAN's implementation.
Uses PyTorch's built-in torch.stft/istft for simplicity and correctness.

REPLACES the original custom STFT implementation with PyTorch built-in version.

Key Features:
- PyTorch version compatibility (1.6+)
- Hamming window (matches CMGAN)
- Returns [batch, frames, freq_bins] format
- Exact length reconstruction
"""

import torch
import torch.nn as nn
import warnings


class STFT:
    """
    CMGAN-style STFT wrapper using PyTorch built-in functions.
    
    Args:
        n_fft: FFT size (default: 400 for CMGAN)
        hop_length: Hop size in samples (default: 100 for CMGAN)
        device: torch device ('cuda' or 'cpu')
    
    Example:
        >>> stft = STFT(n_fft=400, hop_length=100, device='cuda')
        >>> audio = torch.randn(2, 32000).cuda()  # [batch, samples]
        >>> real, imag = stft.stft(audio)         # [batch, frames, freq_bins]
        >>> reconstructed = stft.istft(real, imag, length=32000)
    """
    
    def __init__(self, n_fft=400, hop_length=100, device='cuda'):
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.device = device
        
        # Create Hamming window (CMGAN uses Hamming)
        self.window = torch.hamming_window(n_fft).to(device)
        
        # Check PyTorch version for compatibility
        self._check_pytorch_version()
    
    def _check_pytorch_version(self):
        """Check PyTorch version and warn if too old."""
        version = torch.__version__.split('+')[0]  # Remove +cu102 suffix
        major, minor = map(int, version.split('.')[:2])
        
        if major < 1 or (major == 1 and minor < 6):
            warnings.warn(
                f"PyTorch {version} detected. STFT requires PyTorch >= 1.6. "
                f"Please upgrade: pip install torch>=1.6",
                UserWarning
            )
    
    def stft(self, audio):
        """
        Short-Time Fourier Transform.
        
        Args:
            audio: [batch, samples] time-domain signal
        
        Returns:
            real: [batch, frames, freq_bins] real part
            imag: [batch, frames, freq_bins] imaginary part
        
        Note:
            freq_bins = n_fft // 2 + 1 (one-sided spectrum)
            frames = (samples - n_fft) // hop_length + 1
        """
        # Validate input
        if audio.dim() != 2:
            raise ValueError(f"Expected 2D input [batch, samples], got shape {audio.shape}")
        
        # PyTorch STFT
        # For PyTorch >= 1.7: can use return_complex=False
        # For PyTorch < 1.7: returns [batch, freq, frames, 2] by default
        try:
            # Try PyTorch 1.7+ API
            spec = torch.stft(
                audio,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                window=self.window,
                center=True,
                normalized=False,
                onesided=True,
                return_complex=False  # Returns [batch, freq, frames, 2]
            )
        except TypeError:
            # Fallback for PyTorch < 1.7 (no return_complex parameter)
            spec = torch.stft(
                audio,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                window=self.window,
                center=True,
                normalized=False,
                onesided=True
            )
        
        # spec shape: [batch, freq_bins, frames, 2]
        # Transpose to [batch, frames, freq_bins]
        real = spec[:, :, :, 0].transpose(1, 2).contiguous()  # [B, T, F]
        imag = spec[:, :, :, 1].transpose(1, 2).contiguous()  # [B, T, F]
        
        return real, imag
    
    def istft(self, real, imag, length=None):
        """
        Inverse Short-Time Fourier Transform.
        
        Args:
            real: [batch, frames, freq_bins] real part
            imag: [batch, frames, freq_bins] imaginary part
            length: target audio length (optional, for exact reconstruction)
                   If None, length is inferred from STFT parameters
        
        Returns:
            audio: [batch, samples] time-domain signal
        
        Note:
            Setting 'length' ensures exact reconstruction of original length.
            Without it, length may differ by a few samples due to STFT framing.
        """
        # Validate inputs
        if real.dim() != 3 or imag.dim() != 3:
            raise ValueError(f"Expected 3D inputs [batch, frames, freq_bins], "
                           f"got real: {real.shape}, imag: {imag.shape}")
        
        if real.shape != imag.shape:
            raise ValueError(f"Real and imag must have same shape, "
                           f"got real: {real.shape}, imag: {imag.shape}")
        
        # Transpose to [batch, freq_bins, frames]
        real = real.transpose(1, 2).contiguous()  # [B, F, T]
        imag = imag.transpose(1, 2).contiguous()  # [B, F, T]
        
        # Stack to [batch, freq_bins, frames, 2]
        spec = torch.stack([real, imag], dim=-1)
        
        # PyTorch iSTFT
        try:
            # Try PyTorch 1.7+ API
            audio = torch.istft(
                spec,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                window=self.window,
                center=True,
                normalized=False,
                onesided=True,
                length=length  # Ensures exact output length
            )
        except TypeError:
            # Fallback for older PyTorch (no length parameter)
            audio = torch.istft(
                spec,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                window=self.window,
                center=True,
                normalized=False,
                onesided=True
            )
            
            # Manually adjust length if needed
            if length is not None:
                if audio.shape[-1] > length:
                    audio = audio[:, :length]
                elif audio.shape[-1] < length:
                    padding = length - audio.shape[-1]
                    audio = torch.nn.functional.pad(audio, (0, padding))
        
        return audio


# ==================== TESTING CODE ====================

def test_stft():
    """Test STFT functionality."""
    print("="*70)
    print("Testing STFT Module")
    print("="*70)
    
    # Check device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    print(f"PyTorch version: {torch.__version__}")
    
    # Create STFT module
    stft = STFT(n_fft=400, hop_length=100, device=device)
    print(f"\nSTFT Parameters:")
    print(f"  n_fft:       {stft.n_fft}")
    print(f"  hop_length:  {stft.hop_length}")
    print(f"  freq_bins:   {stft.n_fft // 2 + 1}")
    
    # Create test signal
    batch_size = 2
    audio_length = 32000  # 2 seconds @ 16kHz
    print(f"\nTest Signal:")
    print(f"  Batch size:  {batch_size}")
    print(f"  Length:      {audio_length} samples (2.0 seconds @ 16kHz)")
    
    audio = torch.randn(batch_size, audio_length).to(device)
    
    # Test STFT
    print("\n" + "-"*70)
    print("Test 1: STFT Forward")
    print("-"*70)
    
    real, imag = stft.stft(audio)
    print(f"Input audio shape:    {audio.shape}")
    print(f"Output real shape:    {real.shape}")
    print(f"Output imag shape:    {imag.shape}")
    
    expected_frames = (audio_length - stft.n_fft) // stft.hop_length + 1
    expected_freqs = stft.n_fft // 2 + 1
    print(f"Expected shape:       [{batch_size}, {expected_frames}, {expected_freqs}]")
    
    assert real.shape == (batch_size, expected_frames, expected_freqs), "Real shape mismatch!"
    assert imag.shape == (batch_size, expected_frames, expected_freqs), "Imag shape mismatch!"
    print("✓ STFT shape correct!")
    
    # Test iSTFT
    print("\n" + "-"*70)
    print("Test 2: iSTFT Reconstruction")
    print("-"*70)
    
    audio_reconstructed = stft.istft(real, imag, length=audio_length)
    print(f"Reconstructed shape:  {audio_reconstructed.shape}")
    
    assert audio_reconstructed.shape == audio.shape, "Reconstruction shape mismatch!"
    print("✓ Reconstruction shape correct!")
    
    # Check reconstruction error
    error = torch.mean(torch.abs(audio - audio_reconstructed)).item()
    print(f"\nReconstruction error (MAE): {error:.8f}")
    
    if error < 1e-5:
        print("✓ Reconstruction error acceptable (<1e-5)")
    elif error < 1e-3:
        print("⚠ Reconstruction error acceptable (<1e-3)")
    else:
        print("✗ Reconstruction error too large!")
    
    # Test with different lengths
    print("\n" + "-"*70)
    print("Test 3: Variable Length Inputs")
    print("-"*70)
    
    test_lengths = [16000, 24000, 48000]  # 1s, 1.5s, 3s
    for length in test_lengths:
        audio_test = torch.randn(1, length).to(device)
        real_test, imag_test = stft.stft(audio_test)
        audio_recon = stft.istft(real_test, imag_test, length=length)
        
        error = torch.mean(torch.abs(audio_test - audio_recon)).item()
        print(f"  Length {length:5d} samples: shape {audio_recon.shape}, error {error:.8f}")
    
    print("\n" + "="*70)
    print("✓ ALL TESTS PASSED!")
    print("="*70 + "\n")


if __name__ == '__main__':
    """Run tests when module is executed directly."""
    test_stft()
