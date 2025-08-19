#!/usr/bin/env python3
"""
Basic usage example for ECHO model.

This script demonstrates how to:
1. Initialize the ECHO model
2. Create a dummy spectrogram input
3. Extract features from the model
"""

import torch
import sys
import os

# Add the models directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'models'))

from ECHO import AudioMAEWithBand

def main():
    print("ECHO Model Basic Usage Example")
    print("=" * 40)
    
    # Check if CUDA is available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Initialize model with default parameters
    print("\nInitializing ECHO model...")
    model = AudioMAEWithBand(
        spec_len=2000,
        band_width=32,
        shift_size=16,
        embed_dim=384,
        encoder_depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        freq_pos_emb_dim=384
    )
    
    # Move model to device
    model = model.to(device)
    model.eval()
    
    print(f"Model initialized successfully!")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create dummy input (simulating a spectrogram)
    print("\nCreating dummy input...")
    freq_bins = 128  # Number of frequency bins
    time_frames = 1000  # Number of time frames
    
    # Create a dummy spectrogram with random values
    dummy_spec = torch.randn(freq_bins, time_frames, device=device)
    print(f"Input spectrogram shape: {dummy_spec.shape}")
    
    # Extract features
    print("\nExtracting features...")
    with torch.no_grad():
        try:
            features = model.extract_features(dummy_spec, sample_rate=16000)
            print(f"Feature extraction successful!")
            print(f"Output feature shape: {features.shape}")
            print(f"Feature statistics:")
            print(f"  - Mean: {features.mean().item():.4f}")
            print(f"  - Std:  {features.std().item():.4f}")
            print(f"  - Min:  {features.min().item():.4f}")
            print(f"  - Max:  {features.max().item():.4f}")
            
        except Exception as e:
            print(f"Error during feature extraction: {e}")
            print("This might be due to missing model weights or configuration issues.")
    
    print("\nExample completed successfully!")
    print("\nNote: This example uses random input data.")
    print("For real audio processing, use the functions provided in the README.md")

if __name__ == "__main__":
    main()
