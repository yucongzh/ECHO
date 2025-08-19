import torch
from torch import nn

class PatchEmbed(nn.Module):
    """ 1D Spectrogram to Patch Embedding with sliding window
        Input: (B, band_width, T)
        Output: (B, num_patches, embed_dim)
    """
    def __init__(self, band_width, shift_size, in_chans=1, embed_dim=768):
        super().__init__()
        
        self.band_width = band_width
        self.shift_size = shift_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        
        # Use 1D convolution with kernel size (band_width, shift_size)
        # This will slide horizontally across the spectrogram
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=(band_width, shift_size), stride=(band_width, shift_size))
        
    def forward(self, x):
        """
        Args:
            x: Input spectrograms (B, band_width, T)
        Returns:
            patches: (B, num_patches, embed_dim), num_patches is dynamic
        """
        if x.dim() == 3:
            x = x.unsqueeze(1)

        assert x.dim() == 4, f"Input shape must be (B, band_width, T), but got {x.shape}"
        B, C, H, W = x.shape
        
        # Ensure input height matches band_width
        assert H == self.band_width, f"Input height ({H}) doesn't match band_width ({self.band_width})"
        
        # Calculate how many patches we can extract
        num_patches = W // self.shift_size
        
        # If we have remaining width that's less than shift_size, pad it
        if W % self.shift_size != 0:
            padding_width = self.shift_size - (W % self.shift_size)
            x = torch.nn.functional.pad(x, (0, padding_width, 0, 0), mode='constant', value=0)
            num_patches += 1
        
        # Apply convolution to extract patches
        # Output shape: (B, embed_dim, 1, num_patches)
        patches = self.proj(x)
        
        # Reshape to (B, num_patches, embed_dim), num_patches is dynamic
        patches = patches.squeeze(2).transpose(1, 2)
        
        return patches
