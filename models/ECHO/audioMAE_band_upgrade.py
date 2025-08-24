import torch
import torch.nn as nn
from einops import rearrange
from modules import PatchEmbed
from utils import get_2d_sincos_pos_embed, get_sincos_encoding_1d, get_1d_sincos_pos_embed_from_grid
import numpy as np
import os

class AudioMAEWithBand(nn.Module):
    def __init__(self, spec_len=1024, band_width=32, shift_size=16, in_chans=1,
                 embed_dim=768, encoder_depth=12, num_heads=12,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, fix_pos_emb=True, mask_ratio=0.8, freq_pos_emb_dim=768):
        super().__init__()
        self.band_width = band_width
        self.shift_size = shift_size
        self.embed_dim = embed_dim
        self.norm_layer = norm_layer
        self.fix_pos_emb = fix_pos_emb
        self.in_chans = in_chans
        self.encoder_depth = encoder_depth
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.spec_len = spec_len
        self.mask_ratio = mask_ratio
        self.freq_pos_emb_dim = freq_pos_emb_dim

        # Patch embedding for each band with rectangular patches
        # Height equals band_width, width equals shift_size
        self.patch_embed = PatchEmbed(
            band_width=band_width,
            shift_size=shift_size,
            in_chans=in_chans,
            embed_dim=embed_dim
        )
        self.grid_h = 1  # Only one row since height equals band_width
        self.grid_w = spec_len // shift_size # number of patches in time dimension
        # Calculate total number of patches including padding
        if spec_len % shift_size != 0:
            self.num_patches_per_band = (spec_len // shift_size) + 1
        else:
            self.num_patches_per_band = spec_len // shift_size

        # cls token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        # pos embedding (for each band), including the CLS token position
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches_per_band + 1, embed_dim),
            requires_grad=(not self.fix_pos_emb)
        ) # (1, self.grid_w+1, D)
        # Global frequency positional embedding
        self.freq_pos_emb_dim = freq_pos_emb_dim # default to be same as embed_dim
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=int(embed_dim * mlp_ratio),
                activation='gelu',
                batch_first=True, 
                layer_norm_eps=1e-6,
                dropout=0.1,
            ),
            num_layers=encoder_depth
        )
        self.norm = norm_layer(embed_dim)
        self._init_weights()

    @classmethod
    def from_pretrained(cls, checkpoint_path, device=None, **kwargs):
        """
        Load a pre-trained ECHO model from checkpoint.
        
        Args:
            checkpoint_path: Path to the checkpoint file
            device: Device to load the model on (default: auto-detect)
            **kwargs: Additional arguments to override default config
            
        Returns:
            model: Loaded ECHO model
        """
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        # Auto-detect device if not specified
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Get model_type from checkpoint
        model_type = checkpoint.get('model_type', 'small')
        print(f"Detected model type: {model_type}")
        
        # Define model configurations based on model_type
        model_configs = {
            'small': {
                'embed_dim': 384,
                'encoder_depth': 12,
                'num_heads': 6,
                'freq_pos_emb_dim': 384,
            },
            'base': {
                'embed_dim': 768,
                'encoder_depth': 12,
                'num_heads': 12,
                'freq_pos_emb_dim': 768,
            },
            'large': {
                'embed_dim': 1024,
                'encoder_depth': 24,
                'num_heads': 16,
                'freq_pos_emb_dim': 1024,
            }
        }
        
        # Get config for the detected model type
        if model_type not in model_configs:
            print(f"Warning: Unknown model type '{model_type}', using 'small' config")
            model_type = 'small'
        
        config = model_configs[model_type]
        
        # Override with any provided kwargs
        for key, value in kwargs.items():
            if key in config:
                config[key] = value
        
        # Set default values for other parameters
        config.update({
            'spec_len': 2000,
            'band_width': 32,
            'shift_size': 16,
            'in_chans': 1,
            'mlp_ratio': 4.0,
            'norm_layer': nn.LayerNorm,
            'fix_pos_emb': True,
            'mask_ratio': 0.75,
        })
        
        print(f"Model configuration: {config}")
        
        # Create model with extracted config
        model = cls(**config)
        
        # Load weights
        if 'encoder' in checkpoint:
            model.load_state_dict(checkpoint['encoder'])
        else:
            model.load_state_dict(checkpoint)
        
        # Move to device
        model = model.to(device)
        model.eval()
        
        print(f"ECHO model loaded from {checkpoint_path}")
        print(f"Running on device: {device}")
        
        return model

    def _init_weights(self):
        # Use 1D position embedding since we only have time dimension patches
        # Generate 1D position indices for time dimension
        pos_indices = np.arange(self.grid_w, dtype=np.float32)
        pos_embed = get_1d_sincos_pos_embed_from_grid(
            self.pos_embed.shape[-1], pos_indices) # (self.grid_w, D)
        # Add CLS token position embedding
        cls_pos_embed = np.zeros([1, self.pos_embed.shape[-1]])
        pos_embed = np.concatenate([cls_pos_embed, pos_embed], axis=0) # (self.grid_w+1, D)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        torch.nn.init.normal_(self.cls_token, std=.02)
        self.apply(self._init_module_weights)

    def _init_module_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def band_split(self, x, freq_lengths):
        """
        Split a batch of spectrograms into frequency bands.
        Args:
            x: (B, 1, F, T) full spectrograms
            freq_lengths: (B,) frequency lengths for each sample (not used here, kept for interface compatibility)
        Returns:
            bands: (total_bands, 1, band_width, T)
            band_batch_indices: (total_bands,) each band belongs to which batch
            band_patch_indices: (total_bands, 2) [start_freq, end_freq) for each band
        """
        B, C, F, T = x.shape
        bands = []
        band_batch_indices = []
        band_patch_indices = []
        for b in range(B):
            for start in range(0, F, self.band_width):
                end = min(start + self.band_width, F)
                band = x[b:b+1, :, start:end, :]
                # pad if last band is smaller than band_width
                if end - start < self.band_width:
                    pad = self.band_width - (end - start)
                    band = torch.nn.functional.pad(band, (0, 0, 0, pad))
                bands.append(band)
                band_batch_indices.append(b)
                band_patch_indices.append([start, end])
        bands = torch.cat(bands, dim=0) if bands else torch.empty(0, 1, self.band_width, T, device=x.device)
        band_batch_indices = torch.tensor(band_batch_indices, device=x.device, dtype=torch.long)
        band_patch_indices = torch.tensor(band_patch_indices, device=x.device, dtype=torch.long)
        return bands, band_batch_indices, band_patch_indices

    def patchify(self, bands):
        # bands: (total_bands, band_width, T)
        patches = self.patch_embed(bands)  # (total_bands, N_patch, embed_dim)
        return patches

    def random_masking(self, patches, mask_ratio):
        """
        Mask patches while maintaining their original order when mask_ratio=0.
        This ensures positional encodings remain consistent with patch content.
        
        Args:
            patches: (N, L, D) tensor of patches
            mask_ratio: ratio of patches to mask
            
        Returns:
            patches_masked: masked patches in original order
            mask: binary mask indicating masked positions
            ids_restore: indices to restore original order (identity in this case)
            ids_keep: indices of kept patches in original order
        """
        N, L, D = patches.shape # N: number of bands, L: number of patches, D: embedding dimension
        len_keep = int(L * (1 - mask_ratio))
        
        # If mask_ratio=0, keep all patches in original order
        if mask_ratio == 0:
            patches_masked = patches
            mask = torch.zeros([N, L], device=patches.device)  # All patches kept
            ids_restore = torch.arange(L, device=patches.device).unsqueeze(0).expand(N, -1)  # Identity mapping
            ids_keep = torch.arange(L, device=patches.device).unsqueeze(0).expand(N, -1)  # All indices kept
            return patches_masked, mask, ids_restore, ids_keep
        
        # Generate random noise for masking decisions
        noise = torch.rand(N, L, device=patches.device)  # noise in [0, 1]

        # Sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # Keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        patches_unmasked = torch.gather(patches, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # Generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=patches.device)
        mask[:, :len_keep] = 0
        # Unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return patches_unmasked, mask, ids_restore, ids_keep

    def _compute_frequency_position_encoding(self, spectrogram, sample_rate):
        """
        Band splitting and frequency position encoding for single spectrogram.
        
        Args:
            spectrogram: Single spectrogram tensor (freq_bins, time)
            sample_rate: Audio sample rate (scalar)
            
        Returns:
            freq_pos_emb: Frequency position embeddings for all bands
            band_spectrograms: List of band spectrograms (num_bands, band_width, time)
            band_patch_indices: List of band patch indices (num_bands, 2)
        """
        num_freq_bins = spectrogram.shape[0]
        num_bands = max(1, (num_freq_bins + self.band_width - 1) // self.band_width)
        
        band_spectrograms = []
        band_patch_indices = []
        all_freq_pos_norm = []
        
        for band_idx in range(num_bands):
            start_freq = band_idx * self.band_width
            end_freq = min((band_idx + 1) * self.band_width, num_freq_bins)
            
            if end_freq > start_freq:
                band_spec = spectrogram[start_freq:end_freq, :]  # (band_freq, time)
                
                # Pad the band if it's smaller than band_width
                if band_spec.shape[0] < self.band_width:
                    padding_height = self.band_width - band_spec.shape[0]
                    band_spec = torch.nn.functional.pad(band_spec, (0, 0, padding_height, 0), mode='constant', value=0)
                
                band_spectrograms.append(band_spec)
                band_patch_indices.append(torch.tensor([start_freq, end_freq], dtype=torch.long, device=spectrogram.device))
                
                # Compute frequency position encoding
                center_bin = (start_freq + end_freq - 1) / 2.0
                n_fft = int(0.025 * sample_rate)
                center_freq = center_bin * sample_rate / n_fft
                nyquist = sample_rate / 2.0
                freq_pos_norm = center_freq / nyquist if nyquist > 0 else 0.0
                all_freq_pos_norm.append(freq_pos_norm)
        
        # Convert to tensors
        if band_spectrograms:
            band_spectrograms = torch.stack(band_spectrograms, dim=0)  # (num_bands, band_width, T)
            band_patch_indices = torch.stack(band_patch_indices, dim=0)  # (num_bands, 2)
            freq_pos_norm = torch.tensor(all_freq_pos_norm, dtype=torch.float32, device=spectrogram.device).unsqueeze(1)  # (num_bands, 1)
            freq_pos_emb = get_sincos_encoding_1d(freq_pos_norm, self.freq_pos_emb_dim)  # (num_bands, freq_pos_emb_dim)
            
        else:
            raise ValueError("No bands found in spectrogram")
        
        return freq_pos_emb, band_spectrograms, band_patch_indices


    def _compute_frequency_position_encoding_random(self, spectrogram, sample_rate):
        """
        Band splitting and frequency position encoding for single spectrogram.
        
        Args:
            spectrogram: Single spectrogram tensor (freq_bins, time)
            sample_rate: Audio sample rate (scalar)
            
        Returns:
            freq_pos_emb: Frequency position embeddings for all bands
            band_spectrograms: List of band spectrograms (num_bands, band_width, time)
            band_patch_indices: List of band patch indices (num_bands, 2)
        """
        num_freq_bins = spectrogram.shape[0]

        # randomly select 2-3 num_bands bands
        num_bands = np.random.randint(2, 6)
        band_spectrograms = []
        band_patch_indices = []
        all_freq_pos_norm = []
        for _ in range(num_bands):
            start_freq = np.random.randint(0, num_freq_bins - self.band_width)
            end_freq = start_freq + self.band_width
            band_spec = spectrogram[start_freq:end_freq, :]  # (band_freq, time)
            band_spectrograms.append(band_spec)
            band_patch_indices.append(torch.tensor([start_freq, end_freq], dtype=torch.long, device=spectrogram.device))
            center_bin = (start_freq + end_freq - 1) / 2.0
            n_fft = int(0.025 * sample_rate)
            center_freq = center_bin * sample_rate / n_fft
            nyquist = sample_rate / 2.0
            freq_pos_norm = center_freq / nyquist if nyquist > 0 else 0.0
            all_freq_pos_norm.append(freq_pos_norm)
        
        band_spectrograms = torch.stack(band_spectrograms, dim=0)  # (num_bands, band_width, T)
        band_patch_indices = torch.stack(band_patch_indices, dim=0)  # (num_bands, 2)
        freq_pos_norm = torch.tensor(all_freq_pos_norm, dtype=torch.float32, device=spectrogram.device).unsqueeze(1)  # (num_bands, 1)
        freq_pos_emb = get_sincos_encoding_1d(freq_pos_norm, self.freq_pos_emb_dim)  # (num_bands, freq_pos_emb_dim)
        
        return freq_pos_emb, band_spectrograms, band_patch_indices


    def _compute_frequency_position_encoding_batch(self, spectrogram, sample_rate, random_select_bands=False):
        """
        Band splitting and frequency position encoding for list of spectrograms.
        
        Args:
            spectrogram: List of full spectrograms, each with shape (freq_bins, time)
            sample_rate: Audio sample rate (B,) or scalar
            
        Returns:
            freq_pos_emb: Frequency position embeddings for all bands from all samples, (total_bands, freq_pos_emb_dim)
            band_spectrograms: Concatenated band spectrograms from all samples, (total_bands, band_width, T)
            band_patch_indices: Concatenated band patch indices from all samples, (total_bands, 2)
        """
        # Handle sample_rate
        if isinstance(sample_rate, (int, float)) or (hasattr(sample_rate, 'dim') and sample_rate.dim() == 0):
            # scalar -> (B,)
            sample_rate = [sample_rate] * len(spectrogram)
        
        all_band_spectrograms = []
        all_band_patch_indices = []
        all_freq_pos_emb = []
        
        for spec, sr in zip(spectrogram, sample_rate):
            # Process each spectrogram using the single spectrogram function
            if random_select_bands:
                freq_pos_emb, band_spectrograms, band_patch_indices = self._compute_frequency_position_encoding_random(spec, sr)
            else:
                freq_pos_emb, band_spectrograms, band_patch_indices = self._compute_frequency_position_encoding(spec, sr)
            
            # Collect results
            all_band_spectrograms.append(band_spectrograms)
            all_band_patch_indices.append(band_patch_indices)
            all_freq_pos_emb.append(freq_pos_emb)
        
        # Concatenate all results
        if all_band_spectrograms:
            band_spectrograms = torch.cat(all_band_spectrograms, dim=0)  # (total_bands, band_width, T)
            band_patch_indices = torch.cat(all_band_patch_indices, dim=0)  # (total_bands, 2)
            freq_pos_emb = torch.cat(all_freq_pos_emb, dim=0)  # (total_bands, freq_pos_emb_dim)
        else:
            raise ValueError("No bands found in spectrogram")
        
        return freq_pos_emb, band_spectrograms, band_patch_indices

    def extract_features(self, spectrogram, sample_rate):
        """
        Extract features from full spectrogram with band splitting and frequency position encoding.
        
        Args:
            spectrogram: Full spectrogram tensor (freq_bins, time)
            sample_rate: Audio sample rate
            
        Returns:
            features: Concatenated CLS tokens from all bands
        """
        # Compute frequency position encoding and band splitting
        freq_pos_emb, band_spectrograms, band_patch_indices = self._compute_frequency_position_encoding(
            spectrogram, sample_rate
        )
        
        # Add channel dimension for model input
        band_spectrograms = band_spectrograms.unsqueeze(1)  # (num_bands, 1, freq, time)
        
        # Create freq_lengths for all bands
        freq_lengths = torch.tensor([spectrogram.shape[-1]] * band_spectrograms.shape[0], device=spectrogram.device)
        
        # Create band_batch_indices (all bands from same audio)
        band_batch_indices = torch.zeros(band_spectrograms.shape[0], dtype=torch.long, device=spectrogram.device)
        
        # Forward pass with no masking for feature extraction
        tokens, mask, ids_restore, band_patch_indices, patches, ids_keep, freq_pos_emb, band_spectrograms = self.forward(
            spectrogram, 
            sample_rate, 
            freq_pos_emb=freq_pos_emb,
            band_spectrograms=band_spectrograms,
            band_patch_indices=band_patch_indices,
            mask_ratio=0.0
        )
        
        # Extract CLS tokens from each band and concatenate
        cls_tokens = tokens[:, 0, :]  # (num_bands, embed_dim)
        utt_feature = cls_tokens.flatten()  # (num_bands * embed_dim,)

        frame_feature = tokens[:, 1:, :]  # (num_bands, num_patches, embed_dim)
        frame_feature = rearrange(frame_feature, 'b t d -> t (b d)')  # (num_patches, num_bands * embed_dim)
        
        return utt_feature, frame_feature

    def forward(self, x, sample_rate, mask_ratio=None, freq_pos_emb=None, band_spectrograms=None, band_patch_indices=None):
        """
        Unified forward method for both student and teacher modes.
        
        Args:
            x: Input spectrograms, a list of spectrograms
            sample_rate: Audio sample rate, a list of sample rates
            mask_ratio: Masking ratio (None for default, 0 for teacher, >0 for student)
        """
        # Use default mask_ratio if not provided
        if mask_ratio is None:
            mask_ratio = self.mask_ratio
        
        if freq_pos_emb is None or band_spectrograms is None or band_patch_indices is None:
            freq_pos_emb, band_spectrograms, band_patch_indices = self._compute_frequency_position_encoding_batch(x, sample_rate, random_select_bands=True)
        
        # Patch embedding
        band_spectrograms = band_spectrograms.to(self.pos_embed.device)
        patches = self.patchify(band_spectrograms)  # (total_bands, num_patches, embed_dim)
        
        # Add patch positional encoding
        patches = patches + self.pos_embed[:, 1:patches.shape[1]+1, :]  # Skip CLS token position

        # Add frequency positional encoding if provided, including the cls token
        if freq_pos_emb is not None:
            freq_pos_emb = freq_pos_emb.to(self.pos_embed.device)
            freq_pos_emb_expanded = freq_pos_emb.unsqueeze(1).expand(-1, patches.shape[1], -1)
            patches = patches + freq_pos_emb_expanded

        # Apply masking (random_masking handles mask_ratio=0 correctly)
        patches_unmasked, mask, ids_restore, ids_keep = self.random_masking(patches, mask_ratio)
        
        # Add CLS token
        N, L_unmasked, D = patches_unmasked.shape
        cls_token = self.cls_token.expand(N, -1, -1) # (N, 1, D)
        # Add frequency position encoding to the cls token
        if freq_pos_emb is not None:
            cls_freq_pos_emb = freq_pos_emb.unsqueeze(1)  # (total_bands, 1, freq_pos_emb_dim)
            cls_token = cls_token + cls_freq_pos_emb
        tokens = torch.cat((cls_token, patches_unmasked), dim=1) # (N, L_unmasked+1, D)

        # Pass through transformer encoder
        tokens = self.encoder(tokens)
        tokens = self.norm(tokens) # normalize to suit downstream tasks
        
        return tokens, mask, ids_restore, band_patch_indices, patches, ids_keep, freq_pos_emb, band_spectrograms


    def preprocess_audio_to_spectrogram(self, audio_signal, sample_rate, max_length=2000):
        """
        Convert audio to spectrogram for ECHO model input.
        
        Args:
            audio_signal: Audio tensor of shape [channels, samples]
            sample_rate: Audio sample rate
            max_length: Maximum time length for spectrogram
            
        Returns:
            spec: Preprocessed spectrogram tensor
        """
        import torch
        import torchaudio
        
        # Remove DC component
        waveform = audio_signal - audio_signal.mean()
        
        # Convert to spectrogram
        window_size = int(0.025 * sample_rate)  # 25ms window
        hop_size = int(0.01 * sample_rate)      # 10ms hop
        
        stft = torchaudio.transforms.Spectrogram(
            n_fft=window_size,
            hop_length=hop_size,
            power=1, 
            center=False
        )
        
        spec = stft(waveform.squeeze(0))
        spec = torch.log(spec + 1e-9)
        
        # Normalize (using ECHO-specific values)
        norm_mean = -5.874158
        norm_std = 5.223174
        spec = (spec - norm_mean) / (norm_std * 2)
        
        return spec

    def extract_features_from_audio(self, audio_signal, sample_rate=16000):
        """
        Extract features directly from audio signal using ECHO model.
        
        Args:
            audio_signal: Audio tensor of shape [channels, samples]
            sample_rate: Audio sample rate
            
        Returns:
            features: Extracted features tensor
        """
        import torch
        
        # Preprocess audio
        spec = self.preprocess_audio_to_spectrogram(audio_signal, sample_rate)
        
        # Process in segments if needed
        max_length = 2000
        input_specs = []
        num_segments = spec.shape[-1] // max_length
        
        for i in range(num_segments):
            segment = spec[..., i * max_length:(i + 1) * max_length]
            input_specs.append(segment)
        
        # Handle remaining part
        if num_segments * max_length < spec.shape[-1]:
            remaining = spec[..., -max_length:]
            input_specs.append(remaining)
        
        # Extract features
        utt_features = []
        frame_features = []
        with torch.no_grad():
            for segment_spec in input_specs:
                if hasattr(self, 'extract_features'):
                    feature = self.extract_features(segment_spec, sample_rate)
                else:
                    # Use forward method if extract_features not available
                    segment_input = segment_spec.unsqueeze(0).unsqueeze(0)
                    feature = self.forward(segment_input, sample_rate)
                utt_features.append(feature[0].cpu())
                frame_features.append(feature[1].cpu())
        
        # Aggregate features (mean pooling)
        utt_features = torch.stack(utt_features, dim=0).mean(dim=0)
        frame_features = torch.vstack(frame_features)
        return utt_features, frame_features


