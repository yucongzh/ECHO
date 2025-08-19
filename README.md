# ECHO

**Frequency-aware Hierarchical Encoding for Variable-length Signal**

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Hugging Face](https://img.shields.io/badge/HuggingFace-ECHO-orange)](https://huggingface.co/yucongzh/ECHO/tree/main)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.16900172.svg)](https://doi.org/10.5281/zenodo.16900172)

ECHO is a state-of-the-art audio representation learning model based on Masked Autoencoders (MAE) with band-splitting and frequency position encoding.

## Features

- **Band-splitting architecture**: Processes audio in frequency bands for better local and global representation learning
- **Frequency position encoding**: Incorporates frequency information into the model for better audio understanding
- **Efficient patch embedding**: Uses sliding window patches for temporal modeling, enabling varying time lengths

## Installation

```bash
pip install -r requirements.txt
```

## Pre-trained Models

| Model | Embed Dim | Layers | Heads | Parameters | Download |
|-------|-----------|--------|-------|------------|----------|
| ECHO-Small | 384 | 12 | 6 | ~21.5M | [Hugging Face](https://huggingface.co/yucongzh/ECHO/tree/main) · [Zenodo](https://zenodo.org/records/16900172) |
| ECHO-Base | - | - | - | - | Coming Soon |
| ECHO-Mini | - | - | - | - | Coming Soon |
| ECHO-Tiny | - | - | - | - | Coming Soon |

## Model Architecture

The ECHO model consists of:
- **Patch Embedding**: Converts spectrogram bands to patch embeddings
- **Transformer Encoder**: Multi-layer transformer for feature learning
- **Frequency Position Encoding**: Adds frequency-aware positional information
- **Band Processing**: Splits audio into frequency bands for hierarchical learning

## Usage

### Quick Start with Pre-trained Models

1. **Download the model** from [Hugging Face](https://huggingface.co/yucongzh/ECHO/tree/main)
2. **Load and use** the model:

```python
from models.ECHO import AudioMAEWithBand

# Load pre-trained ECHO-Small model
model = AudioMAEWithBand.from_pretrained("path_to_echo_small_checkpoint.pth")

# Extract features from audio
features = model.extract_features(spectrogram, sample_rate=16000)
print(f"Feature dimension: {features.shape}")
```

#### Download via Hugging Face Hub API

You can programmatically download checkpoints using `huggingface_hub`:

```bash
pip install huggingface_hub
```

```python
import os
from huggingface_hub import hf_hub_download, snapshot_download
from models.ECHO import AudioMAEWithBand

# Option A: Download a single file
ckpt_path = hf_hub_download(
    repo_id="yucongzh/ECHO",            # repository
    filename="ECHO-small.pth"  # file inside repo
)
model = AudioMAEWithBand.from_pretrained(ckpt_path)

# Option B: Download entire repo snapshot to a local directory
local_dir = snapshot_download(repo_id="yucongzh/ECHO")
ckpt_path = os.path.join(local_dir, "ECHO-small.pth")
model = AudioMAEWithBand.from_pretrained(ckpt_path)
```

### Basic Feature Extraction

#### Option 1: Load Pre-trained Model (Recommended)

```python
import torch
import torchaudio
from models.ECHO import AudioMAEWithBand

# Load pre-trained model from Hugging Face
# First download the model from: https://huggingface.co/yucongzh/ECHO/tree/main
model = AudioMAEWithBand.from_pretrained("path_to_downloaded_checkpoint.pth")

# The model will automatically detect the model type and configure parameters
# For ECHO-Small: embed_dim=384, encoder_depth=12, num_heads=6
```

#### Option 2: Manual Model Initialization

```python
import torch
import torchaudio
from models.ECHO import AudioMAEWithBand

# Initialize model manually
model = AudioMAEWithBand(
    spec_len=2000,
    band_width=32,
    shift_size=16,
    embed_dim=384,
    encoder_depth=12,
    num_heads=6,
    mlp_ratio=4.0,
    freq_pos_emb_dim=384
)

# Load checkpoint manually (if needed)
# checkpoint = torch.load('path_to_checkpoint.pth')
# model.load_state_dict(checkpoint['encoder'])

model.eval()

# Prepare audio input
def extract_features(audio_path, sample_rate=16000):
    # Load audio
    waveform, sr = torchaudio.load(audio_path)
    if sr != sample_rate:
        waveform = torchaudio.transforms.Resample(sr, sample_rate)(waveform)
    
    # Convert to mono if stereo
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    
    # Remove DC component
    waveform = waveform - waveform.mean()
    
    # Convert to spectrogram
    window_size = int(0.025 * sample_rate)  # 25ms
    hop_size = int(0.01 * sample_rate)      # 10ms
    
    stft = torchaudio.transforms.Spectrogram(
        n_fft=window_size,
        hop_length=hop_size,
        power=1,
        center=False
    )
    
    spec = stft(waveform.squeeze(0))
    spec = torch.log(spec + 1e-9)
    
    # Normalize (use your model's normalization values)
    norm_mean = -5.874158
    norm_std = 5.223174
    spec = (spec - norm_mean) / (norm_std * 2)
    
    # Extract features
    with torch.no_grad():
        features = model.extract_features(spec, sample_rate)
    
    return features

# Example usage
if __name__ == "__main__":
    # Extract features from audio file
    features = extract_features("path_to_audio.wav")
    print(f"Feature dimension: {features.shape}")
    print(f"Feature values range: [{features.min():.4f}, {features.max():.4f}]")
```

## Citation
coming soon
<!-- ```bibtex
@article{echo2024,
  title={ECHO: Enhanced Contextual Hierarchical Output for Audio Representation Learning},
  author={Your Name},
  journal={arXiv preprint},
  year={2024}
}
``` -->

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.


## Model Download Instructions

### From Hugging Face Hub

1. **Visit**: [https://huggingface.co/yucongzh/ECHO/tree/main](https://huggingface.co/yucongzh/ECHO/tree/main)
2. **Download**: Click on the checkpoint file (e.g., `ECHO-small.pth`)
3. **Use**: Load with `AudioMAEWithBand.from_pretrained("path_to_checkpoint.pth")`

### From Zenodo

1. **Visit**: [https://zenodo.org/records/16900172](https://zenodo.org/records/16900172)
2. **DOI**: [10.5281/zenodo.16900172](https://doi.org/10.5281/zenodo.16900172)  
   [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.16900172.svg)](https://doi.org/10.5281/zenodo.16900172)
3. **Access**: The record is public; files may be restricted to users with access
4. **Use**: Download the checkpoint and load via `AudioMAEWithBand.from_pretrained("path_to_checkpoint.pth")`


## Acknowledgments

We gratefully acknowledge the following open-source projects for inspiration and foundations:

- **AudioMAE**: Masked Autoencoders that Listen. [GitHub](https://github.com/facebookresearch/AudioMAE/tree/bd60e29651285f80d32a6405082835ad26e6f19f)
- **FISHER**: A foundation model for industrial signal comprehensive representation. [GitHub](https://github.com/jianganbai/FISHER)
- **EAT**: Self-Supervised Pre-Training with Efficient Audio Transformer. [GitHub](https://github.com/cwx-worst-one/EAT)

