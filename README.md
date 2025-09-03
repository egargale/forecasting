# Forecasting Service

A production-ready forecasting service that provides time series predictions using state-of-the-art models including TiRex and Amazon Chronos. This service is designed for deployment as a serverless endpoint on RunPod.

## Overview

This project implements a forecasting API that supports multiple time series forecasting models:
- **TiRex**: A transformer-based forecasting model from NX-AI
- **Chronos**: Amazon's pretrained time series forecasting models

The service is optimized for serverless deployment and provides both point forecasts and uncertainty quantification through quantile predictions.

## Features

- **Multi-Model Support**: Choose between TiRex and Chronos models
- **Quantile Forecasting**: Get uncertainty estimates with 10th, 50th, and 90th percentiles
- **Serverless Ready**: Optimized for RunPod serverless deployment
- **GPU Accelerated**: Leverages CUDA for fast inference
- **CPU Fallback**: Automatic CPU detection and fallback when CUDA unavailable
- **Production Ready**: Error handling, logging, and input validation
- **Auto-Configuration**: Automatic CPU/CUDA detection with environment variable overrides

## Quick Start

### Local Development

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Test CPU/CUDA Configuration**
   ```bash
   # Check automatic CPU/CUDA detection
   python test_cpu_config.py
   ```

3. **Run Local Tests**
   ```bash
   # Test with TiRex model
   python -c "import json; f=open('test_input.json'); print(json.load(f))"
   
   # Test with Chronos model
   python -c "import json; f=open('test_input_chronos.json'); print(json.load(f))"
   ```

4. **Start Serverless Worker**
   ```bash
   python rp_handler.py
   ```

### API Usage

The service accepts POST requests with the following JSON structure:

```json
{
    "input": {
        "model": "tirex",
        "context": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        "prediction_length": 5
    }
}
```

**Parameters:**
- `model`: Choose between `"tirex"` or `"chronos"` (default: `"tirex"`)
- `context`: Array of historical time series values
- `prediction_length`: Number of future points to predict

**Response Format:**
```json
{
    "model": "tirex",
    "forecast": [11.2, 12.3, 13.1, 14.5, 15.2],
    "quantiles": [[10.1, 11.2, 12.3], [11.5, 12.3, 13.8], ...]
}
```

## CPU/CUDA Configuration

The service automatically detects CPU/CUDA availability and configures models accordingly:

### Automatic Detection
- **CUDA Available**: Uses GPU acceleration with `torch_dtype=torch.bfloat16`
- **CUDA Unavailable**: Falls back to CPU with `torch_dtype=torch.float32`

### Environment Variables
Override automatic detection with these environment variables:

| Variable | Values | Description |
|----------|--------|-------------|
| `USE_CPU` | `true`, `1`, `yes` | Force CPU mode regardless of CUDA availability |
| `TIREX_NO_CUDA` | `1` | Disable TiRex CUDA kernels (set automatically in CPU mode) |

### Examples
```bash
# Force CPU mode
export USE_CPU=true
python rp_handler.py

# Use CUDA (default when available)
unset USE_CPU
python rp_handler.py
```

## Deployment

### RunPod Serverless Deployment

1. **Build Docker Image**
   ```bash
   docker build -t forecasting-service .
   ```

2. **Deploy to RunPod**
   - Upload your Docker image to a container registry
   - Create a new serverless endpoint on RunPod
   - Configure GPU requirements (minimum: 1x A100 or T4)

3. **Environment Variables**
   - No additional environment variables required
   - Models are downloaded automatically on startup

### Local Testing with RunPod

Test your handler locally before deployment:

```bash
# Install runpod package
pip install runpod

# Test the handler
python rp_handler.py
```

## Models

### TiRex (NX-AI/TiRex)
- **Type**: Transformer-based forecasting model
- **Strengths**: Fast inference, good for short to medium-term forecasts
- **Model Size**: ~100M parameters
- **Use Case**: General-purpose forecasting with uncertainty

### Chronos-Bolt-Small (amazon/chronos-bolt-small)
- **Type**: Pretrained time series language model
- **Strengths**: Zero-shot forecasting, handles multiple frequencies
- **Model Size**: ~50M parameters
- **Use Case**: Cross-domain forecasting without retraining

## Project Structure

```
forecasting/
├── main.py                 # Simple entry point for local testing
├── rp_handler.py          # RunPod serverless handler
├── test_cpu_config.py     # CPU/CUDA configuration test script
├── requirements.txt       # Python dependencies
├── pyproject.toml        # Project configuration
├── test_input.json       # Sample TiRex input
├── test_input_chronos.json  # Sample Chronos input
├── test_input_tirex.json  # Sample TiRex input (alias)
├── chronos-forecasting/  # Chronos model submodule
└── tirex/               # TiRex model submodule
```

## Development

### Setup Development Environment

1. **Clone with Submodules**
   ```bash
   git clone --recurse-submodules <repository-url>
   cd forecasting
   ```

2. **Install in Development Mode**
   ```bash
   pip install -e .
   ```

3. **Update Submodules**
   ```bash
   git submodule