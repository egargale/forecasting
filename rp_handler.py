import os
import sys
import runpod
import torch
from tirex import load_model as load_tirex_model
from chronos import BaseChronosPipeline

# Auto-detect CPU vs CUDA usage
USE_CPU = not torch.cuda.is_available() or os.environ.get('USE_CPU', 'false').lower() in ('true', '1', 'yes')

# Configure environment for CPU if needed
if USE_CPU:
    os.environ['TIREX_NO_CUDA'] = '1'
    print("CPU mode detected - configuring for CPU-only usage")
    device_str = "cpu"
    torch_dtype = torch.float32
    device_map = "cpu"
else:
    print("CUDA mode detected - using GPU acceleration")
    device_str = "cuda:0"
    torch_dtype = torch.bfloat16
    device_map = "cuda"

# Load models once when the worker starts
print("Loading models...")

# Add diagnostic logging for CUDA compilation
if not USE_CPU:
    print("CUDA compilation diagnostics:")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA version: {torch.version.cuda}")
    print(f"PyTorch version: {torch.__version__}")
    # Check if TORCH_CUDA_ARCH_LIST is set
    cuda_arch_list = os.environ.get('TORCH_CUDA_ARCH_LIST', 'not set')
    print(f"TORCH_CUDA_ARCH_LIST: {cuda_arch_list}")
    
    # Set compilation flags to fix template instantiation issues
    # This addresses the -static-global-template-stub=true warning
    os.environ['XLSTM_EXTRA_CUDA_CFLAGS'] = '-static-global-template-stub=false'
    print("Set XLSTM_EXTRA_CUDA_CFLAGS to fix template instantiation")

try:
    tirex_model = load_tirex_model("NX-AI/TiRex", device=device_str)
    print("TiRex model loaded successfully")
except Exception as e:
    print(f"Error loading TiRex model: {str(e)}")
    print("This may be due to CUDA compilation issues with sLSTM extension")
    # Fall back to CPU if CUDA compilation fails
    if not USE_CPU:
        print("Attempting fallback to CPU mode...")
        os.environ['TIREX_NO_CUDA'] = '1'
        tirex_model = load_tirex_model("NX-AI/TiRex", device="cpu")
    else:
        raise

try:
    chronos_pipeline = BaseChronosPipeline.from_pretrained(
        "amazon/chronos-bolt-base",
        device_map=device_map,
        torch_dtype=torch_dtype,
    )
    print("Chronos model loaded successfully")
except Exception as e:
    print(f"Error loading Chronos model: {str(e)}")
    raise

print("Models loaded successfully")

def handler(event):
    """
    This function processes incoming requests to your Serverless endpoint.
    
    Args:
        event (dict): Contains the input data and request metadata
    
    Returns:
        dict: The forecast result to be returned to the client
    """
    print("Worker Start")
    
    # Extract input data
    input_data = event['input']
    
    # Get model type and parameters
    model_type = input_data.get('model', 'tirex')  # Default to tirex
    context = input_data.get('context')
    prediction_length = input_data.get('prediction_length')
    
    if context is None or prediction_length is None:
        return {
            "error": "Missing required parameters: 'context' and 'prediction_length' are required"
        }
    
    print(f"Model type: {model_type}")
    print(f"Received context: {context}")
    print(f"Prediction length: {prediction_length}")
    
    try:
        # Convert context to tensor with proper dtype and shape
        tensor_data = torch.tensor(context, dtype=torch.float32)
        
        if model_type.lower() == 'tirex':
            # Generate forecast using TiRex model
            quantiles, mean = tirex_model.forecast(context=tensor_data, prediction_length=prediction_length)
            
            # Return both quantiles and mean forecasts as lists
            return {
                "model": "tirex",
                "forecast": mean.tolist(),
                "quantiles": quantiles.tolist()
            }
        
        elif model_type.lower() == 'chronos':
            # Ensure tensor is 1D for Chronos
            if tensor_data.dim() == 1:
                tensor_data = tensor_data.unsqueeze(0)  # Add batch dimension
            
            # Generate forecast using Chronos model with predict_quantiles
            quantiles, mean = chronos_pipeline.predict_quantiles(
                context=tensor_data,
                prediction_length=prediction_length,
                quantile_levels=[0.1, 0.5, 0.9],
            )
            
            return {
                "model": "chronos",
                "forecast": mean.squeeze().tolist(),
                "quantiles": quantiles.squeeze().tolist()
            }
        
        else:
            return {
                "error": f"Invalid model type: '{model_type}'. Supported models are 'tirex' and 'chronos'"
            }
    
    except Exception as e:
        print(f"Error during forecasting: {str(e)}")
        return {
            "error": f"Forecasting failed: {str(e)}"
        }

# Start the Serverless function when the script is run
if __name__ == '__main__':
    runpod.serverless.start({'handler': handler})