import torch
import sys

def get_device():
    """Get the best available device (CUDA GPU or CPU)"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def test_cuda():
    try:
        device = get_device()
        print(f"Python version: {sys.version}")
        print(f"PyTorch version: {torch.__version__}")
        print(f"Using device: {device}")
        
        if device.type == "cuda":
            print(f"CUDA version: {torch.version.cuda}")
            print(f"cuDNN version: {torch.backends.cudnn.version()}")
            print(f"Device name: {torch.cuda.get_device_name(0)}")
            
            # Get device properties
            device_props = torch.cuda.get_device_properties(0)
            print("\nDevice Properties:")
            print(f"  Name: {device_props.name}")
            print(f"  Compute Capability: {device_props.major}.{device_props.minor}")
            print(f"  Total Memory: {device_props.total_memory / 1024**2:.1f} MB")
            print(f"  Multi Processor Count: {device_props.multi_processor_count}")
        
        # Test tensor operations with error handling
        print("\nTesting tensor operations...")
        try:
            # Create tensors on CPU first
            x = torch.rand(5, 3)
            y = torch.rand(5, 3)
            
            # Move to target device (CPU or GPU)
            x = x.to(device)
            y = y.to(device)
            
            # Example of loading and moving a model
            print("\nExample of loading a model:")
            print("# Load model:")
            print("model = YourModel()")
            print("model.load_state_dict(torch.load('model.pth', map_location=device))")
            print("model = model.to(device)")
            
            # Perform operations
            z = x + y
            print("\n✓ Successfully created tensors")
            print(f"  Tensor device: {z.device}")
            print(f"  Tensor shape: {z.shape}")
            
            result = torch.matmul(x, y.t())
            print("✓ Successfully performed matrix multiplication")
            print(f"  Result shape: {result.shape}")
            
            print("\nNotes for model portability:")
            print("1. When saving model:")
            print("   torch.save(model.state_dict(), 'model.pth')")
            print("2. When loading on any device:")
            print("   model.load_state_dict(torch.load('model.pth', map_location=device))")
            
        except RuntimeError as e:
            print(f"❌ Runtime error during tensor operations: {e}")
        except Exception as e:
            print(f"❌ Unexpected error during tensor operations: {e}")
            
    except Exception as e:
        print(f"❌ Error during device test: {e}")

if __name__ == "__main__":
    test_cuda() 