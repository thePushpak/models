import torch

# Check if CUDA is available
if torch.cuda.is_available():
    # Create a tensor on GPU
    x = torch.tensor([1.0, 2.0]).cuda()

    # Print device information
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: True")
    print(f"Current device: {torch.cuda.get_device_name(0)}")
    print(f"Tensor on device: {x}")
else:
    print("PyTorch is installed, but CUDA is not available. GPU acceleration is not supported.")
