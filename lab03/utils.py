import torch
import time
import numpy as np
from torchinfo import summary
from thop import profile
from models import MLP, CNN, AttentionModel

def measure_latency(model, batch_size, runs):
    model.eval()
    dummy_input = torch.randn(batch_size, 1, 28, 28)
    
    with torch.no_grad():
        for _ in range(10):
            _ = model(dummy_input)
            
    latencies = []
    with torch.no_grad():
        for _ in range(runs):
            start = time.perf_counter()
            _ = model(dummy_input)
            end = time.perf_counter()
            latencies.append((end - start) * 1000) # Convert to ms
            
    return np.median(latencies)

def profile_model(model_class, name):
    model = model_class()
    print(f"\n{'='*40}")
    print(f"--- Profiling {name} ---")
    print(f"{'='*40}")
    
    summary(model, input_size=(128, 1, 28, 28), verbose=1)
    
    dummy_input = torch.randn(1, 1, 28, 28)
    macs, params = profile(model, inputs=(dummy_input,), verbose=False)
    flops = macs * 2
    print(f"\nTHOP Automated FLOPs (Batch 1): {flops:,.0f}")
    
    lat_b1 = measure_latency(model, batch_size=1, runs=100)
    lat_b128 = measure_latency(model, batch_size=128, runs=20)
    print(f"Latency (Batch 1, median of 100): {lat_b1:.3f} ms")
    print(f"Latency (Batch 128, median of 20): {lat_b128:.3f} ms\n")

if __name__ == "__main__":
    profile_model(MLP, "MLP")
    profile_model(CNN, "CNN")
    profile_model(AttentionModel, "Attention Model")