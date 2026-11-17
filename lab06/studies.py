import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.profiler import profile, record_function, ProfilerActivity
import time
import pandas as pd
from pipeline import FashionCNN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
trainset = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)

def run_profiler():
    print("--- Task 2: PyTorch Profiling ---")
    # Using num_workers=0 to prevent your system's multiprocessing crash
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True, num_workers=0)
    model = FashionCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    with profile(activities=[ProfilerActivity.CPU], record_shapes=True, profile_memory=True) as prof:
        for i, (inputs, labels) in enumerate(trainloader):
            if i >= 5: break # Profile just a few batches of one epoch
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            with record_function("forward_pass"):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            with record_function("backward_pass"):
                loss.backward()
            with record_function("optimizer_step"):
                optimizer.step()
                
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))

def study_num_workers():
    print("\n--- Task 3: num_workers Study ---")
    workers_list = [0, 1, 2, 4]
    results = []
    for w in workers_list:
        try:
            loader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True, num_workers=w)
            start = time.perf_counter()
            for i, _ in enumerate(loader):
                if i >= 50: break # subset for speed
            duration = time.perf_counter() - start
            results.append({'num_workers': w, 'time_50_batches': duration})
            print(f"Workers: {w} | Time: {duration:.4f}s")
        except Exception as e:
            print(f"Workers: {w} | Failed due to multiprocessing crash (expected on this OS)")
    
    if results:
        print(pd.DataFrame(results))

def study_batch_size():
    print("\n--- Task 4: Batch Size Study ---")
    batch_sizes = [32, 64, 128, 256, 512]
    for bs in batch_sizes:
        # Using num_workers=0 here as well
        loader = torch.utils.data.DataLoader(trainset, batch_size=bs, shuffle=True, num_workers=0)
        start = time.perf_counter()
        count = 0
        for i, (inputs, _) in enumerate(loader):
            count += len(inputs)
            if i >= 20: break
        duration = time.perf_counter() - start
        throughput = count / duration
        print(f"Batch Size: {bs} | Throughput: {throughput:.2f} samples/s")

if __name__ == "__main__":
    run_profiler()
    study_num_workers()
    study_batch_size()
    print("\nNote for Task 5: Optimizer (SGD+Momentum vs Adam) and Schedulers are configurable in config.yaml for independent MLflow logging.")