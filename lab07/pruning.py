import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import torchvision
import torchvision.transforms as transforms
import copy
import time
import os
import matplotlib.pyplot as plt
from baseline import OversizedCNN  # Import your teacher model

def get_dataloader():
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    trainset = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True, num_workers=0)
    testset = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False, num_workers=0)
    return trainloader, testloader

def evaluate(model, testloader, device):
    model.eval()
    correct = 0
    total = 0
    start = time.perf_counter()
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    latency = (time.perf_counter() - start) / len(testloader) * 1000
    return 100 * correct / total, latency

def finetune(model, trainloader, device, epochs=1):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    model.train()
    for epoch in range(epochs):
        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainloader, testloader = get_dataloader()
    
    # Load original Teacher model
    original_model = OversizedCNN().to(device)
    # weights_only=True for security best practices when loading state dicts
    original_model.load_state_dict(torch.load('checkpoints/teacher_model.pth', weights_only=True)) 
    
    sparsities = [0.0, 0.25, 0.5, 0.75, 0.9, 0.95]
    global_accs = []
    
    print("--- Task 2: Global Unstructured Pruning ---")
    for amount in sparsities:
        model = copy.deepcopy(original_model)
        parameters_to_prune = (
            (model.features[0], 'weight'),
            (model.features[3], 'weight'),
            (model.classifier[0], 'weight'),
            (model.classifier[2], 'weight'),
        )
        if amount > 0:
            prune.global_unstructured(parameters_to_prune, pruning_method=prune.L1Unstructured, amount=amount)
            print(f"Fine-tuning Global Sparsity {amount*100}%...")
            finetune(model, trainloader, device, epochs=1) 
        
        acc, lat = evaluate(model, testloader, device)
        global_accs.append(acc)
        print(f"Sparsity {amount*100}% -> Acc: {acc:.2f}%, Latency: {lat:.2f} ms/batch")
        
        # Save the 95% pruned model for Task 3 comparison
        if amount == 0.95:
            torch.save(model.state_dict(), 'checkpoints/pruned_dense.pth')
            # Create a true sparse dictionary
            sparse_state = {k: v.to_sparse() if 'weight' in k else v for k, v in model.state_dict().items()}
            torch.save(sparse_state, 'checkpoints/pruned_sparse.pth')

    print("\n--- Task 2: Structured Pruning (Filters) ---")
    struct_sparsities = [0.10, 0.25, 0.50]
    struct_accs = []
    
    for amount in struct_sparsities:
        model = copy.deepcopy(original_model)
        # Prune entire filters based on L2 norm
        prune.ln_structured(model.features[0], name="weight", amount=amount, n=2, dim=0)
        prune.ln_structured(model.features[3], name="weight", amount=amount, n=2, dim=0)
        
        print(f"Fine-tuning Structured Sparsity {amount*100}%...")
        finetune(model, trainloader, device, epochs=1)
        acc, lat = evaluate(model, testloader, device)
        struct_accs.append(acc)
        print(f"Structured Sparsity {amount*100}% -> Acc: {acc:.2f}%, Latency: {lat:.2f} ms/batch")

    print(f"\n--- Task 3: Storage & Latency Analysis ---")
    dense_size = os.path.getsize('checkpoints/pruned_dense.pth') / (1024 * 1024)
    sparse_size = os.path.getsize('checkpoints/pruned_sparse.pth') / (1024 * 1024)
    print(f"Dense Model File Size: {dense_size:.2f} MB")
    print(f"Sparse Format File Size: {sparse_size:.2f} MB")
    print("\nHardware Reality (Why latency didn't drop):")
    print("Unstructured pruning leaves random zeros in the weight matrices. Standard CPUs process data in dense, fixed-size blocks (SIMD). Multiplying by zero still takes the exact same number of CPU cycles as multiplying by 5 unless you use specialized sparse hardware kernels.")

    # Generate Graded Evidence Plot
    plt.plot([s*100 for s in sparsities], global_accs, marker='o', label='Global Unstructured')
    plt.plot([s*100 for s in struct_sparsities], struct_accs, marker='s', label='Structured (Filters)')
    plt.xlabel('Sparsity (%)')
    plt.ylabel('Accuracy (%)')
    plt.title('Lab 7: Model Sparsity vs Accuracy')
    plt.legend()
    plt.grid()
    plt.savefig('pruning_curves.png')
    print("\nSaved sparsity curve plot to 'pruning_curves.png'")

if __name__ == "__main__":
    main()