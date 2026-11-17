import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import os
import time

# 1. The Oversized Baseline Model (~1M parameters)
class OversizedCNN(nn.Module):
    def __init__(self):
        super(OversizedCNN, self).__init__()
        # 1 input channel (grayscale), 64 output channels
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        # 128 channels * 7 * 7 spatial size = 6272
        # 6272 * 150 approx = 940k parameters (hitting that 1M target)
        self.classifier = nn.Sequential(
            nn.Linear(128 * 7 * 7, 150),
            nn.ReLU(),
            nn.Linear(150, 10)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # num_workers=0 to prevent the forkserver crash from Lab 6
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    trainset = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True, num_workers=0)
    
    testset = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False, num_workers=0)

    model = OversizedCNN().to(device)
    print(f"Total Parameters: {count_parameters(model):,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print("Training the oversized baseline...")
    epochs = 3 # Enough to get decent accuracy for the teacher
    for epoch in range(epochs):
        model.train()
        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}/{epochs} complete.")

    # Measure Accuracy
    model.eval()
    correct = 0
    total = 0
    start_time = time.perf_counter()
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    end_time = time.perf_counter()
    accuracy = 100 * correct / total
    latency = (end_time - start_time) / len(testloader) * 1000 # ms per batch
    
    print(f"Baseline Accuracy: {accuracy:.2f}%")
    print(f"Inference Latency: {latency:.2f} ms / batch")

    # Save model and check file size
    os.makedirs('checkpoints', exist_ok=True)
    save_path = 'checkpoints/teacher_model.pth'
    torch.save(model.state_dict(), save_path)
    file_size = os.path.getsize(save_path) / (1024 * 1024)
    print(f"Model File Size: {file_size:.2f} MB")

if __name__ == "__main__":
    main()