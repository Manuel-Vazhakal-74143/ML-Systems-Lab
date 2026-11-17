import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import time
import os
from baseline import OversizedCNN, count_parameters

# Task 4: The Compact Student Model (~100k parameters)
class StudentCNN(nn.Module):
    def __init__(self):
        super(StudentCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        # 32 channels * 7 * 7 = 1568
        self.classifier = nn.Sequential(
            nn.Linear(1568, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

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

def train_plain_student(student, trainloader, device, epochs=3):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(student.parameters(), lr=0.001)
    student.train()
    print("Training Plain Student (No Teacher)...")
    for epoch in range(epochs):
        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = student(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}/{epochs} complete.")

def train_distilled_student(student, teacher, trainloader, device, epochs=3, T=4.0, alpha=0.5):
    optimizer = torch.optim.Adam(student.parameters(), lr=0.001)
    student.train()
    teacher.eval()
    print("\nTraining Distilled Student (With Teacher)...")
    for epoch in range(epochs):
        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            
            student_logits = student(inputs)
            with torch.no_grad():
                teacher_logits = teacher(inputs)
                
            # Distillation Loss Function
            ce_loss = F.cross_entropy(student_logits, labels)
            kl_loss = nn.KLDivLoss(reduction='batchmean')(
                F.log_softmax(student_logits / T, dim=1),
                F.softmax(teacher_logits / T, dim=1)
            )
            loss = alpha * ce_loss + (1 - alpha) * (T ** 2) * kl_loss
            
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}/{epochs} complete.")

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainloader, testloader = get_dataloader()

    # Load Teacher
    teacher = OversizedCNN().to(device)
    teacher.load_state_dict(torch.load('checkpoints/teacher_model.pth', weights_only=True))

    # Initialize Students
    plain_student = StudentCNN().to(device)
    distilled_student = StudentCNN().to(device)
    
    print(f"Student Parameters: {count_parameters(plain_student):,}\n")

    # 1. Train Plain Student
    train_plain_student(plain_student, trainloader, device)
    plain_acc, plain_lat = evaluate(plain_student, testloader, device)
    
    # 2. Train Distilled Student
    train_distilled_student(distilled_student, teacher, trainloader, device)
    dist_acc, dist_lat = evaluate(distilled_student, testloader, device)

    # Save and measure sizes
    torch.save(distilled_student.state_dict(), 'checkpoints/student_model.pth')
    student_size = os.path.getsize('checkpoints/student_model.pth') / (1024 * 1024)

    print("\n--- Task 4: Final Comparison ---")
    print(f"Plain Student Accuracy:     {plain_acc:.2f}% | Latency: {plain_lat:.2f} ms")
    print(f"Distilled Student Accuracy: {dist_acc:.2f}% | Latency: {dist_lat:.2f} ms")
    print(f"Student File Size:          {student_size:.2f} MB")

if __name__ == "__main__":
    main()