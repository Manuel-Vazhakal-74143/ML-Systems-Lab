import torch
import torch.nn as nn
import torch.optim as optim
from main import get_dataloaders
from models import MLP, CNN, AttentionModel

def train_and_evaluate(model, name, train_loader, test_loader, device, epochs=5):
    print(f"\n--- Training {name} ---")
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
        print(f"Epoch {epoch+1}/{epochs} - Loss: {running_loss/len(train_loader):.4f}")
        
    
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    accuracy = 100 * correct / total
    print(f"{name} Test Accuracy: {accuracy:.2f}%")
    return accuracy

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    train_loader, _, test_loader = get_dataloaders()
    
    results = {}
    results['MLP'] = train_and_evaluate(MLP(), "MLP", train_loader, test_loader, device)
    results['CNN'] = train_and_evaluate(CNN(), "CNN", train_loader, test_loader, device)
    results['Attention'] = train_and_evaluate(AttentionModel(), "Attention Model", train_loader, test_loader, device)
    
    print("\n--- Final Accuracies ---")
    for name, acc in results.items():
        print(f"{name}: {acc:.2f}%")