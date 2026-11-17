import yaml
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import mlflow
import os
import random
import numpy as np

# Reproducibility
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# Best Lab 3 Model (CNN)
class FashionCNN(nn.Module):
    def __init__(self):
        super(FashionCNN, self).__init__()
        self.layer1 = nn.Sequential(nn.Conv2d(1, 16, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2))
        self.fc = nn.Sequential(nn.Linear(32 * 7 * 7, 128), nn.ReLU(), nn.Linear(128, 10))

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        return self.fc(out)

def train():
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    set_seed(config["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    trainset = torchvision.datasets.FashionMNIST(root=config['data_dir'], train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=config['batch_size'], shuffle=True, num_workers=config['num_workers'])

    model = FashionCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    
    start_epoch = 0
    checkpoint_path = os.path.join(config['checkpoint_dir'], "checkpoint.pth")

    # Task 7: Resume from checkpoint logic
    if config['resume_from_checkpoint'] and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        print(f"Resumed from epoch {start_epoch}")

    mlflow.set_experiment(config['experiment_name'])
    with mlflow.start_run():
        mlflow.log_params(config)

        for epoch in range(start_epoch, config['epochs']):
            model.train()
            running_loss = 0.0
            optimizer.zero_grad()

            for i, (inputs, labels) in enumerate(trainloader):
                inputs, labels = inputs.to(device), labels.to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                # Task 6: Gradient Accumulation
                loss = loss / config['grad_accum_steps']
                loss.backward()

                if (i + 1) % config['grad_accum_steps'] == 0 or (i + 1) == len(trainloader):
                    optimizer.step()
                    optimizer.zero_grad()

                running_loss += loss.item() * config['grad_accum_steps']

            avg_loss = running_loss / len(trainloader)
            print(f"Epoch [{epoch+1}/{config['epochs']}], Loss: {avg_loss:.4f}")
            mlflow.log_metric("train_loss", avg_loss, step=epoch)

            # Save checkpoint
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, checkpoint_path)
            
            # Task 7: Interrupt simulation
            if epoch == 2 and not config['resume_from_checkpoint']:
                print("Interrupting training at epoch 3 (Task 7)...")
                print("Change 'resume_from_checkpoint: true' in config.yaml and run again.")
                return

        mlflow.pytorch.log_model(model, "model", input_example=inputs.cpu().numpy())
        print("Training complete.")

if __name__ == "__main__":
    train()