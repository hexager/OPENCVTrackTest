import torch
import json
from torch import nn, optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from model import CNN

def compute_stats(dataset):
    """
    Compute the global mean and std of the dataset.
    Expects dataset with transform=ToTensor() only.
    """
    loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)
    data, _ = next(iter(loader))  # data shape: [N, 1, 28, 28]
    mean = data.mean().item()
    std = data.std().item()
    return mean, std

def train_model(epochs=5, batch_size=64, learning_rate=0.001, 
                model_save_path='cnn_mnist.pth', stats_path='stats.json', validation_split=0.1):
    
    # Load dataset with only ToTensor so we can compute stats
    base_transform = transforms.ToTensor()
    train_dataset_for_stats = datasets.MNIST(root='./data', train=True, download=True, transform=base_transform)
    
    # Compute mean and std
    mean, std = compute_stats(train_dataset_for_stats)
    print(f"Computed Mean: {mean:.4f}  Std: {std:.4f}")
    
    # Save these stats to a file for later use in inference
    with open(stats_path, 'w') as f:
        json.dump({'mean': mean, 'std': std}, f)
    
    # Define final transform using computed statistics
    final_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((mean,), (std,))
    ])
    
    # Reload training dataset with the final transform (normalization)
    full_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=final_transform)
    total_len = len(full_dataset)
    val_size = int(total_len * validation_split)
    train_size = total_len - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    print("Starting training...")
    for epoch in range(epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()         # reset gradients
            outputs = model(images)       # forward pass
            loss = criterion(outputs, labels)
            loss.backward()               # backpropagation
            optimizer.step()              # weight update
            running_loss += loss.item()
        
        avg_train_loss = running_loss / len(train_loader)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        
        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = correct / total
        
        print(f"Epoch [{epoch+1}/{epochs}] "
              f"Train Loss: {avg_train_loss:.4f} "
              f"Val Loss: {avg_val_loss:.4f} "
              f"Val Accuracy: {val_accuracy:.4f}")
    
    # Save the model state
    torch.save(model.state_dict(), model_save_path)
    print(f"Training complete. Model saved to {model_save_path}")

if __name__ == '__main__':
    train_model()
