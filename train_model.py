import kagglehub
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import json

# Setup device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def download_and_prepare_data():
    print("Downloading dataset...")
    # Download dataset
    path = kagglehub.dataset_download("mahmoudreda55/satellite-image-classification")

    print("Dataset downloaded to:", path)
    
    data_dir = os.path.join(path, "data")
    
    transform = transforms.Compose([
        transforms.RandomResizedCrop(128, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    dataset = datasets.ImageFolder(root=data_dir, transform=transform)
    
    # Save classes for the frontend
    with open('classes.json', 'w') as f:
        json.dump(dataset.classes, f)
    print("Classes:", dataset.classes)

    # Split into train/val (80/20 split)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_data, val_data = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    val_loader   = DataLoader(val_data, batch_size=32, shuffle=False)
    
    return train_loader, val_loader, dataset.classes

def evaluate_model(model, val_loader, criterion):
    model.eval()
    correct, total, val_loss = 0, 0, 0.0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total, val_loss / len(val_loader)

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, epochs=5):
    best_acc = 0.0
    for epoch in range(epochs):
        model.train()
        running_loss, correct, total = 0.0, 0, 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_acc = correct / total
        val_acc, val_loss = evaluate_model(model, val_loader, criterion)
        scheduler.step()

        print(f"Epoch {epoch+1}/{epochs} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f} | Val Loss: {val_loss:.4f}")
        
        # Save the best model
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), 'best_model.pth')
            print(f"Saved best model with Val Acc: {best_acc:.4f}")

def main():
    train_loader, val_loader, classes = download_and_prepare_data()

    # EfficientNet-B3 for stronger performance
    print("Initializing EfficientNet-B3...")
    model = models.efficientnet_b3(pretrained=True)
    num_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_features, len(classes))
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)

    print("Starting training...")
    train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, epochs=5)
    print("Training finished.")

if __name__ == "__main__":
    main()
