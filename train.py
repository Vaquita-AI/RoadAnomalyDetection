import os
import copy
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import models, datasets, transforms

def main():

    # Each subfolder inside this dir should represent a class -> inside each subfolder are the images of this class
    dataset_path = "Dataset/iteration_3" 
    save_dir = 'Runs'

    run_name = input("Please enter the run name 🏃: ")

    num_epochs = 1
    patience = 5

    num_classes = 2

    print(f"Current GPU 🖥️ : {torch.cuda.get_device_name(0)}")
    #----------------------------------------Preprocessing----------------------------------------#
    # Define transformations for training, validation, and test sets
    train_transforms = transforms.Compose([
        #transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),  # Randomly crop and resize to 224x224
        transforms.Resize((256, 256)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize with ImageNet mean and std
        #RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value='random')  # Randomly erase a rectangle region with 50% probability
    ])

    val_test_transforms = transforms.Compose([
        transforms.Resize((256, 256)),
        #transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load the dataset
    dataset = datasets.ImageFolder(root=dataset_path, transform=train_transforms)

    # Split the dataset into training, validation, and test sets
    train_size = int(0.8 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    # Apply validation/test transformations to the validation and test sets
    val_dataset.dataset.transform = val_test_transforms
    test_dataset.dataset.transform = val_test_transforms

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=8)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=8)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=8)

    #----------------------------------------Training----------------------------------------#
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = models.resnet18(pretrained=True)

    model.fc = nn.Linear(model.fc.in_features, num_classes)

    model.to(device)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)  # weight_decay for L2 regularization
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)#, verbose=True)

    # Early stopping parameters

    best_val_loss = float('inf')
    best_model_wts = copy.deepcopy(model.state_dict())
    early_stop_counter = 0

    # Training loop


    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)

        # Training phase
        model.train()
        running_loss = 0.0
        running_corrects = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)  # Move data to GPU
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / train_size
        epoch_acc = running_corrects.double() / train_size

        print(f'Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

        # Validation phase
        model.eval()
        val_running_loss = 0.0
        val_running_corrects = 0

        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)  # Move data to GPU
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            val_running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            val_running_corrects += torch.sum(preds == labels.data)

        val_loss = val_running_loss / val_size
        val_acc = val_running_corrects.double() / val_size

        print(f'Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}')

        # Check for improvement
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            early_stop_counter = 0
        else:
            early_stop_counter += 1

        # Adjust learning rate
        scheduler.step(val_loss)

        # Early stopping
        if early_stop_counter >= patience:
            print("Early stopping")
            break

    # Load best model weights
    model.load_state_dict(best_model_wts)

    # Save the best model
    model_save_path = os.path.join(save_dir, f'best_model_{run_name}.pth')
    torch.save(model.state_dict(), model_save_path)

    print("Training Complete! 🥳🎉")
    print("Performing validation on the Test set...")
    model.to(device)

    model.eval()
    test_running_corrects = 0

    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)  # Move data to GPU
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        test_running_corrects += torch.sum(preds == labels.data)

    test_acc = test_running_corrects.double() / test_size
    print(f'Test Acc: {test_acc:.4f}')

if __name__ == "__main__":
    main()