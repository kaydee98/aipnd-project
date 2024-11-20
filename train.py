""" 
Train a new network on a dataset and save the model as a checkpoint.
"""
import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.models as models
from torchvision import datasets, transforms
from torchvision.models import VGG16_Weights, VGG13_Weights

dataloaders = {}

def get_input_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train a new network on a dataset")
    parser.add_argument("data_dir", type=str, help="Dataset directory")
    parser.add_argument("--save_dir", type=str, default="./", help="Directory to save checkpoints")
    parser.add_argument("--arch", type=str, default="vgg16", help="Architecture - vgg16 or vgg13")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--hidden_units", type=int, default=512, help="Number of hidden units")
    parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--gpu", action="store_true", help="Use GPU for training")
    return parser.parse_args()

def load_data(data_dir):
    """Load data from a directory."""
    train_dir = os.path.join(data_dir, "train")
    valid_dir = os.path.join(data_dir, "valid")
    test_dir = os.path.join(data_dir, "test")

    # Define your transforms for the training, validation, and testing sets
    data_transforms = {
        "train": transforms.Compose([
            transforms.RandomRotation(30),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        "valid": transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        "test": transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }

    # Load the datasets with ImageFolder
    image_datasets = {
        "train": datasets.ImageFolder(train_dir, transform=data_transforms["train"]),
        "valid": datasets.ImageFolder(valid_dir, transform=data_transforms["valid"]),
        "test": datasets.ImageFolder(test_dir, transform=data_transforms["test"])
    }

    # Define the dataloader with image datasets and the transforms
    dataloaders['train'] = DataLoader(image_datasets['train'], batch_size=64, shuffle=True)
    dataloaders['valid'] = DataLoader(image_datasets['valid'], batch_size=64)
    dataloaders['test'] = DataLoader(image_datasets['test'], batch_size=64)

def create_model(arch, hidden_units):
    """Create a model."""
    if arch == "vgg16":
        model = models.vgg16(weights=VGG16_Weights.DEFAULT)
    elif arch == "vgg13":
        model = models.vgg13(weights=VGG13_Weights.DEFAULT)
    else:
        print("Invalid model architecture")
        return None

    #Hyperparameters for the network
    input_size = 25088
    output_size = 102

    # Freeze parameters
    for param in model.parameters():
        param.requires_grad = False

    # Define classifier
    classifier = nn.Sequential(
        nn.Linear(input_size, hidden_units),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(hidden_units, output_size),
        nn.LogSoftmax(dim=1)
    )

    # Replace the classifier in the model
    model.classifier = classifier

    return model


def validation(model, loader, criterion):
    """
        validation of model against test/validation data
    """
    model.eval()
    accuracy = 0
    loss = 0

    with torch.no_grad():

        for images, labels in loader:
            output = model(images)
            loss += criterion(output, labels).item()

            # Calculate accuracy
            ps = torch.exp(output)
            top_class = ps.topk(1, dim=1)[1]
            equals = top_class == labels.view(*top_class.shape)
            accuracy += equals.float().mean().item()

    # Return the average loss and accuracy
    return loss/len(loader), accuracy/len(loader)


def train_model(model, optimizer, criterion, epochs, gpu):
    """
        Trains a model using a dataset
    """
    trainloader = dataloaders["train"]
    validloader = dataloaders["valid"]

    print("Training started")

    device = torch.device("cuda" if gpu and torch.cuda.is_available() else "cpu")
    model.to(device)

    for e in range(epochs):
        model.train()
        running_loss = 0

        # Training Loop
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()

            output = model(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        val_loss, val_accuracy = validation(model, validloader, criterion)

        print(f"Epoch {e+1}/{epochs}.. "
          f"Training Loss: {running_loss/len(trainloader):.3f}.. "
          f"Validation Loss: {val_loss:.3f}.. "
          f"Validation Accuracy: {val_accuracy*100:.2f}%")

    print("Training complete")

    return

def main():
    """Main function."""
    in_args = get_input_args()

    # Load data
    load_data(in_args.data_dir)

    # Hyperparameters for the network
    epochs = in_args.epochs
    learning_rate = in_args.learning_rate
    hidden_units = in_args.hidden_units

    gpu =  in_args.gpu
    # Create model
    model = create_model(in_args.arch, hidden_units)

    #Define the loss function and optimizer
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)

    # Train model
    train_model(model, optimizer, criterion, epochs, gpu)

    # Test the model
    print("Testing the model")
    test_loss, test_accuracy = validation(model, dataloaders['test'], criterion )
    print(f"Test Loss: {test_loss:.3f}.. "
      f"Test Accuracy: {test_accuracy*100:.2f}%")

    # Save the checkpoint
    checkpoint_path = os.path.join(in_args.save_dir, "train_checkpoint.pth")

    model.class_to_idx = dataloaders['train'].dataset.class_to_idx
    checkpoint = {
        'arch': 'vgg16',
        'input_size': 25088,
        'output_size': 102,
        'hidden_units': hidden_units,
        'epochs': epochs,
        'learning_rate': learning_rate,
        'classifier': model.classifier,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'class_to_idx': model.class_to_idx
    }
    torch.save(checkpoint, checkpoint_path)

if __name__ == "__main__":
    main()
