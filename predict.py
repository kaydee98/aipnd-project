""" Class Prediction """
import argparse
import json
import torch
from PIL import Image
import numpy as np
from torchvision import models
from torchvision.models import VGG16_Weights, VGG13_Weights


def get_input_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Predict flower name and class probability.")
    parser.add_argument('--image_path', type=str, help="Path to the image file")
    parser.add_argument('--checkpoint', type=str, help="Path to the model checkpoint")
    parser.add_argument('--top_k', type=int, default=5, help="Return top K most likely classes")
    parser.add_argument('--category_names', default='cat_to_name.json', type=str, help="Path to mapping JSON")
    parser.add_argument('--gpu', action='store_true', help="Use GPU for inference")
    return parser.parse_args()

def load_category_names(category_names):
    """Load category names from a JSON file."""
    with open(category_names, 'r', encoding='utf-8') as f:
        return json.load(f)

def load_checkpoint(filepath):
    """
    Load a model checkpoint from a file.
    """
    checkpoint = torch.load(filepath, weights_only=False)
    arch = checkpoint['arch']
    model = models.vgg16(weights=VGG16_Weights.DEFAULT) if arch == 'vgg16' else models.vgg13(weights=VGG13_Weights.DEFAULT)

    # freeze parameters to avoid updating them during training
    for param in model.parameters():
        param.requires_grad = False

    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['model_state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']

    model.eval()

    return model

def process_image(image_path):
    """ 
    Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns a PyTorch tensor
    """
    image = Image.open(image_path).convert("RGB")
    # Resize and crop
    image = image.resize((256, 256))
    image = image.crop((16, 16, 240, 240))
    # Normalize
    np_image = np.array(image) / 255.0
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_image = (np_image - mean) / std
    np_image = np_image.transpose((2, 0, 1))

    # Convert to a tensor
    return torch.tensor(np_image, dtype=torch.float32).unsqueeze(0)

def predict(image_path, model, top_k, device):
    """
    Predict the class (or classes) of an image using a trained deep learning model.

    Returns:
    tuple: A tuple containing:
        - probs (list): List of probabilities corresponding to the top_k classes.
        - classes (list): List of class indices corresponding to the top_k probabilities.
    """
    model.eval()
    image = process_image(image_path).to(device)
    with torch.no_grad():
        outputs = model(image)
        probs, indices = torch.exp(outputs).topk(top_k)

    # Convert tensors to lists and map indices to class labels
    probs = probs.squeeze().tolist()
    indices = indices.squeeze().tolist()

    # Map indices to class labels
    idx_to_class = {v: k for k, v in model.class_to_idx.items()}
    classes = [idx_to_class[idx] for idx in indices]

    return probs, classes

def main():
    """Main function."""
    args = get_input_args()
    device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")

    # Load model from checkpoint
    model = load_checkpoint(args.checkpoint)
    model = model.to(device)

    # Make prediction
    probs, classes = predict(args.image_path, model, args.top_k, device)

    # Load category names
    cat_to_name = load_category_names(args.category_names)
    class_names = [cat_to_name[str(cls)] for cls in classes] # Map class index to class name

    print("Predictions:")
    for prob, cls in zip(probs, class_names):
        print(f"{cls}: {prob:.3f}")

if __name__ == "__main__":
    main()
