from models import BiTLikeModel

global model

def load_model():
    DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device-------------------: {DEVICE}")

    # Load the model checkpoint
    checkpoint = torch.load('best.pt', map_location=torch.device(DEVICE))

    # Initialize the model
    model = BiTLikeModel(2)  # Replace with your model class

    # Get the model's state_dict
    model_state_dict = model.state_dict()

    # Filter out mismatched keys (e.g., fc layer)
    filtered_checkpoint = {k: v for k, v in checkpoint.items() if k in model_state_dict and model_state_dict[k].shape == v.shape}

    # Load the compatible weights
    model_state_dict.update(filtered_checkpoint)
    model.load_state_dict(model_state_dict)

    # Reinitialize the fc layer
    torch.nn.init.xavier_uniform_(model.fc.weight)
    torch.nn.init.zeros_(model.fc.bias)

    # Set the model to evaluation mode
    model.eval()

    print("Model loaded successfully!")
    return model

import torch
from PIL import Image
from torchvision import transforms

def predict_image_class(model, image_path):
    DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {DEVICE}")

    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load and preprocess the image
    image = Image.open(image_path).convert('RGB')
    input_tensor = preprocess(image).unsqueeze(0).to(DEVICE) # Move input tensor to the determined device

    # Move the model to the same device if it's not already there
    model.to(DEVICE)
    model.eval() # Set the model to evaluation mode

    # Perform inference
    with torch.no_grad():
        output = model(input_tensor)
        print(f"output: {output}")
        probabilities = torch.softmax(output, dim=1)
        print(f"probabilities: {probabilities}")

    # Get the predicted class and probability
    predicted_probability, predicted_class_index = torch.max(probabilities, dim=1)
    predicted_class = predicted_class_index.item()
    predicted_probability = predicted_probability.item()

    print(f"Predicted class index: {predicted_class}, Probability: {predicted_probability:.4f}")
    return predicted_class, predicted_probability

import os
def test_image(model, image_path):
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Image file '{image_path}' not found.")
    return predict_image_class(model, image_path)


