import os
import torch
import numpy as np
from torchvision import transforms, models
from PIL import Image
from lime import lime_image
from skimage.segmentation import mark_boundaries
import matplotlib.pyplot as plt
import torch.nn as nn

# Important parameters
IMAGE_DIR = 'input_images'  # Directory containing images to explain
MODEL_PATH = 'best_model.pth'  # Path to the trained model
CLASS_NAMES = ["Class0_Normal", "Class1_Anomalies"]  # Class names
NUM_SAMPLES = 3000  # Number of samples for LIME
NUM_FEATURES = 1  # Number of features to show in LIME explanation

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the trained model
model = models.resnet18(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, len(CLASS_NAMES))  # Adjust for number of classes
model.load_state_dict(torch.load(MODEL_PATH))
model.to(device)
model.eval()

# Define the transformations
image_transforms = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Define a function to get predictions
def predict(images):
    pil_images = [Image.fromarray(image) for image in images]  # Convert NumPy array to PIL Image
    transformed_images = torch.stack([image_transforms(image) for image in pil_images])
    transformed_images = transformed_images.to(device)
    outputs = model(transformed_images)
    probabilities = torch.nn.functional.softmax(outputs, dim=1)  # Convert outputs to probabilities
    return probabilities.detach().cpu().numpy()

# Initialize LIME Image Explainer
explainer = lime_image.LimeImageExplainer()

# Get all image paths
image_paths = [os.path.join(IMAGE_DIR, img_name) for img_name in os.listdir(IMAGE_DIR) 
               if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]

# Generate explanations for all images
explanations = []
for img_path in image_paths:
    image = Image.open(img_path).convert('RGB')
    explanation = explainer.explain_instance(
        np.array(image),
        predict,
        top_labels=1,
        hide_color=0,
        num_samples=NUM_SAMPLES
    )
    explanations.append((img_path, explanation))

# Create a figure for display
fig, ax = plt.subplots(figsize=(10, 8))
plt.subplots_adjust(bottom=0.2)  # Make space for status text

# Add status text
status_text = fig.text(0.5, 0.05, "", ha='center', va='center')

# Initialize current image index
current_idx = 0

def update_display(idx):
    """Update the display with the explanation at the given index"""
    global current_idx
    current_idx = idx % len(explanations)  # Wrap around
    
    img_path, explanation = explanations[current_idx]
    image = Image.open(img_path).convert('RGB')
    top_label = explanation.top_labels[0]
    temp, mask = explanation.get_image_and_mask(
        top_label,
        positive_only=True,
        num_features=NUM_FEATURES,
        hide_rest=False
    )

    # Get the predicted class name
    probabilities = predict([np.array(image)])
    predicted_index = np.argmax(probabilities[0])  # Get the index of the highest probability
    class_name = CLASS_NAMES[predicted_index]

    # Display the image with explanation
    ax.clear()
    ax.imshow(mark_boundaries(temp / 255.0, mask))
    ax.axis('off')
    ax.set_title(f"Image {current_idx + 1}/{len(explanations)}: {os.path.basename(img_path)}")
    # Add prediction text on top of the image
    ax.text(10, 20, f"Prediction: {class_name}", color='white', fontsize=12, backgroundcolor='black')
    
    status_text.set_text(f"Press ← for previous, → for next, ESC to close")
    
    fig.canvas.draw()

def on_key(event):
    """Handle keyboard events"""
    if event.key == 'right':
        update_display(current_idx + 1)
    elif event.key == 'left':
        update_display(current_idx - 1)
    elif event.key == 'escape':
        plt.close()

# Connect the keyboard event handler
fig.canvas.mpl_connect('key_press_event', on_key)

# Show the first explanation
update_display(0)

plt.show()
