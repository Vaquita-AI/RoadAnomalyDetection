import os
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from torchcam.methods import GradCAMpp
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import cv2
import torch.nn as nn

# Ask user for mode
mode = 'display' #'display' or 'save'
image_dir = "input_images" #dir containing the imgs
output_dir = r"output_images\gradcam"

# Load the trained model
num_classes = 2
model = models.resnet18()
model.fc = nn.Linear(model.fc.in_features, num_classes)
model_path = 'best_model.pth'
model.load_state_dict(torch.load(model_path))
model.eval()

# Define preprocessing transforms 
image_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Define custom labels
labels = ["Class0_Normal", "Class1_Anomalies"] 

# Define the directory containing images
image_dir = "input_images"

# Initialize Grad-CAM
cam_extractor = GradCAMpp(model, target_layer='layer4')

# Get all image paths
image_paths = [os.path.join(image_dir, img_name) for img_name in os.listdir(image_dir) 
               if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]


# Create a figure for display
fig, ax = plt.subplots(figsize=(10, 8))
plt.subplots_adjust(bottom=0.2)  # Make space for status text

# Add status text
status_text = fig.text(0.5, 0.05, "", ha='center', va='center')

# Initialize current image index
current_idx = 0

def update_display(idx):
    """Update the display with the image at the given index"""
    global current_idx
    current_idx = idx % len(image_paths)  # Wrap around
    
    img_path = image_paths[current_idx]
    image = Image.open(img_path)
    
    # Preprocess and predict
    input_tensor = image_transforms(image).unsqueeze(0)
    output = model(input_tensor)
    class_idx = torch.argmax(output[0]).item()
    class_name = labels[class_idx] if class_idx < len(labels) else "Unknown"
    
    # Generate Grad-CAM heatmap
    activation_map = cam_extractor(class_idx=class_idx, scores=output)
    activation_map = activation_map[0].squeeze().cpu().numpy()
    activation_map = (activation_map - activation_map.min()) / (activation_map.max() - activation_map.min())
    activation_map = cv2.resize(activation_map, (image.size[0], image.size[1]), interpolation=cv2.INTER_LINEAR)
    
    # Create heatmap overlay
    heatmap = cv2.applyColorMap(np.uint8(255 * activation_map), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlayed_image = Image.blend(image.convert("RGB"), Image.fromarray(heatmap), alpha=0.5)
    
    if mode == 'display':
        # Update display
        ax.clear()
        ax.imshow(overlayed_image)
        ax.axis('off')
        ax.set_title(f"Image {current_idx + 1}/{len(image_paths)}: {os.path.basename(img_path)}")
        
        # Add prediction text on top of the image
        ax.text(10, 20, f"Prediction: {class_name}", color='white', fontsize=12, backgroundcolor='black')
        
        status_text.set_text(f"Press ← for previous, → for next, ESC to close")
        
        fig.canvas.draw()
    elif mode == 'save':
        # Save the overlayed image
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, os.path.basename(img_path))
        overlayed_image.save(output_path)
        print(f"Saved: {output_path}")

def on_key(event):
    """Handle keyboard events"""
    if event.key == 'right':
        update_display(current_idx + 1)
    elif event.key == 'left':
        update_display(current_idx - 1)
    elif event.key == 'escape':
        plt.close()

if mode == 'display':
    # Connect the keyboard event handler
    fig.canvas.mpl_connect('key_press_event', on_key)

    # Show the first image
    update_display(0)

    plt.show()
else:
    # In save mode, process all images
    for idx in range(len(image_paths)):
        update_display(idx)