import os
import shutil
from PIL import Image
import torch
import torch.nn as nn
from torchvision import models, transforms


print(f"Current GPU 🖥️ : {torch.cuda.get_device_name(0)}")

user_input = input("Do you want to copy the images to the output directory? (yes/y/any other key): ").strip().lower()
copy_images = user_input in ['yes', 'y']


# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the trained model
num_classes = 2
model = models.resnet18()
model.fc = nn.Linear(model.fc.in_features, num_classes)
model_path = 'best_model.pth'
model.load_state_dict(torch.load(model_path))
model.to(device)
model.eval()

# Define preprocessing transforms 
image_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Directory containing images
image_directory = "input_images"

# Output directory for sorted images
output_directory = "output_images" 

# Initialize counters for each class
class_counts = [0] * num_classes
total_images = 0

# Function to copy images to the output directory
def copy_image_to_output(image_path, prediction):
    class_folder = os.path.join(output_directory, f'Class_{prediction}')
    os.makedirs(class_folder, exist_ok=True)
    shutil.copy(image_path, class_folder)

# Iterate over all images in the directory
for filename in os.listdir(image_directory):
    if filename.endswith(('.png', '.jpg', '.jpeg')): 
        image_path = os.path.join(image_directory, filename)
        image = Image.open(image_path)
        image = image_transforms(image).unsqueeze(0)  # Add batch dimension
        image = image.to(device)

        # Make prediction
        with torch.no_grad():
            output = model(image)
            _, prediction = torch.max(output, 1)
            class_counts[prediction.item()] += 1
            total_images += 1

            # Copy the image to the output directory (if set yes)
            if copy_images:
                copy_image_to_output(image_path, prediction.item())

# Calculate and print the percentage of each class
print("Classification results:")
for i, count in enumerate(class_counts):
    percentage = (count / total_images) * 100
    print(f'Class {i}: {percentage:.2f}%')