import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image, ImageDraw, ImageFilter
import numpy as np
import matplotlib.pyplot as plt

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the trained model
num_classes = 2
model = models.resnet18()
model.fc = nn.Linear(model.fc.in_features, num_classes)
model_path = 'best_model.pth'
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

# Define preprocessing transforms 
image_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Directory containing images and labels
image_directory = "input_images"
labels_directory = f"{image_directory}/labels"

def occlude_bbox(image, bbox):
    """Blend the area inside the bounding box with surroundings."""
    x1, y1, x2, y2 = bbox
    region = image.crop((x1, y1, x2, y2))
    blurred_region = region.filter(ImageFilter.GaussianBlur(radius=10))
    image.paste(blurred_region, (x1, y1, x2, y2))
    return image

def get_bboxes(label_file):
    """Extract bounding boxes from YOLO format label file."""
    bboxes = []
    if not os.path.exists(label_file):
        print(f"No label found for {label_file}")
        return bboxes

    with open(label_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip().split()
            _, x_center, y_center, width, height = map(float, line)
            bboxes.append((x_center, y_center, width, height))
    return bboxes

def convert_bbox_to_pixels(image, bbox):
    """Convert YOLO format bbox to pixel coordinates."""
    img_width, img_height = image.size
    x_center, y_center, width, height = bbox
    x_center *= img_width
    y_center *= img_height
    width *= img_width
    height *= img_height
    x1 = int(x_center - width / 2)
    y1 = int(y_center - height / 2)
    x2 = int(x_center + width / 2)
    y2 = int(y_center + height / 2)
    return (x1, y1, x2, y2)

def get_confidence(image):
    """Get model confidence for class 1."""
    image = image_transforms(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(image)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
    return probabilities[0][1].item()

def visualize_images(images_data):
    """Visualize images with confidence scores and allow cycling."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    plt.subplots_adjust(bottom=0.2)
    
    index = 0

    def update_display(index):
        for ax in axes:
            ax.clear()
        
        original_image, occluded_image, bbox_pixels, confidence_before, confidence_after = images_data[index]
        
        # Original image with bounding box
        draw = ImageDraw.Draw(original_image)
        draw.rectangle(bbox_pixels, outline="red", width=3)
        axes[0].imshow(original_image)
        axes[0].set_title(f"Original\nClass 1 Confidence: {confidence_before:.4f}")
        axes[0].axis('off')
        
        # Occluded image
        axes[1].imshow(occluded_image)
        axes[1].set_title(f"Occluded\nClass 1 Confidence: {confidence_after:.4f}")
        axes[1].axis('off')
        
        plt.draw()

    def on_key(event):
        nonlocal index
        if event.key == 'right':
            index = (index + 1) % len(images_data)
        elif event.key == 'left':
            index = (index - 1) % len(images_data)
        update_display(index)

    fig.canvas.mpl_connect('key_press_event', on_key)
    update_display(index)
    plt.show()

def main():
    images_data = []
    total_images = 0
    class_0_count = 0
    class_1_count = 0
    total_confidence_class_0 = 0.0
    total_confidence_class_1 = 0.0
    total_change_class_0 = 0.0
    total_change_class_1 = 0.0

    for image_name in os.listdir(image_directory):
        if image_name.endswith('.jpg') or image_name.endswith('.png'):
            image_path = os.path.join(image_directory, image_name)
            label_path = os.path.join(labels_directory, image_name.replace('.jpg', '.txt').replace('.png', '.txt'))
            
            # Load image
            image = Image.open(image_path).convert('RGB')
            
            # Get bounding boxes
            bboxes = get_bboxes(label_path)
            bbox_pixels_list = [convert_bbox_to_pixels(image, bbox) for bbox in bboxes]
            
            # Get confidence before occlusion
            confidence_before = get_confidence(image)
            
            # Occlude each bounding box and calculate confidence after occlusion
            occluded_image = image.copy()
            for bbox_pixels in bbox_pixels_list:
                occluded_image = occlude_bbox(occluded_image, bbox_pixels)
            
            confidence_after = get_confidence(occluded_image)
            
            # Store images data for visualization
            if bbox_pixels_list:
                images_data.append((image.copy(), occluded_image, bbox_pixels_list[0], confidence_before, confidence_after))
            
            # Determine predicted class
            predicted_class = 0 if confidence_before < 0.5 else 1
            
            # Accumulate statistics
            total_images += 1
            if predicted_class == 0:
                class_0_count += 1
                total_confidence_class_0 += confidence_before
                total_change_class_0 += (confidence_before - confidence_after)
            else:
                class_1_count += 1
                total_confidence_class_1 += confidence_before
                total_change_class_1 += (confidence_before - confidence_after)

    # Calculate averages
    avg_confidence_class_1 = total_confidence_class_1 / class_1_count if class_1_count > 0 else 0
    avg_change_class_1 = total_change_class_1 / class_1_count if class_1_count > 0 else 0

    # Print statistics
    print(f"Total Images: {total_images}")
    print(f"Images predicted as class 0: {class_0_count}")
    print(f"Images predicted as class 1: {class_1_count}")
    print(f"Average confidence for class 1: {avg_confidence_class_1:.4f}")
    print(f"Average change in confidence for class 1 after occlusion: {avg_change_class_1:.4f} (Higher means more precise model)")

    # Visualize images
    visualize_images(images_data)

if __name__ == "__main__":
    main()
