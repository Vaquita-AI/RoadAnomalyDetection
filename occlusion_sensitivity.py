import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image, ImageDraw, ImageFilter
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm # For progress bar

# --- User selectable modes ---
# Choose 'gaussian_blur' for heatmap visualization  
# Choose 'occlusion_boxes' for standard occlusion box visualization 
visualization_type = 'occlusion_boxes' # Options: 'gaussian_blur', 'occlusion_boxes'

# If visualization_type is 'gaussian_blur', choose how to handle output:
# 'display': Shows the heatmap interactively.
# 'save': Saves the heatmap to a file.
mode = 'display' # Options: 'display', 'save' (only applicable for 'gaussian_blur')

# If mode is 'save' and visualization_type is 'gaussian_blur', choose what to save:
# 'figure': Saves the entire matplotlib figure (original image + heatmap side-by-side).
# 'heatmap_overlay': Saves only the original image with the heatmap overlaid on it.
save_what = 'heatmap_overlay' # Options: 'figure', 'heatmap_overlay' (only applicable for 'gaussian_blur' and mode 'save')
# ---------------------------

image_dir = "input_images" # Directory containing the images
output_dir = r"output_images\occlusion"
os.makedirs(output_dir, exist_ok=True) # Ensure output directory exists

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the trained model
num_classes = 2 # Assuming your model is trained for 2 classes
model = models.resnet18()
model.fc = nn.Linear(model.fc.in_features, num_classes)
model_path = 'best_model.pth' # Make sure this path is correct
try:
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    print(f"Model loaded successfully from {model_path} and moved to {device}")
except FileNotFoundError:
    print(f"Error: Model file not found at {model_path}. Please ensure the model path is correct.")
    exit() # Exit if model not found

# Define preprocessing transforms
# Use 224x224 for ResNet, consistent with Code 2's improvement
image_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Inverse transform for visualization (to convert tensor back to PIL Image)
inv_normalize = transforms.Normalize(
    mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
    std=[1/0.229, 1/0.224, 1/0.225]
)
to_pil_image = transforms.ToPILImage()

def visualize_occlusion_sensitivity(model, image_path, image_transforms, inv_normalize, to_pil_image,
                                   window_size=50, stride=25, blur_radius_occlusion=10,
                                   blur_radius_heatmap=15, visualization_type='gaussian_blur',
                                   mode='display', output_dir='.', save_what='figure'):
    """
    Visualizes occlusion sensitivity for a given image.

    Args:
        model (torch.nn.Module): The trained model.
        image_path (str): Path to the input image.
        image_transforms (torchvision.transforms.Compose): Transforms for model input.
        inv_normalize (torchvision.transforms.Normalize): Inverse normalization for visualization.
        to_pil_image (torchvision.transforms.ToPILImage): Transform to convert tensor to PIL Image.
        window_size (int): Size of the occlusion window.
        stride (int): Stride for moving the occlusion window.
        blur_radius_occlusion (int): Gaussian blur radius for the occluded region (used in both types).
        blur_radius_heatmap (int): Gaussian blur radius for smoothing the final heatmap (only in 'gaussian_blur').
        visualization_type (str): 'gaussian_blur' or 'occlusion_boxes'.
        mode (str): 'display' or 'save' (only applicable for 'gaussian_blur' type).
        output_dir (str): Directory to save output (only applicable for 'gaussian_blur' type).
        save_what (str): 'figure' or 'heatmap_overlay' (only applicable for 'gaussian_blur' type).
    """
    original_image = Image.open(image_path).convert("RGB")
    input_tensor = image_transforms(original_image).unsqueeze(0).to(device)

    # Get original prediction
    with torch.no_grad():
        original_output = model(input_tensor)
        original_probs = torch.nn.functional.softmax(original_output, dim=1)
        _, predicted_class = torch.max(original_output, 1)
        original_confidence = original_probs[0, predicted_class.item()].item()
        print(f"Original image: {os.path.basename(image_path)}, Predicted Class: {predicted_class.item()}, Confidence: {original_confidence:.4f}")

    # Create a heatmap array
    img_width, img_height = original_image.size
    heatmap = np.zeros((img_height, img_width), dtype=np.float32)
    count_map = np.zeros((img_height, img_width), dtype=np.float32) # To average values in overlapping regions

    # Convert the original image to a tensor suitable for direct manipulation (without normalization for blurring)
    original_tensor_for_blur = transforms.ToTensor()(original_image).unsqueeze(0)

    # Iterate through windows
    for y in tqdm(range(0, img_height - window_size + 1, stride), desc=f"Processing {os.path.basename(image_path)}"):
        for x in range(0, img_width - window_size + 1, stride):
            # Create a copy of the tensor for occlusion
            occluded_tensor_for_blur = original_tensor_for_blur.clone()

            # Create a PIL image from the tensor to apply Gaussian blur
            pil_occluded_region = to_pil_image(occluded_tensor_for_blur[0, :, y:y+window_size, x:x+window_size])
            blurred_region = pil_occluded_region.filter(ImageFilter.GaussianBlur(blur_radius_occlusion)) # Always use blur_radius_occlusion
            blurred_region_tensor = transforms.ToTensor()(blurred_region)

            # Place the blurred region back into the occluded tensor
            occluded_tensor_for_blur[0, :, y:y+window_size, x:x+window_size] = blurred_region_tensor

            # Apply normalization to the occluded tensor
            occluded_input_tensor = image_transforms(to_pil_image(occluded_tensor_for_blur.squeeze(0))).unsqueeze(0).to(device)

            # Get prediction for occluded image
            with torch.no_grad():
                occluded_output = model(occluded_input_tensor)
                occluded_probs = torch.nn.functional.softmax(occluded_output, dim=1)
                occluded_confidence = occluded_probs[0, predicted_class.item()].item()

            # Calculate change in confidence for the predicted class
            # A larger drop (positive value) indicates higher importance
            confidence_drop = original_confidence - occluded_confidence

            # Accumulate values in the heatmap
            heatmap[y:y+window_size, x:x+window_size] += confidence_drop
            count_map[y:y+window_size, x:x+window_size] += 1

    # Average the heatmap values (handle overlapping windows)
    heatmap = np.divide(heatmap, count_map, out=np.zeros_like(heatmap), where=count_map!=0)

    if visualization_type == 'gaussian_blur':
        # Apply Gaussian blur to the *heatmap* for smoothing (Code 1's specific step)
        min_val_heatmap = np.min(heatmap)
        max_val_heatmap = np.max(heatmap)
        if max_val_heatmap == min_val_heatmap:
            heatmap_for_pil = np.zeros_like(heatmap, dtype=np.uint8)
        else:
            heatmap_for_pil = (heatmap - min_val_heatmap) / (max_val_heatmap - min_val_heatmap) * 255
            heatmap_for_pil = heatmap_for_pil.astype(np.uint8)

        heatmap_pil = Image.fromarray(heatmap_for_pil, mode='L')
        heatmap_blurred = heatmap_pil.filter(ImageFilter.GaussianBlur(blur_radius_heatmap))
        heatmap = np.array(heatmap_blurred).astype(np.float32)

    # Normalize heatmap for visualization (0-1 range)
    min_heatmap_val = np.min(heatmap)
    max_heatmap_val = np.max(heatmap)
    heatmap_norm = (heatmap - min_heatmap_val) / (max_heatmap_val - min_heatmap_val + 1e-8) # Add small epsilon

    # --- Plotting and Saving Logic based on visualization_type ---
    if visualization_type == 'gaussian_blur':
        if mode == 'display' or (mode == 'save' and save_what == 'figure'):
            fig, axes = plt.subplots(1, 2, figsize=(16, 7))

            # Original Image
            axes[0].imshow(original_image)
            axes[0].set_title('Original Image')
            axes[0].axis('off')

            # Heatmap
            axes[1].imshow(original_image) # Display original image first
            im_heatmap = axes[1].imshow(heatmap_norm, cmap='jet', alpha=0.5, interpolation='bilinear')
            axes[1].set_title('Occlusion Sensitivity Heatmap')
            axes[1].axis('off')

            # Add a color bar
            cbar = fig.colorbar(im_heatmap, ax=axes[1], fraction=0.046, pad=0.04)
            cbar.set_label('Confidence Drop (Higher = More Important)')

            plt.suptitle(f"Explanation for: {os.path.basename(image_path)}\nPredicted Class: {predicted_class.item()}, Original Confidence: {original_confidence:.4f}")
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])

            if mode == 'display':
                plt.show()
            elif mode == 'save':
                output_filepath = os.path.join(output_dir, os.path.splitext(os.path.basename(image_path))[0] + "_heatmap_figure.png")
                plt.savefig(output_filepath)
                print(f"Saved full figure to {output_filepath}")
                plt.close(fig)

        elif mode == 'save' and save_what == 'heatmap_overlay':
            fig, ax = plt.subplots(figsize=(original_image.width/100, original_image.height/100), dpi=100)
            ax.imshow(original_image)
            im_heatmap = ax.imshow(heatmap_norm, cmap='jet', alpha=0.5, interpolation='bilinear')
            ax.axis('off')
            fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

            output_filepath = os.path.join(output_dir, os.path.splitext(os.path.basename(image_path))[0] + "_heatmap_overlay.png")
            plt.savefig(output_filepath)
            print(f"Saved heatmap overlay to {output_filepath}")
            plt.close(fig)
        else:
            print("Invalid mode or save_what option selected for 'gaussian_blur'. Please check your settings.")

    elif visualization_type == 'occlusion_boxes':
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))

        # Original Image
        axes[0].imshow(original_image)
        axes[0].set_title('Original Image')
        axes[0].axis('off')

        # Heatmap
        axes[1].imshow(original_image) # Display original image first
        im_heatmap = axes[1].imshow(heatmap_norm, cmap='jet', alpha=0.5) # Overlay heatmap
        axes[1].set_title('Occlusion Sensitivity Heatmap')
        axes[1].axis('off')

        # Add a color bar
        cbar = fig.colorbar(im_heatmap, ax=axes[1], fraction=0.046, pad=0.04)
        cbar.set_label('Confidence Drop (Higher = More Important)')

        plt.suptitle(f"Explanation for: {os.path.basename(image_path)}\nPredicted Class: {predicted_class.item()}, Original Confidence: {original_confidence:.4f}")
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        if mode == 'display':
            plt.show()
        elif mode == 'save':
            output_filepath = os.path.join(output_dir, os.path.splitext(os.path.basename(image_path))[0] + "_occlusion_figure.png")
            plt.savefig(output_filepath)
            print(f"Saved occlusion figure to {output_filepath}")
            plt.close(fig)
        else:
            print("Invalid mode selected for 'occlusion_boxes'. Choose 'display' or 'save'.")

    else:
        print("Invalid visualization_type selected. Choose 'gaussian_blur' or 'occlusion_boxes'.")


# Get list of image files
image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]

if not image_files:
    print(f"No image files found in '{image_dir}'. Please place some images there.")
else:
    for image_file in image_files:
        image_path = os.path.join(image_dir, image_file)
        visualize_occlusion_sensitivity(model, image_path, image_transforms, inv_normalize, to_pil_image,
                                         window_size=50, stride=25, blur_radius_occlusion=10, blur_radius_heatmap=40, # blur_radius_occlusion acts as blur_radius for both
                                         visualization_type=visualization_type, mode=mode, output_dir=output_dir, save_what=save_what)