import os
import random
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms

# Define the path to the dataset
dataset_path = r"input_images"

# List all files in the directory
image_files = [f for f in os.listdir(dataset_path) if os.path.isfile(os.path.join(dataset_path, f))]

# number of images to process
num_images = 6

# mode of transformation application
"""
'combined': combines all transforms in the output image (realistic for training)
'single': only one transform is done at a time
"""
mode = 'combined' 

# Define the transformations
individual_transformations = {
    "Resize": transforms.Resize((256, 256)),
    "HorizontalFlip": transforms.RandomHorizontalFlip(p=1),
    "Rotation": transforms.RandomRotation(degrees=30),
    "ColorJitter": transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2),
    "Affine": transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.8, 1.2), shear=10),
    "Perspective": transforms.RandomPerspective(distortion_scale=0.3, p=1),
    "Grayscale": transforms.RandomGrayscale(p=1),
}

combined_transformations = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomHorizontalFlip(p=1),
    transforms.RandomRotation(degrees=30),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.10),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.8, 1.2), shear=10),
    transforms.RandomPerspective(distortion_scale=0.2, p=1),
    #transforms.RandomGrayscale(p=1),
    transforms.ToTensor(),
    transforms.ToPILImage()
])

# Function to display images
def display_images(original, transformed, transform_name):
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(original)
    axes[0].set_title("Original Image")
    axes[0].axis('off')

    axes[1].imshow(transformed)
    axes[1].set_title(transform_name)
    axes[1].axis('off')

    plt.show()

# Apply transformations to the specified number of images
for _ in range(num_images):
    # Randomly select an image
    random_image_file = random.choice(image_files)
    image_path = os.path.join(dataset_path, random_image_file)

    # Open the image
    original_image = Image.open(image_path)

    if mode == 'single':
        # Apply each transformation separately and display the result
        for name, transform in individual_transformations.items():
            transformed_image = transform(original_image)
            display_images(original_image, transformed_image, name)
    elif mode == 'combined':
        # Apply all transformations to the same image and display the result
        transformed_image = combined_transformations(original_image)
        display_images(original_image, transformed_image, "Combined Transformations")
    else:
        print("Invalid mode selected. Please enter 'single' or 'combined'.")