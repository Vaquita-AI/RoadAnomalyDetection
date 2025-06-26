import os
import torch
import numpy as np
from torchvision import transforms, models
from PIL import Image, ImageFilter
from lime import lime_image
from skimage.segmentation import mark_boundaries, slic, felzenszwalb, quickshift
import matplotlib.pyplot as plt
import torch.nn as nn

# --- Configuration Parameters ---
IMAGE_DIR = 'input_images'
MODEL_PATH = 'best_model.pth'
CLASS_NAMES = ["Class0_Normal", "Class1_Anomalies"]
NUM_SAMPLES = 1
NUM_FEATURES = 5
SAVE_DIR = r'output_images\lime'

# --- Mode Selection Variable ---
OPERATION_MODE = 'display'

# --- Device Setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Model Loading ---
model = models.resnet18()
model.fc = nn.Linear(model.fc.in_features, len(CLASS_NAMES))

# Ensure the model file exists before attempting to load
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found at {MODEL_PATH}. Please train and save your model.")

model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

# --- Image Transformations ---
image_transforms = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# --- Modified Gaussian Blur Hide Color Function ---
sample_image_path = None
if os.path.exists(IMAGE_DIR):
    image_files = [f for f in os.listdir(IMAGE_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]
    if image_files:
        sample_image_path = os.path.join(IMAGE_DIR, image_files[0])

BLUR_RADIUS = 10
HIDE_COLOR_VALUE = 0

if sample_image_path:
    try:
        sample_image = Image.open(sample_image_path).convert('RGB')
        blurred_sample_image = sample_image.filter(ImageFilter.GaussianBlur(radius=BLUR_RADIUS))
        blurred_array = np.array(blurred_sample_image)
        HIDE_COLOR_VALUE = np.mean(blurred_array, axis=(0, 1)).astype(int)
        print(f"Calculated hide_color from blurred sample image: {HIDE_COLOR_VALUE}")
    except Exception as e:
        print(f"Could not calculate hide_color from sample image: {e}. Using default black.")
        HIDE_COLOR_VALUE = 0
else:
    print(f"No sample image found in '{IMAGE_DIR}'. Using default black for hide_color.")

# --- Prediction Function ---
def predict(images):
    pil_images = []
    for img_array in images:
        if img_array.dtype != np.uint8:
            if np.max(img_array) <= 1.0 and img_array.dtype == np.float32:
                img_array = (img_array * 255).astype(np.uint8)
            else:
                img_array = img_array.astype(np.uint8)

        pil_images.append(Image.fromarray(img_array))

    transformed_images = torch.stack([image_transforms(image) for image in pil_images])
    transformed_images = transformed_images.to(device)
    outputs = model(transformed_images)
    probabilities = torch.nn.functional.softmax(outputs, dim=1)
    return probabilities.detach().cpu().numpy()

# --- LIME Explainer Initialization ---
explainer = lime_image.LimeImageExplainer()

# --- Segmentation Method Selection ---
def select_segmentation_method(method='slic', **kwargs):
    if method == 'slic':
        return lambda image: slic(image, **kwargs)
    elif method == 'felzenszwalb':
        return lambda image: felzenszwalb(image, **kwargs)
    elif method == 'quickshift':
        return lambda image: quickshift(image, **kwargs)
    else:
        raise ValueError("Unsupported segmentation method. Choose from 'slic', 'felzenszwalb', or 'quickshift'.")

segmentation_method = 'quickshift'
segmentation_params = {
    'slic': {'n_segments': 40, 'compactness': 1, 'sigma': 0.25},
    'felzenszwalb': {'scale': 61, 'sigma': 2, 'min_size': 5},
    'quickshift': {'kernel_size': 4, 'max_dist': 16, 'ratio': 0.99}
}
segmentation_fn = select_segmentation_method(segmentation_method, **segmentation_params[segmentation_method])

# --- Main Logic ---
def run_lime_explanation(mode):
    image_paths = [os.path.join(IMAGE_DIR, img_name) for img_name in os.listdir(IMAGE_DIR)
                   if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]

    if not image_paths:
        print(f"No images found in '{IMAGE_DIR}'. Please ensure images are present in the directory.")
        return

    # Create the save directory if not in display mode
    if mode == 'save':
        os.makedirs(SAVE_DIR, exist_ok=True)
        print(f"Saving LIME explanations to '{SAVE_DIR}'...")

    explanations = []
    print(f"Generating LIME explanations for {len(image_paths)} images using Gaussian blur's mean color for hidden regions...")
    for img_path in image_paths:
        image = Image.open(img_path).convert('RGB')
        print(f"Explaining: {os.path.basename(img_path)}")
        explanation = explainer.explain_instance(
            np.array(image),
            predict,
            top_labels=1,
            hide_color=HIDE_COLOR_VALUE,
            num_samples=NUM_SAMPLES,
            segmentation_fn=segmentation_fn
        )
        explanations.append((img_path, explanation)) # Keep this for display mode

        if mode == 'save':
            original_image = Image.open(img_path).convert('RGB')
            top_label = explanation.top_labels[0]
            temp, mask = explanation.get_image_and_mask(
                top_label,
                positive_only=True,
                num_features=NUM_FEATURES,
                hide_rest=False
            )

            probabilities = predict([np.array(original_image)])
            predicted_index = np.argmax(probabilities[0])
            class_name = CLASS_NAMES[predicted_index]

            marked_image = mark_boundaries(temp / 255.0, mask)

            save_path = os.path.join(SAVE_DIR, f"LIME_{class_name}_{os.path.basename(img_path)}")
            plt.imsave(save_path, marked_image)
            print(f"Saved: {save_path}")

    print("LIME explanation generation complete.")

    if mode == 'display':
        # --- Display Mode ---
        fig, ax = plt.subplots(figsize=(10, 8))
        plt.subplots_adjust(bottom=0.2)

        status_text = fig.text(0.5, 0.05, "", ha='center', va='center')
        current_idx = 0

        def update_display(idx):
            nonlocal current_idx
            current_idx = idx % len(explanations)
            img_path, explanation = explanations[current_idx]
            image = Image.open(img_path).convert('RGB')
            top_label = explanation.top_labels[0]
            temp, mask = explanation.get_image_and_mask(
                top_label,
                positive_only=True,
                num_features=NUM_FEATURES,
                hide_rest=False
            )

            probabilities = predict([np.array(image)])
            predicted_index = np.argmax(probabilities[0])
            class_name = CLASS_NAMES[predicted_index]

            ax.clear()
            ax.imshow(mark_boundaries(temp / 255.0, mask))
            ax.axis('off')
            ax.set_title(f"Image {current_idx + 1}/{len(explanations)}: {os.path.basename(img_path)}")
            ax.text(10, 20, f"Prediction: {class_name}", color='white', fontsize=12, backgroundcolor='black')

            status_text.set_text(f"Press ← for previous, → for next, ESC to close")
            fig.canvas.draw()

        def on_key(event):
            if event.key == 'right':
                update_display(current_idx + 1)
            elif event.key == 'left':
                update_display(current_idx - 1)
            elif event.key == 'escape':
                plt.close()

        fig.canvas.mpl_connect('key_press_event', on_key)
        update_display(0)
        plt.show()

if __name__ == '__main__':
    run_lime_explanation(OPERATION_MODE)