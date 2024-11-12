import os
import time
import json
import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.transforms import functional as F
import shutil
import random
from PIL import Image, ImageDraw, ImageFont

# ==============================
# 1. Setup and Configuration
# ==============================

# Create a timestamped folder to save outputs
output_base_dir = "maskrcnn_output"
os.makedirs(output_base_dir, exist_ok=True)

timestamp = time.strftime("%Y%m%d-%H%M%S")
output_dir = os.path.join(output_base_dir, timestamp)
os.makedirs(output_dir, exist_ok=True)

predicted_images_dir = os.path.join(output_dir, "predicted_images")
os.makedirs(predicted_images_dir, exist_ok=True)

# Save a copy of this script in the timestamped folder for record-keeping
try:
    script_path = os.path.abspath(__file__)
    shutil.copy(script_path, os.path.join(output_dir, os.path.basename(__file__)))
except NameError:
    pass  # __file__ is not defined in some environments like Jupyter notebooks

# Define class-to-index mapping based on your dataset
class_to_idx = {
    'background': 0,    # Always include background as class 0
    'rsc': 1,
    'looper': 2,
    'rsm': 3,
    'thrips': 4,
    'jassid': 5,
    'tmb': 6, 
    'healthy': 7
}
num_classes = len(class_to_idx)
idx_to_class = {v: k for k, v in class_to_idx.items()}

# ==============================
# 2. Custom Dataset Class
# ==============================

class CustomDataset(Dataset):
    def __init__(self, image_dir, annotation_file, transforms=None):
        self.image_dir = image_dir
        self.transforms = transforms

        # Load annotations from JSON file
        with open(annotation_file, 'r') as f:
            annotations = json.load(f)

        self.imgs = []
        self.annotations = {}
        for key, value in annotations.items():
            filename = value['filename']
            regions = value.get('regions', {})
            # Ensure that regions is a list
            if isinstance(regions, dict):
                regions = [regions[k] for k in regions]
            if len(regions) > 0:
                self.imgs.append(filename)
                self.annotations[filename] = value

        self.imgs = sorted(list(set(self.imgs)))

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img_filename = self.imgs[idx]
        img_path = os.path.join(self.image_dir, img_filename)

        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            img = Image.new('RGB', (224, 224), (0, 0, 0))  # Black image

        img_width, img_height = img.size

        boxes, labels, masks = [], [], []

        ann = self.annotations.get(img_filename, None)
        if ann is not None and 'regions' in ann:
            regions = ann['regions']
            if isinstance(regions, dict):
                regions = [regions[key] for key in regions]

            for region in regions:
                shape_attrs = region.get('shape_attributes', {})
                region_attrs = region.get('region_attributes', {})
                label_name = region_attrs.get('label', 'background')
                label = class_to_idx.get(label_name, 0)
                labels.append(label)

                if shape_attrs.get('name') == 'polygon':
                    all_points_x = shape_attrs.get('all_points_x', [])
                    all_points_y = shape_attrs.get('all_points_y', [])

                    if len(all_points_x) < 3 or len(all_points_y) < 3:
                        continue

                    mask = Image.new('L', (img_width, img_height), 0)
                    polygon = list(zip(all_points_x, all_points_y))
                    ImageDraw.Draw(mask).polygon(polygon, outline=1, fill=1)
                    mask = np.array(mask, dtype=np.uint8)

                    masks.append(mask)

                    # Validate box coordinates
                    x_min, x_max = min(all_points_x), max(all_points_x)
                    y_min, y_max = min(all_points_y), max(all_points_y)
                    if x_max <= x_min or y_max <= y_min:
                        print(f"Invalid box coordinates for image {img_filename}: {[x_min, y_min, x_max, y_max]}")
                        continue
                    boxes.append([x_min, y_min, x_max, y_max])

                elif shape_attrs.get('name') == 'rect':
                    x = shape_attrs.get('x', 0)
                    y = shape_attrs.get('y', 0)
                    width = shape_attrs.get('width', 0)
                    height = shape_attrs.get('height', 0)
                    x_min, y_min = x, y
                    x_max, y_max = x + width, y + height

                    # Validate box coordinates
                    if x_max <= x_min or y_max <= y_min:
                        print(f"Invalid rectangle for image {img_filename}: {shape_attrs}")
                        continue

                    boxes.append([x_min, y_min, x_max, y_max])
                    masks.append(self.create_rect_mask(x_min, y_min, x_max, y_max, img_width, img_height))

        if boxes:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)
            masks = torch.as_tensor(np.stack(masks), dtype=torch.uint8)

            # Ensure that masks have the correct shape
            if masks.dim() != 3:
                masks = masks.unsqueeze(0)
        else:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
            masks = torch.zeros((0, img_height, img_width), dtype=torch.uint8)

        img = F.to_tensor(img)
        target = {
            "boxes": boxes,
            "labels": labels,
            "masks": masks,
            "image_id": torch.tensor([idx]),
            "area": (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]),
            "iscrowd": torch.zeros((boxes.shape[0],), dtype=torch.int64)
        }

        if self.transforms:
            img = self.transforms(img)

        return img, target

    def create_rect_mask(self, x_min, y_min, x_max, y_max, img_width, img_height):
        mask = Image.new('L', (img_width, img_height), 0)
        ImageDraw.Draw(mask).rectangle([x_min, y_min, x_max, y_max], outline=1, fill=1)
        return np.array(mask, dtype=np.uint8)

# ==============================
# 3. Collate Function
# ==============================

def collate_fn(batch):
    return tuple(zip(*batch))

# ==============================
# 4. Data Preparation
# ==============================

# Update these paths to your actual data locations
image_dir = '/home/idrone2/Desktop/new'             # Path to your images folder
annotation_file = '/home/idrone2/Desktop/tea_pest.json'  # Path to your annotations file

dataset = CustomDataset(image_dir=image_dir, annotation_file=annotation_file)

dataset_size = len(dataset)
test_size = int(0.2 * dataset_size)
train_size = dataset_size - test_size

random_seed = 42
torch.manual_seed(random_seed)
random.seed(random_seed)

train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

data_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=4, collate_fn=collate_fn)
data_loader_test = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4, collate_fn=collate_fn)

# ==============================
# 5. Model Initialization
# ==============================

# Addressing the deprecation warning by using 'weights' instead of 'pretrained'
from torchvision.models.detection import MaskRCNN_ResNet50_FPN_Weights

# Choose the appropriate weights, e.g., COCO_V1 or DEFAULT
weights = MaskRCNN_ResNet50_FPN_Weights.COCO_V1
model = maskrcnn_resnet50_fpn(weights=weights)

# Replace the box predictor with a new one for our custom classes
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)

# Replace the mask predictor as well if you're training masks
in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
hidden_layer = 256
model.roi_heads.mask_predictor = torchvision.models.detection.mask_rcnn.MaskRCNNPredictor(
    in_channels=in_features_mask,
    dim_reduced=hidden_layer,
    num_classes=num_classes
)

# ==============================
# 6. Training and Validation Functions
# ==============================

def train_one_epoch(model, optimizer, data_loader, device, epoch, train_losses, print_freq=10, max_grad_norm=10):
    model.train()
    running_loss = 0.0
    for i, (images, targets) in enumerate(data_loader):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # Check for empty targets and skip if necessary
        if any([len(t["boxes"]) == 0 for t in targets]):
            print(f"Skipping batch {i} in epoch {epoch} due to empty targets.")
            continue

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        # Check for NaN losses
        if torch.isnan(losses):
            print(f"NaN loss encountered at epoch {epoch}, iteration {i}. Skipping update.")
            continue

        optimizer.zero_grad()
        losses.backward()

        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

        optimizer.step()

        running_loss += losses.item()
        if i % print_freq == 0:
            print(f"Epoch [{epoch}], Iteration [{i}/{len(data_loader)}], Loss: {losses.item():.4f}")

    epoch_loss = running_loss / len(data_loader)
    train_losses.append(epoch_loss)
    print(f"Epoch [{epoch}] Training Loss: {epoch_loss:.4f}")

def validate_one_epoch(model, data_loader, device, val_losses):
    model.train()  # Set model to training mode to compute losses
    running_val_loss = 0.0

    with torch.no_grad():
        for images, targets in data_loader:
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            # Check for empty targets and skip if necessary
            if any([len(t["boxes"]) == 0 for t in targets]):
                print("Empty target found during validation. Skipping this batch.")
                continue

            # Calculate validation loss
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            # Check for NaN losses
            if torch.isnan(losses):
                print("NaN loss encountered during validation. Skipping this batch.")
                continue

            running_val_loss += losses.item()

    if len(data_loader) > 0:
        epoch_val_loss = running_val_loss / len(data_loader)
        val_losses.append(epoch_val_loss)
        print(f"Validation Loss: {epoch_val_loss:.4f}")
    else:
        print("No data in validation loader.")
        val_losses.append(None)

# ==============================
# 7. Training Loop
# ==============================

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)

num_epochs = 10
train_losses = []
val_losses = []

params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.001, momentum=0.9, weight_decay=0.0005)  # Reduced learning rate

for epoch in range(num_epochs):
    print(f"Starting epoch {epoch + 1} / {num_epochs}")

    # Training
    train_one_epoch(model, optimizer, data_loader, device, epoch + 1, train_losses)

    # Validation (Calculate validation loss without updating weights)
    try:
        validate_one_epoch(model, data_loader_test, device, val_losses)
    except RuntimeError as e:
        print(f"RuntimeError during validation: {e}")
        break  # Exit training loop if validation fails

# Save the model after all epochs are completed
torch.save(model.state_dict(), os.path.join(output_dir, "maskrcnn_final.pth"))
print(f"Model saved to {os.path.join(output_dir, 'maskrcnn_final.pth')}")

# ==============================
# 8. Save Loss Graphs
# ==============================

# Adjust the plotting to handle cases where validation fails early
epochs_range = range(1, len(train_losses) + 1)

# Plot Training Loss
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, train_losses, label='Training Loss', color='blue')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training Loss Over Epochs')
plt.legend()

# Plot Validation Loss only if available
if val_losses and any(v is not None for v in val_losses):
    valid_epochs = [i for i, v in enumerate(val_losses, 1) if v is not None]
    valid_losses = [v for v in val_losses if v is not None]
    plt.subplot(1, 2, 2)
    plt.plot(valid_epochs, valid_losses, label='Validation Loss', color='orange')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Validation Loss Over Epochs')
    plt.legend()
else:
    print("No valid validation losses to plot.")

plt.tight_layout()
plt.savefig(os.path.join(output_dir, "loss_graphs.png"))
plt.close()
print(f"Loss graphs saved to {os.path.join(output_dir, 'loss_graphs.png')}")

# ==============================
# 9. Prediction and Visualization
# ==============================

def save_predictions_as_images(model, data_loader, device, idx_to_class, confidence_threshold=0.5, semantic_segmentation=False):
    model.eval()
    font = None
    try:
        font = ImageFont.truetype("/home/idrone2/linux-fonts/arial.ttf", 66)  # Bold font
    except IOError:
        print("Arial Bold font not found. Using default font.")
        font = ImageFont.load_default()

    with torch.no_grad():
        for idx, (images, img_filenames) in enumerate(data_loader):
            images = list(img.to(device) for img in images)
            outputs = model(images)

            if not isinstance(outputs, list) or len(outputs) == 0:
                print(f"No predictions found for image {idx+1}. Skipping.")
                continue

            output = outputs[0]

            scores = output['scores'].cpu().numpy()
            high_conf_indices = np.where(scores >= confidence_threshold)[0]
            if len(high_conf_indices) == 0:
                print(f"No predictions above confidence threshold for image {idx+1}. Skipping.")
                continue

            boxes = output['boxes'][high_conf_indices].cpu().numpy()
            labels = output['labels'][high_conf_indices].cpu().numpy()
            masks = output['masks'][high_conf_indices].cpu().numpy()

            image = images[0].cpu().mul(255).permute(1, 2, 0).byte().numpy()
            image_pil = Image.fromarray(image).convert("RGBA")

            overlay = Image.new('RGBA', image_pil.size, (255, 255, 255, 0))

            masks = masks > 0.5

            # Define colors for different classes
            class_colors = {
                1: (255, 0, 0, 128),   # Red
                2: (0, 255, 0, 128),   # Green
                3: (0, 0, 255, 128),   # Blue
                4: (255, 255, 0, 128), # Yellow
                5: (255, 0, 255, 128), # Magenta
                6: (0, 255, 255, 128), # Cyan
                7: (128, 0, 128, 128), # Purple
                8: (128, 128, 0, 128), # Olive
                # Add more colors if needed
            }

            draw = ImageDraw.Draw(image_pil)

            for i, box in enumerate(boxes):
                label = labels[i]
                class_name = idx_to_class.get(label, "Unknown")

                # Draw bounding box
                draw.rectangle(box.tolist(), outline="red", width=6)

                # Draw label with background
                text = f"{class_name}"
                # text_size = draw.textsize(text, font=font)  # Replaced due to AttributeError
                # Using textbbox to get text size
                text_bbox = draw.textbbox((0, 0), text, font=font)
                text_width = text_bbox[2] - text_bbox[0]
                text_height = text_bbox[3] - text_bbox[1]

                # Adjust text position if it goes beyond the image boundary
                text_x = max(box[0], 0)
                text_y = max(box[1] - text_height, 0)

                # Draw rectangle behind text for better visibility
                text_background = [text_x, text_y, text_x + text_width, text_y + text_height]
                draw.rectangle(text_background, fill='black')
                draw.text((text_x, text_y), text, fill='white', font=font)

                # Apply mask
                mask = masks[i, 0]
                mask_image = Image.fromarray((mask.astype(np.uint8) * 255)).convert("L")
                color = class_colors.get(label, (255, 255, 0, 128))
                colored_mask = Image.new('RGBA', image_pil.size, color)
                mask_colored = Image.composite(colored_mask, overlay, mask_image)
                overlay = Image.alpha_composite(overlay, mask_colored)

            # Composite overlay onto image
            image_pil = Image.alpha_composite(image_pil, overlay)

            # Save the instance segmentation result
            image_pil = image_pil.convert("RGB")
            output_filename = os.path.splitext(img_filenames[0])[0]  # Use original image name
            image_pil.save(os.path.join(predicted_images_dir, f"{output_filename}_instance_segmentation.png"))
            print(f"Saved {output_filename}_instance_segmentation.png")

            # Semantic segmentation
            if semantic_segmentation:
                semantic_mask = np.zeros((masks.shape[-2], masks.shape[-1]), dtype=np.uint8)
                for i, mask in enumerate(masks):
                    label = labels[i]
                    semantic_mask[mask[0]] = label

                # Apply color map
                color_mapped_mask = Image.fromarray(semantic_mask.astype(np.uint8), mode='P')
                palette = []
                # Background color
                palette.extend([0, 0, 0])
                # Class colors
                class_palette = [
                    (255, 0, 0),        # red
                    (0, 255, 0),        # green
                    (0, 0, 255),        # blue
                    (255, 255, 0),      # yellow
                    (255, 0, 255),      # megenta
                    (0, 255, 255),      # cyan
                    (128, 0, 128),      # purple
                    (128, 128, 0),      # olvie
                    # Add more colors if needed
                ]
                for color in class_palette:
                    palette.extend(color)
                # Ensure the palette has 768 values (256 * 3)
                palette.extend([0] * (768 - len(palette)))
                color_mapped_mask.putpalette(palette)

                color_mapped_mask = color_mapped_mask.convert("RGB")
                color_mapped_mask.save(os.path.join(predicted_images_dir, f"{output_filename}_semantic_segmentation.png"))
                print(f"Saved {output_filename}_semantic_segmentation.png")

# ==============================
# 10. Inference (Prediction) Loop
# ==============================

# Since during inference we don't provide targets, create a DataLoader without targets
# Modify the DataLoader to return only images for inference
class InferenceDataset(Dataset):
    def __init__(self, image_dir, annotation_file=None, transforms=None):
        self.image_dir = image_dir
        self.transforms = transforms

        # Load annotations from JSON file if provided
        if annotation_file:
            with open(annotation_file, 'r') as f:
                annotations = json.load(f)

            self.imgs = []
            for key, value in annotations.items():
                filename = value['filename']
                regions = value.get('regions', {})
                if isinstance(regions, dict):
                    regions = [regions[k] for k in regions]
                if len(regions) > 0:
                    self.imgs.append(filename)
        else:
            # Include all images if no annotation file is provided
            self.imgs = sorted([f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))])

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img_filename = self.imgs[idx]
        img_path = os.path.join(self.image_dir, img_filename)

        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            img = Image.new('RGB', (224, 224), (0, 0, 0))  # Black image

        img = F.to_tensor(img)

        if self.transforms:
            img = self.transforms(img)

        return img, img_filename  # Return filename for saving

# Create an inference dataset and dataloader
inference_dataset = InferenceDataset(image_dir=image_dir, transforms=None)
data_loader_inference = DataLoader(inference_dataset, batch_size=1, shuffle=False, num_workers=4, collate_fn=collate_fn)

# ==============================
# 11. Save Predictions After Training
# ==============================

# Call the prediction function after training
save_predictions_as_images(model, data_loader_inference, device, idx_to_class, confidence_threshold=0.5, semantic_segmentation=False)
print(f"Predicted images saved to {predicted_images_dir}")

