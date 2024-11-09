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
from sklearn.metrics import confusion_matrix, classification_report
import shutil
import random
from PIL import Image, ImageDraw

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
    'early tmb': 1,
    'late tmb': 2,
    'looper': 3
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

                    x_min, x_max = min(all_points_x), max(all_points_x)
                    y_min, y_max = min(all_points_y), max(all_points_y)
                    boxes.append([x_min, y_min, x_max, y_max])

                elif shape_attrs.get('name') == 'rect':
                    x = shape_attrs.get('x', 0)
                    y = shape_attrs.get('y', 0)
                    width = shape_attrs.get('width', 0)
                    height = shape_attrs.get('height', 0)
                    x_min, y_min = x, y
                    x_max, y_max = x + width, y + height
                    boxes.append([x_min, y_min, x_max, y_max])
                    masks.append(self.create_rect_mask(x_min, y_min, x_max, y_max, img_width, img_height))

        if boxes:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)
            masks = torch.as_tensor(np.stack(masks), dtype=torch.uint8)
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

image_dir = '/home/idrone2/Desktop/ANNOTATED_LEAF'            # Adjust the path to your images folder
annotation_file = '/home/idrone2/Desktop/labels_tea-pest_2024-10-21-02-31-24.json'  # Adjust the path to your annotations file

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

model = maskrcnn_resnet50_fpn(pretrained=True)

in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)

# ==============================
# 6. Training and Validation Functions
# ==============================


def train_one_epoch(model, optimizer, data_loader, device, epoch, train_losses, print_freq=10):
    model.train()
    running_loss = 0.0
    for i, (images, targets) in enumerate(data_loader):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        running_loss += losses.item()
        if i % print_freq == 0:
            print(f"Epoch [{epoch}], Iteration [{i}/{len(data_loader)}], Loss: {losses.item():.4f}")

    train_losses.append(running_loss / len(data_loader))

def validate_one_epoch(model, data_loader, device, val_losses):
    model.train()  # Keep the model in train mode for loss calculation
    running_val_loss = 0.0

    with torch.no_grad():
        for images, targets in data_loader:
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            # Calculate validation loss
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            running_val_loss += losses.item()

    val_losses.append(running_val_loss / len(data_loader))

# ==============================
# 7. Training Loop
# ==============================

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)

num_epochs = 10
train_losses = []
val_losses = []

params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

for epoch in range(num_epochs):
    print(f"Starting epoch {epoch + 1} / {num_epochs}")

    # Training
    train_one_epoch(model, optimizer, data_loader, device, epoch, train_losses)

    # Validation (Calculate validation loss without updating weights)
    validate_one_epoch(model, data_loader_test, device, val_losses)

    # Save model after every epoch
    torch.save(model.state_dict(), os.path.join(output_dir, f"maskrcnn_epoch_{epoch+1}.pth"))

# ==============================
# 8. Save Loss Graphs
# ==============================

# Plot both training and validation loss
plt.figure(figsize=(10, 5))

# Plot Training Loss
plt.subplot(1, 2, 1)
plt.plot(range(1, num_epochs+1), train_losses, label='Training Loss', color='blue')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training Loss Over Epochs')
plt.legend()

# Plot Validation Loss
plt.subplot(1, 2, 2)
plt.plot(range(1, num_epochs+1), val_losses, label='Validation Loss', color='orange')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Validation Loss Over Epochs')
plt.legend()

plt.savefig(os.path.join(output_dir, "validation_loss.png"))

# ==============================
# 9. Prediction and Visualization
# ==============================

def save_predictions_as_images(model, data_loader, device, idx_to_class, semantic_segmentation=False):
    model.eval()
    with torch.no_grad():
        for idx, (images, _) in enumerate(data_loader):
            images = list(img.to(device) for img in images)
            outputs = model(images)

            if len(outputs) == 0 or len(outputs[0]['boxes']) == 0:
                print(f"No predictions found for image {idx+1}. Skipping.")
                continue

            # Convert the first image to PIL for visualization
            image = images[0].cpu().mul(255).permute(1, 2, 0).byte().numpy()
            image_pil = Image.fromarray(image).convert("RGB")

            # Create drawing interface for the image
            draw = ImageDraw.Draw(image_pil)

            # Get output masks and labels
            boxes = outputs[0]['boxes'].cpu().numpy()
            labels = outputs[0]['labels'].cpu().numpy()
            masks = outputs[0]['masks'].cpu().numpy()  # (N, 1, H, W)

            # Threshold masks to create binary masks
            masks = masks > 0.5

            # Instance segmentation: Draw masks and boxes
            for i, box in enumerate(boxes):
                label = labels[i]
                class_name = idx_to_class.get(label, "Unknown")

                # Draw the bounding box with thicker lines
                draw.rectangle(box.tolist(), outline="red", width=5)  # Thicker lines
                font = ImageFont.truetype("arial.ttf", 24)  # Larger font
                draw.text((box[0], box[1]), f"{class_name}", fill="red", font=font)  # Larger labels

                # Apply mask (instance segmentation) with transparency
                mask = masks[i, 0]
                mask_image = Image.fromarray(mask.astype(np.uint8) * 255, mode='L').convert("RGBA")
                mask_image.putalpha(128)  # Set transparency

                # Paste the mask on the image with transparency
                image_pil.paste(mask_image, mask=mask_image)

            # Save the instance segmentation result
            image_pil.save(os.path.join(predicted_images_dir, f"instance_segmentation_{idx+1}.png"))

            if semantic_segmentation:
                # For semantic segmentation, accumulate masks by class
                semantic_mask = np.zeros_like(masks[0, 0], dtype=np.uint8)
                for i, mask in enumerate(masks):
                    label = labels[i]
                    semantic_mask[mask[0]] = label

                # Create an image from the semantic mask
                semantic_image = Image.fromarray(semantic_mask).convert("RGB")
                semantic_image.save(os.path.join(predicted_images_dir, f"semantic_segmentation_{idx+1}.png"))

