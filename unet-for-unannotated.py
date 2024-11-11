# ================================
# Import Necessary Libraries
# ================================

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import models, transforms
import cv2
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, precision_recall_curve
from sklearn.preprocessing import label_binarize
import seaborn as sns

# ================================
# Define Dataset Classes
# ================================

class ImageDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        """
        Args:
            image_paths (list): List of image file paths.
            labels (list): List of corresponding labels.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load image
        img_path = self.image_paths[idx]
        image = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        if image is None:
            raise ValueError(f"Failed to load image: {img_path}")
        
        # Handle different number of channels
        if len(image.shape) == 2:
            # Grayscale to RGB
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:
            # RGBA to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
        else:
            # BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Apply transformations
        if self.transform:
            image = self.transform(image)
        
        label = self.labels[idx]
        return image, label

class SegmentationDataset(Dataset):
    def __init__(self, image_paths, masks, transform=None):
        """
        Args:
            image_paths (list): List of image file paths.
            masks (dict): Dictionary mapping image paths to their corresponding masks.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.image_paths = image_paths
        self.masks = masks
        self.transform = transform
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load image
        img_path = self.image_paths[idx]
        image = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        if image is None:
            raise ValueError(f"Failed to load image: {img_path}")
        
        # Handle different number of channels
        if len(image.shape) == 2:
            # Grayscale to RGB
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:
            # RGBA to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
        else:
            # BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Load mask
        mask = self.masks.get(img_path)
        if mask is None:
            raise ValueError(f"No mask found for image: {img_path}")
        
        # Ensure mask is single-channel
        if len(mask.shape) == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
        
        # Ensure image and mask have the same size
        if image.shape[:2] != mask.shape[:2]:
            # Resize image to match mask size
            image = cv2.resize(image, (mask.shape[1], mask.shape[0]), interpolation=cv2.INTER_LINEAR)
            mask = cv2.resize(mask, (mask.shape[1], mask.shape[0]), interpolation=cv2.INTER_NEAREST)
        
        # Apply transformations
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
        else:
            # Basic transformations without Albumentations
            transform_ops = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])
            image = transform_ops(image)
            mask = torch.from_numpy(mask)
        
        mask = mask.long()  # Ensure mask is torch.long
        
        return image, mask

# ================================
# Define Data Loading Function
# ================================

def load_dataset(data_dir, supported_extensions=('.png', '.jpg', '.jpeg', '.tiff', '.tif')):
    """
    Loads image paths and labels from the specified directory.

    Args:
        data_dir (str): Path to the dataset directory.
        supported_extensions (tuple): Supported image file extensions.

    Returns:
        image_paths (list): List of image file paths.
        labels (list): List of corresponding labels.
    """
    classes = sorted(os.listdir(data_dir))
    image_paths = []
    labels = []
    for idx, cls in enumerate(classes):
        cls_dir = os.path.join(data_dir, cls)
        if not os.path.isdir(cls_dir):
            print(f"Skipping {cls_dir} as it is not a directory.")
            continue
        for img_name in os.listdir(cls_dir):
            if img_name.lower().endswith(supported_extensions) and not img_name.startswith('.'):
                image_paths.append(os.path.join(cls_dir, img_name))
                labels.append(idx)
    return image_paths, labels

# ================================
# Define Transformation Pipelines
# ================================

# Define transformations for the classifier
transform_classifier = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet mean
                         std=[0.229, 0.224, 0.225])    # ImageNet std
])

# Define enhanced segmentation transformations with additional augmentations
def get_segmentation_transforms():
    return A.Compose([
        A.Resize(224, 224),  # Ensure both image and mask are resized to 224x224
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Rotate(limit=20, p=0.5),  # Increased rotation limit
        A.RandomBrightnessContrast(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=10, p=0.5),  # Additional augmentation
        A.ElasticTransform(p=0.2),  # Elastic transformation
        A.GaussianBlur(p=0.1),       # Gaussian Blur
        A.RandomGamma(p=0.1),         # Random Gamma Correction
        A.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

# ================================
# Load and Split the Dataset for Classification
# ================================

# Replace with your dataset path
data_dir = '/home/idrone2/Desktop/Kaggle_datasets/blood_cell/3C-BC'  # Replace with your dataset path
image_paths, labels = load_dataset(data_dir)

print(f"Total images found: {len(image_paths)}")

from sklearn.model_selection import train_test_split

# Split into training and validation sets
train_paths, val_paths, train_labels, val_labels = train_test_split(
    image_paths, labels, test_size=0.2, random_state=42, stratify=labels)

print(f"Classifier Training set size: {len(train_paths)}")
print(f"Classifier Validation set size: {len(val_paths)}")

# Create Dataset instances
train_dataset_classifier = ImageDataset(train_paths, train_labels, transform=transform_classifier)
val_dataset_classifier = ImageDataset(val_paths, val_labels, transform=transform_classifier)

# Create DataLoader instances
train_loader = DataLoader(train_dataset_classifier, batch_size=32, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset_classifier, batch_size=32, shuffle=False, num_workers=0)

# ================================
# Define and Train the Classifier
# ================================

class ClassifierModel(nn.Module):
    def __init__(self, num_classes=4):
        super(ClassifierModel, self).__init__()
        self.model = models.resnet18(pretrained=True)
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, num_classes)
        
    def forward(self, x):
        return self.model(x)

# Define device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Initialize Classifier
classifier = ClassifierModel(num_classes=4).to(device)

# Define Loss and Optimizer with Weight Decay (L2 Regularization)
criterion_classifier = nn.CrossEntropyLoss()
optimizer_classifier = optim.Adam(classifier.parameters(), lr=1e-4, weight_decay=1e-5)  # Added weight_decay

# Define Learning Rate Scheduler
scheduler_classifier = optim.lr_scheduler.ReduceLROnPlateau(optimizer_classifier, mode='min', factor=0.5, patience=3, verbose=True)

num_epochs_classifier = 10

# Lists to store metrics
train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []

for epoch in range(num_epochs_classifier):
    print(f"\nEpoch {epoch+1}/{num_epochs_classifier}")
    print("-" * 30)
    
    # ==========================
    # Training Phase
    # ==========================
    classifier.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for images, labels in tqdm(train_loader, desc='Training'):
        images = images.to(device)
        labels = labels.to(device)
        
        optimizer_classifier.zero_grad()
        outputs = classifier(images)
        loss = criterion_classifier(outputs, labels)
        loss.backward()
        optimizer_classifier.step()
        
        running_loss += loss.item() * images.size(0)
        
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    
    epoch_train_loss = running_loss / len(train_loader.dataset)
    epoch_acc = correct / total
    train_losses.append(epoch_train_loss)
    train_accuracies.append(epoch_acc)
    print(f"Training Loss: {epoch_train_loss:.4f} - Accuracy: {epoch_acc:.4f}")
    
    # ==========================
    # Validation Phase
    # ==========================
    classifier.eval()
    val_loss = 0.0
    correct_val = 0
    total_val = 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc='Validation'):
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = classifier(images)
            loss = criterion_classifier(outputs, labels)
            val_loss += loss.item() * images.size(0)
            
            _, preds = torch.max(outputs, 1)
            correct_val += (preds == labels).sum().item()
            total_val += labels.size(0)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    epoch_val_loss = val_loss / len(val_loader.dataset)
    epoch_val_acc = correct_val / total_val
    val_losses.append(epoch_val_loss)
    val_accuracies.append(epoch_val_acc)
    print(f"Validation Loss: {epoch_val_loss:.4f} - Accuracy: {epoch_val_acc:.4f}")
    
    # Step the scheduler
    scheduler_classifier.step(epoch_val_loss)

# ================================
# Plot Training and Validation Metrics
# ================================

epochs = range(1, num_epochs_classifier + 1)

plt.figure(figsize=(14, 6))

# Plot Loss
plt.subplot(1, 2, 1)
plt.plot(epochs, train_losses, 'bo-', label='Training Loss')
plt.plot(epochs, val_losses, 'ro-', label='Validation Loss')
plt.title('Classifier Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# Plot Accuracy
plt.subplot(1, 2, 2)
plt.plot(epochs, train_accuracies, 'bo-', label='Training Accuracy')
plt.plot(epochs, val_accuracies, 'ro-', label='Validation Accuracy')
plt.title('Classifier Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()

# ================================
# Evaluate Classifier with Metrics
# ================================

# Confusion Matrix
cm = confusion_matrix(all_labels, all_preds)
print("Confusion Matrix:")
print(cm)

# Classification Report
class_names = sorted(os.listdir(data_dir))
report = classification_report(all_labels, all_preds, target_names=class_names)
print("Classification Report:")
print(report)

# Plot Confusion Matrix
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Reds',
            xticklabels=class_names,
            yticklabels=class_names)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Classifier Confusion Matrix')
plt.show()

# ROC Curves and AUC for each class

# Binarize the labels for ROC
y_true = label_binarize(all_labels, classes=list(range(len(class_names))))
y_pred_prob = []

# Collect predicted probabilities
classifier.eval()
with torch.no_grad():
    for images, _ in tqdm(val_loader, desc='Collecting Predicted Probabilities for ROC'):
        images = images.to(device)
        outputs = classifier(images)
        probs = torch.softmax(outputs, dim=1)
        y_pred_prob.extend(probs.cpu().numpy())

y_pred_prob = np.array(y_pred_prob)
n_classes = y_true.shape[1]

fpr = dict()
tpr = dict()
roc_auc = dict()

plt.figure(figsize=(10, 8))
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_true[:, i], y_pred_prob[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
    plt.plot(fpr[i], tpr[i], lw=2, label=f'ROC curve for class {class_names[i]} (AUC = {roc_auc[i]:0.2f})')

plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([-0.05, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curves')
plt.legend(loc="lower right")
plt.show()

# Precision-Recall Curves and AUC for each class
precision = dict()
recall = dict()
prc_auc = dict()

plt.figure(figsize=(10, 8))
for i in range(n_classes):
    precision[i], recall[i], _ = precision_recall_curve(y_true[:, i], y_pred_prob[:, i])
    prc_auc[i] = auc(recall[i], precision[i])
    plt.plot(recall[i], precision[i], lw=2, label=f'PRC curve for class {class_names[i]} (AUC = {prc_auc[i]:0.2f})')

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall (PR) Curves')
plt.legend(loc="lower left")
plt.show()

# ================================
# Initialize GradCAM and Generate CAMs
# ================================

# Define target layer for GradCAM (last convolutional layer in ResNet-18)
target_layer = classifier.model.layer4

# Initialize GradCAM
cam = GradCAM(model=classifier.model, target_layers=[target_layer])

# Updated generate_cam function
def generate_cam(image_path, model, cam, device, class_idx, class_name, transform):
    """
    Generates a Class Activation Map (CAM) for a given image and class.

    Args:
        image_path (str): Path to the input image.
        model (nn.Module): Trained classifier model.
        cam (GradCAM): Initialized GradCAM object.
        device (torch.device): Device to perform computations on.
        class_idx (int): Class index for which to generate CAM.
        class_name (str): Class name for the class index.
        transform (callable): Transformation to apply to the input image.

    Returns:
        cam_image (numpy.ndarray): Image with CAM overlay.
    """
    # Load and preprocess the image
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"Failed to load image for CAM: {image_path}")
    
    # Handle different number of channels
    if len(image.shape) == 2:
        # Grayscale to RGB
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 4:
        # RGBA to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
    else:
        # BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Resize the image to match the model input size
    image_resized = cv2.resize(image, (224, 224))
    
    # Prepare the input tensor for the model
    input_tensor = transform(image_resized).unsqueeze(0).to(device)
    
    # Define the target for CAM (the predicted class)
    target = [ClassifierOutputTarget(class_idx)]
    
    # Generate CAM
    grayscale_cam = cam(input_tensor=input_tensor, targets=target)
    grayscale_cam = grayscale_cam[0, :]
    
    # Overlay CAM on the resized image
    cam_image = show_cam_on_image(image_resized / 255.0, grayscale_cam, use_rgb=True)
    
    # Add class name to the CAM image
    cv2.putText(cam_image, class_name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    
    return cam_image

# Generate and display CAM images for a few validation images
num_cam_images = 4  # Number of CAM images to generate
selected_val_paths = val_paths[:num_cam_images]  # Select first few images

for img_path in selected_val_paths:
    # Generate CAM for the image
    try:
        # Get the predicted class
        classifier.eval()
        with torch.no_grad():
            # Load and preprocess the image
            image = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
            if image is None:
                print(f"Failed to load image for CAM: {img_path}")
                continue
            
            # Handle different number of channels
            if len(image.shape) == 2:
                # Grayscale to RGB
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            elif image.shape[2] == 4:
                # RGBA to RGB
                image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
            else:
                # BGR to RGB
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Resize the image to match the model input size
            image_resized = cv2.resize(image, (224, 224))
            
            # Prepare the input tensor for the model
            input_tensor = transform_classifier(image_resized).unsqueeze(0).to(device)
            
            outputs = classifier(input_tensor)
            _, preds = torch.max(outputs, 1)
            class_idx = preds.item()
            class_name = class_names[class_idx]
        
        # Generate CAM
        cam_image = generate_cam(img_path, classifier, cam, device, class_idx, class_name, transform_classifier)
        
        # Display the CAM image
        plt.figure(figsize=(6, 6))
        plt.imshow(cam_image)
        plt.title(f'CAM for {class_name}')
        plt.axis('off')
        plt.show()
    except Exception as e:
        print(f"Error generating CAM for {img_path}: {e}")

# ================================
# Generate Pseudo-Masks for the Validation Set
# ================================

def create_pseudo_mask(image_path, model, cam, device, num_classes=4, threshold=0.2, transform=None):
    """
    Creates a pseudo-mask for a given image by combining CAMs from all classes.

    Args:
        image_path (str): Path to the input image.
        model (nn.Module): Trained classifier model.
        cam (GradCAM): Initialized GradCAM object.
        device (torch.device): Device to perform computations on.
        num_classes (int): Number of classes.
        threshold (float): Threshold for binarizing CAMs.
        transform (callable): Transformation to apply to the input image.

    Returns:
        mask (numpy.ndarray): Generated pseudo-mask.
    """
    # Initialize an empty mask with the same size as the resized image
    mask = np.zeros((224, 224), dtype=np.uint8)
    
    # Load and preprocess the image
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"Failed to load image for CAM: {image_path}")
    
    # Handle different number of channels
    if len(image.shape) == 2:
        # Grayscale to RGB
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 4:
        # RGBA to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
    else:
        # BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Resize the image to match the model input size
    image_resized = cv2.resize(image, (224, 224))
    
    # Prepare the input tensor for the model
    input_tensor = transform(image_resized).unsqueeze(0).to(device)
    
    for cls in range(num_classes):
        target = [ClassifierOutputTarget(cls)]
        grayscale_cam = cam(input_tensor=input_tensor, targets=target)
        grayscale_cam = grayscale_cam[0, :]
        cam_gray = (grayscale_cam * 255).astype(np.uint8)
        _, cam_binary = cv2.threshold(cam_gray, int(threshold * 255), 255, cv2.THRESH_BINARY)
        
        # Assign class label where CAM is active
        mask[cam_binary > 0] = cls
        
    return mask

# Generate pseudo-masks
pseudo_masks = {}
for img_path in tqdm(val_paths, desc='Generating Pseudo-Masks'):
    try:
        mask = create_pseudo_mask(img_path, classifier, cam, device, num_classes=4, threshold=0.2, transform=transform_classifier)
        pseudo_masks[img_path] = mask
    except Exception as e:
        print(f"Error processing {img_path}: {e}")

print(f"Total pseudo-masks generated: {len(pseudo_masks)}")

# ================================
# Define the U-Net Model
# ================================

class UNet(nn.Module):
    def __init__(self, num_classes=4, in_channels=3, dropout_rate=0.7):
        super(UNet, self).__init__()
        def CBR(in_ch, out_ch, dropout=0.7):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout)  # Increased dropout rate
            )
        
        # Encoder
        self.enc1 = CBR(in_channels, 64)
        self.enc2 = CBR(64, 128)
        self.enc3 = CBR(128, 256)
        self.enc4 = CBR(256, 512)
        
        self.pool = nn.MaxPool2d(2)
        
        # Bottleneck
        self.bottleneck = CBR(512, 1024, dropout=0.7)
        
        # Decoder with additional Dropout
        self.upconv4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.dec4 = CBR(1024, 512, dropout=0.7)
        self.dropout4 = nn.Dropout(0.5)
        
        self.upconv3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = CBR(512, 256, dropout=0.7)
        self.dropout3 = nn.Dropout(0.5)
        
        self.upconv2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = CBR(256, 128, dropout=0.7)
        self.dropout2 = nn.Dropout(0.5)
        
        self.upconv1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = CBR(128, 64, dropout=0.7)
        self.dropout1 = nn.Dropout(0.5)
        
        # Final Convolution
        self.conv_final = nn.Conv2d(64, num_classes, kernel_size=1)
        
    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        p1 = self.pool(e1)
        
        e2 = self.enc2(p1)
        p2 = self.pool(e2)
        
        e3 = self.enc3(p2)
        p3 = self.pool(e3)
        
        e4 = self.enc4(p3)
        p4 = self.pool(e4)
        
        # Bottleneck
        b = self.bottleneck(p4)
        
        # Decoder
        up4 = self.upconv4(b)
        up4 = torch.cat([up4, e4], dim=1)
        d4 = self.dec4(up4)
        d4 = self.dropout4(d4)
        
        up3 = self.upconv3(d4)
        up3 = torch.cat([up3, e3], dim=1)
        d3 = self.dec3(up3)
        d3 = self.dropout3(d3)
        
        up2 = self.upconv2(d3)
        up2 = torch.cat([up2, e2], dim=1)
        d2 = self.dec2(up2)
        d2 = self.dropout2(d2)
        
        up1 = self.upconv1(d2)
        up1 = torch.cat([up1, e1], dim=1)
        d1 = self.dec1(up1)
        d1 = self.dropout1(d1)
        
        out = self.conv_final(d1)
        return out

# ================================
# Define Loss Functions
# ================================

class DiceLossSegmentation(nn.Module):
    def __init__(self, smooth=1):
        super(DiceLossSegmentation, self).__init__()
        self.smooth = smooth
        
    def forward(self, inputs, targets):
        """
        Args:
            inputs (torch.Tensor): Predicted logits (N, C, H, W).
            targets (torch.Tensor): Ground truth labels (N, H, W).

        Returns:
            torch.Tensor: Dice loss.
        """
        inputs = torch.softmax(inputs, dim=1)
        targets_one_hot = nn.functional.one_hot(targets, num_classes=inputs.shape[1]).permute(0, 3, 1, 2).float()
        
        intersection = (inputs * targets_one_hot).sum(dim=(2,3))
        dice = (2. * intersection + self.smooth) / (inputs.sum(dim=(2,3)) + targets_one_hot.sum(dim=(2,3)) + self.smooth)
        return 1 - dice.mean()

class CombinedLossSegmentation(nn.Module):
    def __init__(self, ce_weight=None, dice_weight=1.0):
        super(CombinedLossSegmentation, self).__init__()
        self.ce = nn.CrossEntropyLoss(weight=ce_weight)
        self.dice = DiceLossSegmentation()
        self.dice_weight = dice_weight
        
    def forward(self, inputs, targets):
        ce_loss = self.ce(inputs, targets)
        dice_loss = self.dice(inputs, targets)
        return ce_loss + self.dice_weight * dice_loss

# ================================
# Initialize and Train the U-Net
# ================================

# Define segmentation transformations
seg_transforms = get_segmentation_transforms()

# Create Segmentation Dataset
unet_dataset = SegmentationDataset(val_paths, pseudo_masks, transform=seg_transforms)

# Split into training and validation subsets (e.g., 80% train, 20% val)
train_size = int(0.8 * len(unet_dataset))
val_size = len(unet_dataset) - train_size
train_unet, val_unet = random_split(unet_dataset, [train_size, val_size])

print(f"U-Net Training set size: {len(train_unet)}")
print(f"U-Net Validation set size: {len(val_unet)}")

# Create DataLoaders for U-Net
unet_train_loader = DataLoader(train_unet, batch_size=16, shuffle=True, num_workers=0)
unet_val_loader = DataLoader(val_unet, batch_size=16, shuffle=False, num_workers=0)

# Initialize U-Net model
unet_model = UNet(num_classes=4, in_channels=3, dropout_rate=0.7).to(device)

# Define loss function and optimizer with Weight Decay (L2 Regularization)
# You can adjust weight_decay and dice_weight as needed
criterion_unet = CombinedLossSegmentation(ce_weight=None, dice_weight=1.0)
optimizer_unet = optim.Adam(unet_model.parameters(), lr=1e-4, weight_decay=1e-5)  # Added weight_decay

# Define Learning Rate Scheduler
scheduler_unet = optim.lr_scheduler.ReduceLROnPlateau(optimizer_unet, mode='min', factor=0.5, patience=3, verbose=True)

# Define Early Stopping Parameters
early_stopping_patience = 15  # Increased patience
best_val_loss = float('inf')
epochs_no_improve = 0
best_model_wts = None

# Define number of epochs
num_epochs_unet = 100  # Increased epochs to allow early stopping to act

# Lists to store U-Net metrics
unet_train_losses = []
unet_val_losses = []

for epoch in range(num_epochs_unet):
    print(f"\nEpoch {epoch+1}/{num_epochs_unet}")
    print("-" * 30)
    
    # ==========================
    # Training Phase
    # ==========================
    unet_model.train()
    running_loss = 0.0
    for images, masks in tqdm(unet_train_loader, desc='Training'):
        images = images.to(device)
        masks = masks.to(device)
        
        optimizer_unet.zero_grad()
        outputs = unet_model(images)
        loss = criterion_unet(outputs, masks)
        loss.backward()
        optimizer_unet.step()
        
        running_loss += loss.item() * images.size(0)
    
    epoch_train_loss = running_loss / len(unet_train_loader.dataset)
    unet_train_losses.append(epoch_train_loss)
    print(f"Training Loss: {epoch_train_loss:.4f}")
    
    # ==========================
    # Validation Phase
    # ==========================
    unet_model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for images, masks in tqdm(unet_val_loader, desc='Validation'):
            images = images.to(device)
            masks = masks.to(device)
            
            outputs = unet_model(images)
            loss = criterion_unet(outputs, masks)
            
            val_loss += loss.item() * images.size(0)
    
    epoch_val_loss = val_loss / len(unet_val_loader.dataset)
    unet_val_losses.append(epoch_val_loss)
    print(f"Validation Loss: {epoch_val_loss:.4f}")
    
    # Step the scheduler
    scheduler_unet.step(epoch_val_loss)
    
    # Check for improvement
    if epoch_val_loss < best_val_loss:
        best_val_loss = epoch_val_loss
        epochs_no_improve = 0
        best_model_wts = unet_model.state_dict()
        print("Validation loss improved. Saving best model.")
    else:
        epochs_no_improve += 1
        print(f"No improvement in validation loss for {epochs_no_improve} epochs.")
    
    # Early stopping
    if epochs_no_improve >= early_stopping_patience:
        print("Early stopping triggered.")
        break

# Load best model weights
if best_model_wts is not None:
    unet_model.load_state_dict(best_model_wts)
    print("Loaded best model weights.")

# ================================
# Plot U-Net Training and Validation Loss
# ================================

epochs_trained = range(1, len(unet_train_losses) + 1)

plt.figure(figsize=(10, 5))
plt.plot(epochs_trained, unet_train_losses, 'bo-', label='Training Loss')
plt.plot(epochs_trained, unet_val_losses, 'ro-', label='Validation Loss')
plt.title('U-Net Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# ================================
# Define Prediction Function for Segmentation
# ================================

def predict_segmentation(model, image_path, device):
    """
    Predicts segmentation mask for a given image using the trained U-Net model.

    Args:
        model (nn.Module): Trained U-Net model.
        image_path (str): Path to the input image.
        device (torch.device): Device to perform computations on.

    Returns:
        mask (numpy.ndarray): Predicted segmentation mask resized to original image size.
    """
    model.eval()
    
    # Load and preprocess image
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"Failed to load image for prediction: {image_path}")
    
    # Handle different number of channels
    if len(image.shape) == 2:
        # Grayscale to RGB
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 4:
        # RGBA to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
    else:
        # BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Resize the image to match the model input size
    image_resized = cv2.resize(image, (224, 224))
    
    # Define transformation
    transform_ops = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    
    # Apply transformation
    input_tensor = transform_ops(image_resized).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(input_tensor)
        preds = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()
    
    # Resize mask to original image size
    mask = cv2.resize(preds.astype(np.uint8), (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
    return mask

# ================================
# Predict and Visualize Segmentation on a New Image
# ================================

# Replace with the path to your new image
new_image_path = '/home/idrone2/Desktop/Kaggle_datasets/blood_cell/3C-BC/LYT/LYT_0098.tiff'  # Replace with your image path

# Generate predicted mask
predicted_mask = predict_segmentation(unet_model, new_image_path, device)

# Visualization
original_image = cv2.imread(new_image_path, cv2.IMREAD_UNCHANGED)
if original_image is None:
    raise ValueError(f"Failed to load image for visualization: {new_image_path}")

# Handle different number of channels for visualization
if len(original_image.shape) == 2:
    # Grayscale to RGB
    original_image = cv2.cvtColor(original_image, cv2.COLOR_GRAY2RGB)
elif original_image.shape[2] == 4:
    # RGBA to RGB
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGRA2RGB)
else:
    # BGR to RGB
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(original_image)
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(original_image)
plt.imshow(predicted_mask, cmap='jet', alpha=0.5)
plt.title('Predicted Segmentation Mask')
plt.axis('off')

plt.show()

# ================================
# Save and Load Models (Optional)
# ================================

# Save the classifier and U-Net models
torch.save(classifier.state_dict(), 'classifier.pth')
torch.save(unet_model.state_dict(), 'unet.pth')

# To load the models later:
# classifier.load_state_dict(torch.load('classifier.pth'))
# unet_model.load_state_dict(torch.load('unet.pth'))

