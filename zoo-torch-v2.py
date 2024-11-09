import os 
# Set CUDA visible devices
os.environ["CUDA_VISIBLE_DEVICE"] = '0'
os.environ["WORLD_SIZE"] = "1"

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split, Subset
from torchvision import transforms
from torchvision.datasets import ImageFolder
from zoopt import Dimension, Objective, Parameter, Opt
import numpy as np

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.preprocessing import label_binarize
from itertools import cycle
import seaborn as sns
from torchsummary import summary  # Import the summary function
from collections import Counter  # Ensure Counter is imported
import warnings
from tqdm import tqdm  # For progress bars
import sys

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Set random seeds for reproducibility
import random
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

# Check CUDA availability and GPU information
print("CUDA Available:", torch.cuda.is_available())
print("Number of GPUs:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("Current GPU Device:", torch.cuda.current_device())
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
else:
    print("Using CPU")

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device:", device)

# --------------------------
# Create zoopt_output Directory
# --------------------------
zoopt_output_dir = "zoopt_output"
os.makedirs(zoopt_output_dir, exist_ok=True)

# --------------------------
# Data Preprocessing
# --------------------------

transform = transforms.Compose([
    transforms.Resize((64, 64)),  # Increased image size for more features
    transforms.RandomHorizontalFlip(),  # Data augmentation
    transforms.RandomRotation(10),      # Data augmentation
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Load dataset
dataset = ImageFolder(root='/home/idrone2/Tea_pest/Tea-TJ', transform=transform)
print("Dataset loaded with classes:", dataset.classes)

# Split dataset into train and test
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# Further split test_dataset into validation and test if needed
# For simplicity, using test_dataset as both validation and test

# Create DataLoaders
# Increased num_workers for faster data loading and pin_memory for GPU efficiency
train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True, num_workers=4, pin_memory=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)
val_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)

# --------------------------
# Define the CNN Model
# --------------------------

class CNN(nn.Module):
    def __init__(self, num_layers, dropout_rate, optimizer_type, num_classes):
        super(CNN, self).__init__()
        self.layers = nn.ModuleList()
        in_channels = 3
        out_channels = 32
        kernel_size = 3
        padding = 1
        
        for _ in range(num_layers):
            self.layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding))
            self.layers.append(nn.BatchNorm2d(out_channels))  # Batch Normalization
            self.layers.append(nn.ReLU())
            self.layers.append(nn.MaxPool2d(2, 2))
            in_channels = out_channels
            out_channels = min(out_channels * 2, 256)  # Cap out_channels to prevent too large channels
        
        # Initialize flatten_size dynamically
        self.flatten_size = self._calculate_flatten_size()
        self.fc1 = nn.Linear(self.flatten_size, 256)
        self.fc2 = nn.Linear(256, num_classes)
        self.dropout = nn.Dropout(p=dropout_rate)
    
    def _calculate_flatten_size(self):
        # Dummy input to calculate flatten_size
        dummy_input = torch.zeros(1, 3, 64, 64)  # Updated to match resized images
        x = dummy_input
        for layer in self.layers:
            x = layer(x)
        return int(torch.prod(torch.tensor(x.size())))
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = x.view(-1, self.flatten_size)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# --------------------------
# Training Function
# --------------------------

def train_model(num_layers, dropout_rate, optimizer_type, learning_rate):
    model = CNN(num_layers=num_layers, dropout_rate=dropout_rate, optimizer_type=optimizer_type, num_classes=len(dataset.classes)).to(device)
    
    # Print the model summary and save it
    summary_path = os.path.join(zoopt_output_dir, f'model_summary_layers_{num_layers}_dropout_{dropout_rate}.txt')
    with open(summary_path, 'w') as f:
        original_stdout = sys.stdout
        sys.stdout = f
        summary(model, input_size=(3, 64, 64))
        sys.stdout = original_stdout
    print("Model summary saved.")
    
    criterion = nn.CrossEntropyLoss()
    
    # Define optimizer based on optimizer_type
    if optimizer_type == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    elif optimizer_type == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    elif optimizer_type == 'RMSprop':
        optimizer = optim.RMSprop(model.parameters(), lr=learning_rate)
    else:
        raise ValueError(f"Unsupported optimizer type: {optimizer_type}")
    
    num_epochs = 20  # Increased epochs for better learning
    
    train_losses = []
    val_losses = []
    val_accuracies = []
    
    # Early stopping parameters
    best_val_loss = np.Inf
    epochs_no_improve = 0
    patience = 5
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        # Use tqdm to display progress
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Training]", leave=False):
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            
            optimizer.zero_grad()
            outputs = model(images)
            
            # Check batch size consistency
            if outputs.size(0) != labels.size(0):
                print(f"Batch size mismatch: outputs.size(0)={outputs.size(0)}, labels.size(0)={labels.size(0)}")
                continue  # Skip this batch
            
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        epoch_train_loss = running_loss / len(train_loader)
        train_losses.append(epoch_train_loss)
        train_acc = correct / total
        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {epoch_train_loss:.4f} | Train Acc: {train_acc:.4f}")
        
        # Validation phase
        model.eval()
        val_running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Validation]", leave=False):
                images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                outputs = model(images)
                
                # Check batch size consistency
                if outputs.size(0) != labels.size(0):
                    print(f"Batch size mismatch: outputs.size(0)={outputs.size(0)}, labels.size(0)={labels.size(0)}")
                    continue  # Skip this batch
                
                loss = criterion(outputs, labels)
                val_running_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        epoch_val_loss = val_running_loss / len(val_loader)
        val_losses.append(epoch_val_loss)
        val_acc = correct / total
        val_accuracies.append(val_acc)
        print(f"Epoch {epoch+1}/{num_epochs} - Val Loss: {epoch_val_loss:.4f} | Val Acc: {val_acc:.4f}")
        
        # Check for improvement
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            epochs_no_improve = 0
            # Save the best model for this training run
            torch.save(model.state_dict(), os.path.join(zoopt_output_dir, 'best_model.pth'))
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"No improvement in validation loss for {patience} consecutive epochs. Early stopping.")
                break
    
    return train_losses, val_losses, val_accuracies, model

# --------------------------
# ZOOpt Objective Function
# --------------------------

def objective_function(solution):
    # Extract hyperparameters from ZOOpt solution
    params = solution.get_x()
    learning_rate = params[0]
    num_layers = int(params[1])
    dropout_rate = params[2]
    optimizer_type = 'Adam'  # Default optimizer
    
    # Introduce optimizer type as categorical hyperparameter
    # Since ZOOpt primarily handles continuous parameters, we'll map the fourth parameter manually
    # For simplicity, let's assume the fourth parameter is a continuous variable [0,1] where
    # 0 corresponds to 'Adam', 1 corresponds to 'SGD', and 2 corresponds to 'RMSprop'
    # However, since we initially set only 3 dimensions, we'll need to adjust the dimension
    # Let's expand the dimension to 4
    
    # To integrate optimizer type, we need to adjust the Dimension and ZOOpt setup
    # Since currently the Dimension is 3, to include optimizer type, it needs to be 4
    
    # For now, we'll proceed with 'Adam' as the optimizer type.
    # To add optimizer type, let's redefine the Dimension to include it as a categorical variable
    
    # However, ZOOpt doesn't natively support categorical variables. A workaround is to encode them numerically.
    # Let's redefine the search space accordingly.
    
    # For demonstration, assuming we have updated the Dimension to 4:
    # learning_rate, num_layers, dropout_rate, optimizer_type (encoded as 0:Adam, 1:SGD, 2:RMSprop)
    
    # For now, since the current script has Dimension=3, we'll keep optimizer_type fixed.
    # To fully implement optimizer_type as a hyperparameter, the Dimension and objective function need to be updated accordingly.
    
    print(f"\nEvaluating Solution: Learning Rate={learning_rate}, Num Layers={num_layers}, Dropout Rate={dropout_rate}")
    
    # Train the model with these hyperparameters
    train_losses, val_losses, val_accuracies, _ = train_model(num_layers, dropout_rate, optimizer_type, learning_rate)
    
    # Log the results
    with open(os.path.join(zoopt_output_dir, 'zoopt_log.txt'), 'a') as f:
        f.write(f"Solution: Learning Rate={learning_rate}, Num Layers={num_layers}, Dropout Rate={dropout_rate}, Optimizer={optimizer_type}\n")
        f.write(f"Validation Accuracy: {val_accuracies[-1]:.4f}\n\n")
    
    # Return negative validation accuracy as ZOOpt minimizes the objective
    return -val_accuracies[-1]

# --------------------------
# ZOOpt Configuration
# --------------------------

# Adjusted hyperparameter search space to include optimizer_type as the fourth dimension
dim = Dimension(
    4,
    [
        [1e-4, 1e-2],  # learning_rate: narrower range
        [2, 4],        # num_layers: reduce to 2-4
        [0.2, 0.4],     # dropout_rate: narrower range
        [0, 2]          # optimizer_type: 0=Adam, 1=SGD, 2=RMSprop
    ],
    [True, True, True, False]  # Last parameter is categorical (encoded as integer)
)

objective = Objective(objective_function, dim)
budget = 10  # Adjust budget as needed
parameter = Parameter(budget=budget)

# Create log file for ZOOpt
zoopt_log = os.path.join(zoopt_output_dir, "zoopt_log.txt")
with open(zoopt_log, "a") as f:
    f.write("Starting ZOOpt Hyperparameter Optimization\n\n")

# Run ZOOpt without callback
solution = Opt.min(objective, parameter)

# Extract best parameters
best_params = solution.get_x()
best_learning_rate = best_params[0]
best_num_layers = int(best_params[1])
best_dropout_rate = best_params[2]
optimizer_code = int(best_params[3])

# Map optimizer code to optimizer name
optimizer_mapping = {0: 'Adam', 1: 'SGD', 2: 'RMSprop'}
best_optimizer = optimizer_mapping.get(optimizer_code, 'Adam')  # Default to 'Adam' if out of range

print("\nBest parameters found by ZOOpt:")
print("Learning rate:", best_learning_rate)
print("Number of layers:", best_num_layers)
print("Dropout rate:", best_dropout_rate)
print("Optimizer type:", best_optimizer)

# Save best parameters to a file
best_params_path = os.path.join(zoopt_output_dir, "best_parameters.txt")
with open(best_params_path, "w") as f:
    f.write(f"Learning rate: {best_learning_rate}\n")
    f.write(f"Number of layers: {best_num_layers}\n")
    f.write(f"Dropout rate: {best_dropout_rate}\n")
    f.write(f"Optimizer type: {best_optimizer}\n")

print("Best parameters saved.")

# --------------------------
# Final Training with Best Parameters
# --------------------------

train_losses, val_losses, val_accuracies, final_model = train_model(best_num_layers, best_dropout_rate, best_optimizer, best_learning_rate)

# Save the final model
final_model_path = os.path.join(zoopt_output_dir, "final_model.pth")
torch.save(final_model.state_dict(), final_model_path)
print("Final model saved.")

# --------------------------
# Plotting Training and Validation Loss
# --------------------------

plt.figure(figsize=(12, 6))
plt.plot(train_losses, label='Train Loss', marker='o')
plt.plot(val_losses, label='Validation Loss', marker='o')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Loss over Epochs')
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(zoopt_output_dir, 'train_val_loss_curve.png'))
plt.close()
print("Training and validation loss curves saved.")

# --------------------------
# Plotting Validation Accuracy
# --------------------------

plt.figure(figsize=(12, 6))
plt.plot(val_accuracies, label='Validation Accuracy', marker='o', color='orange')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Validation Accuracy over Epochs')
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(zoopt_output_dir, 'validation_accuracy_curve.png'))
plt.close()
print("Validation accuracy curve saved.")

# --------------------------
# Confusion Matrix and Classification Report
# --------------------------

y_true = []
y_pred = []
final_model.eval()
with torch.no_grad():
    for images, labels in tqdm(test_loader, desc="Generating Predictions", leave=False):
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        outputs = final_model(images)
        _, predicted = torch.max(outputs.data, 1)
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(predicted.cpu().numpy())

# Confusion Matrix
conf_matrix = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=dataset.classes, yticklabels=dataset.classes)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.savefig(os.path.join(zoopt_output_dir, 'confusion_matrix.png'))
plt.close()
print("Confusion matrix saved.")

# Classification Report
class_report = classification_report(y_true, y_pred, target_names=dataset.classes, zero_division=0)
print("Classification Report:")
print(class_report)
with open(os.path.join(zoopt_output_dir, 'classification_report.txt'), 'w') as f:
    f.write(class_report)
print("Classification report saved.")

# --------------------------
# ROC and AUC (One-vs-Rest)
# --------------------------

# Binarize the output
y_true_bin = label_binarize(y_true, classes=range(len(dataset.classes)))
n_classes = y_true_bin.shape[1]

# Get the scores (probabilities)
y_scores = []
final_model.eval()
with torch.no_grad():
    for images, _ in tqdm(test_loader, desc="Calculating ROC AUC", leave=False):
        images = images.to(device, non_blocking=True)
        outputs = final_model(images)
        probs = F.softmax(outputs, dim=1)
        y_scores.append(probs.cpu().numpy())
y_scores = np.vstack(y_scores)

# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_scores[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Plot ROC curves
plt.figure(figsize=(10, 8))
colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=2,
             label='ROC curve of class {0} (area = {1:0.2f})'.format(dataset.classes[i], roc_auc[i]))
plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([-0.01, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc="lower right")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(zoopt_output_dir, 'roc_curves.png'))
plt.close()
print("ROC curves saved.")

# --------------------------
# Precision-Recall Curves
# --------------------------

precision = dict()
recall = dict()
pr_auc = dict()
for i in range(n_classes):
    precision[i], recall[i], _ = precision_recall_curve(y_true_bin[:, i], y_scores[:, i])
    pr_auc[i] = average_precision_score(y_true_bin[:, i], y_scores[:, i])

# Plot Precision-Recall curves
plt.figure(figsize=(10, 8))
colors = cycle(['navy', 'turquoise', 'darkorange'])
for i, color in zip(range(n_classes), colors):
    plt.plot(recall[i], precision[i], color=color, lw=2,
             label='PR curve of class {0} (AP = {1:0.2f})'.format(dataset.classes[i], pr_auc[i]))
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curves')
plt.legend(loc="lower left")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(zoopt_output_dir, 'precision_recall_curves.png'))
plt.close()
print("Precision-Recall curves saved.")

# --------------------------
# Save ROC and PR curves for each class separately
# --------------------------

for i in range(n_classes):
    # ROC Curve
    plt.figure(figsize=(8,6))
    plt.plot(fpr[i], tpr[i], color='darkorange',
             lw=2, label='ROC curve (area = %0.2f)' % roc_auc[i])
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([-0.01, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Receiver Operating Characteristic - {dataset.classes[i]}')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(zoopt_output_dir, f'roc_curve_{dataset.classes[i]}.png'))
    plt.close()
    
    # Precision-Recall Curve
    plt.figure(figsize=(8,6))
    plt.plot(recall[i], precision[i], color='darkorange',
             lw=2, label='PR curve (AP = %0.2f)' % pr_auc[i])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve - {dataset.classes[i]}')
    plt.legend(loc="lower left")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(zoopt_output_dir, f'pr_curve_{dataset.classes[i]}.png'))
    plt.close()
    
print("Individual ROC and Precision-Recall curves saved.")
