import torch
import numpy as np
import segmentation_models_pytorch as smp
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import os

# 定義資料集
class CustomDataset(Dataset):
    def __init__(self, images_dir, masks_dir, transform=None):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transform
        self.ids = os.listdir(images_dir)
        
    def __len__(self):
        return len(self.ids)
    
    def __getitem__(self, idx):
        img_file = os.path.join(self.images_dir, self.ids[idx])
        mask_file = os.path.join(self.masks_dir, self.ids[idx])
        image = Image.open(img_file).convert("RGB")
        mask = Image.open(mask_file).convert("L")
        
        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)
            
        return image, mask

# 轉換
transform = transforms.Compose([
    transforms.Resize((256, 256)),  # Resize the image to 256x256 pixels
    transforms.ToTensor(),
])

# DataLoader
train_dataset = CustomDataset(r'F:\nuck\data\ki67\ki67\ex1\train_ex1_patch', r'F:\nuck\data\ki67\ki67\ex1\train_ex1_label', transform=transform)
val_dataset = CustomDataset(r'F:\nuck\data\ki67\ki67\ex1\val_ex1_patch', r'F:\nuck\data\ki67\ki67\ex1\val_ex1_label', transform=transform)
test_dataset = CustomDataset(r'F:\nuck\data\ki67\ki67\ex1\test_ex1_patch', r'F:\nuck\data\ki67\ki67\ex1\test_ex1_label', transform=transform)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

# 建立模型
model = smp.Unet(encoder_name="resnet50", encoder_weights="imagenet", in_channels=3, classes=2)

# 損失函數和優化器
criterion = smp.utils.losses.DiceLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# 訓練函式
def train_model(model, criterion, optimizer, train_loader, val_loader, num_epochs=25, n_classes=2):
    best_miou = 0.0  # Initialize the best MIoU score
    best_weights = None  # Variable to store the best weights

    for epoch in range(num_epochs):
        model.train()  # Set model to training mode
        running_loss = 0.0

        # Training loop
        for images, masks in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)

        # Evaluation loop
        val_miou = evaluate_model(model, val_loader, n_classes=n_classes)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}, Val MIoU: {val_miou:.4f}")

        # Update the best model if the current model is better
        if val_miou > best_miou:
            best_miou = val_miou
            best_weights = model.state_dict().copy()  # Copy the model's state_dict

    # After all epochs, load the best model weights
    model.load_state_dict(best_weights)
    # Optionally, save the best model to disk
    torch.save(best_weights, "best_model_weights.pth")
    print(f"Training complete. Best Val MIoU: {best_miou:.4f}")

    return model  # Return the model with the best weights

#import torch
import numpy as np
import os
from torchvision.utils import save_image

def miou_score(pred, target, smooth=1e-10, n_classes=2):
    """
    Compute the Mean Intersection over Union (MIoU) score.
    :param pred: the model's predicted probabilities
    :param target: the ground truth
    :param smooth: a small value to avoid division by zero
    :param n_classes: the number of classes in the dataset
    :return: the MIoU score
    """
    pred = torch.argmax(pred, dim=1)  # Convert probabilities to predictions
    miou_total = 0.0
    for class_id in range(n_classes):
        true_positive = ((pred == class_id) & (target == class_id)).sum()
        false_positive = ((pred == class_id) & (target != class_id)).sum()
        false_negative = ((pred != class_id) & (target == class_id)).sum()
        intersection = true_positive
        union = true_positive + false_positive + false_negative + smooth
        miou = intersection / union
        miou_total += miou
    return miou_total / n_classes

def evaluate_model(model, loader, n_classes=2):
    model.eval()
    total_miou = 0
    with torch.no_grad():
        for images, masks in loader:
            outputs = model(images)
            total_miou += miou_score(outputs, masks, n_classes=n_classes)
    return total_miou / len(loader)

def predict(model, loader, save_dir="predicted_masks"):
    model.eval()
    os.makedirs(save_dir, exist_ok=True)
    with torch.no_grad():
        for idx, (images, _) in enumerate(loader):
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)  # Convert probabilities to predictions
            for j, pred in enumerate(preds):
                save_image(pred.float() / preds.max(), os.path.join(save_dir, f"mask_{idx * loader.batch_size + j}.png"))


# 主程式（示例）
if __name__ == "__main__":
    best_weights_path = 'best_model_weights.pth'
    if os.path.exists(best_weights_path):
        # 匯入已有的權重
        model.load_state_dict(torch.load(best_weights_path))
        print("Loaded existing model weights.")
    else:
        print("No existing model weights found, starting training from scratch.")

        model = train_model(model, criterion, optimizer, train_loader, val_loader, num_epochs=25)
    
    
    # 逕行測試與預測
    test_MIoU = evaluate_model(model, test_loader)
    print(f"test MIoU: {test_MIoU}")
    predict(model, test_loader, save_dir="predicted_masks")
    print("Finsh All Training and Testing")
