import json
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import timm
from tqdm import tqdm

# ===== 配置 =====
IMG_ROOT = '/nas/data_2/wanhongz/datasets'  # 请确认图片路径
TRAIN_JSON = '/nas/data_2/wanhongz/datasets/DGM4/metadata/train.json'
VAL_JSON = '/nas/data_2/wanhongz/datasets/DGM4/metadata/val.json'
BATCH_SIZE = 64
EPOCHS = 20
LR = 1e-4
DEVICE = 'cuda'
SAVE_PATH = 'tamper_best.pth'

# ===== 数据集 =====
class TamperDataset(Dataset):
    def __init__(self, json_path, img_root, transform):
        with open(json_path, 'r') as f:
            self.data = json.load(f)
        self.img_root = img_root
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        img_path = os.path.join(self.img_root, item['image']) # 关键修改
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)
        label = item['fake_cls'] != 'real'  # 如果 fake_cls 不是 'real' 则为1，否则0
        return image, label

# 数据增强
transform_train = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

transform_val = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

# 加载数据
train_dataset = TamperDataset(TRAIN_JSON, IMG_ROOT, transform_train)
val_dataset = TamperDataset(VAL_JSON, IMG_ROOT, transform_val)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

# ===== 模型 =====
model = timm.create_model('efficientnet_b0', pretrained=True, num_classes=1)
model = model.to(DEVICE)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# ===== 训练 =====
best_acc = 0.0
for epoch in range(EPOCHS):
    model.train()
    train_loss = 0.0
    pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{EPOCHS}')
    for images, labels in pbar:
        images, labels = images.to(DEVICE), labels.float().to(DEVICE).unsqueeze(1)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        pbar.set_postfix({'loss': loss.item()})
    
    # 验证
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            preds = (torch.sigmoid(outputs).squeeze() > 0.5).long()
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    acc = correct / total
    print(f'Epoch {epoch+1}, Train Loss: {train_loss/len(train_loader):.4f}, Val Acc: {acc:.4f}')
    
    if acc > best_acc:
        best_acc = acc
        torch.save(model.state_dict(), SAVE_PATH)
        print(f'Best model saved with acc {best_acc:.4f}')

print('Training finished. Best model saved to', SAVE_PATH)