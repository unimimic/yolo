import os
import json
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = []
        self.annotation_files = []

        # 遍歷所有資料夾，並收集影像和標註檔案
        for subdir, _, files in os.walk(root_dir):
            for file in files:
                if file.endswith('.jpg') or file.endswith('.png'):
                    image_path = os.path.join(subdir, file)
                    annotation_path = os.path.splitext(image_path)[0] + '.json'
                    if os.path.exists(annotation_path):
                        self.image_files.append(image_path)
                        self.annotation_files.append(annotation_path)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # 讀取影像
        image_path = self.image_files[idx]
        image = Image.open(image_path).convert('RGB')
        
        # 讀取標註資料
        annotation_path = self.annotation_files[idx]
        with open(annotation_path, 'r') as f:
            annotation = json.load(f)

        # 從標註中解析 shape 資料，例如 bounding boxes
        boxes = []
        labels = []
        for shape in annotation['shapes']:
            label = shape['label']
            points = shape['points']
            # 處理 bounding box (rectangle)，解析兩個點的座標為 x1, y1, x2, y2
            if shape['shape_type'] == 'rectangle':
                x1, y1 = points[0]
                x2, y2 = points[1]
                boxes.append([x1, y1, x2, y2])
                # 假設 labels 需要轉換為數字，例如 {"qwe": 0, "rty": 1} 等
                label_map = {"qwe": 0, "rty": 1}
                labels.append(label_map.get(label, -1))  # 沒有對應 label 時回傳 -1

        # 轉換為 tensor 格式
        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64)
        target = {'boxes': boxes, 'labels': labels}

        # 影像的 transform 處理（如 resize, normalization）
        if self.transform:
            image = self.transform(image)

        return image, target

# # 使用方式
# transform = transforms.Compose([
#     transforms.Resize((512, 512)),
#     transforms.ToTensor(),
# ])

# # 建立訓練與測試資料集
# train_dataset = CustomDataset(root_dir='Dataset/train', transform=transform)
# test_dataset = CustomDataset(root_dir='Dataset/test', transform=transform)

# # 建立 DataLoader
# train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
# test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)

# # 測試讀取資料
# for images, targets in train_loader:
#     print(images.shape)  # 應該輸出 [batch_size, 3, 512, 512]
#     print(targets)       # 輸出每個批次的目標標註，包括 boxes 和 labels
#     break
