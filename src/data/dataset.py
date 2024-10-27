from datasets import load_dataset
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from PIL import Image
import requests
from io import BytesIO
from datamodules import FlickrDataModule
from transformers import BertTokenizer


class FlickrDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, transform, tokenizer):
        self.dataset = dataset
        self.transform = transform
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        
        # 加载图像
        image = Image.open(BytesIO(requests.get(item['image_url']).content)).convert('RGB')
        if self.transform:
            image = self.transform(image)
        
        # 处理文本
        # 随机选择一个描述（每张图片有多个描述）
        caption = item['captions'][0]  # 或者随机选择：random.choice(item['captions'])
        
        encoded_caption = self.tokenizer(
            caption,
            padding='max_length',
            max_length=128,
            truncation=True,
            return_tensors='pt'
        )
        
        return image, encoded_caption['input_ids'].squeeze()

# 添加数据集验证函数
def validate_dataset():
    data_module = FlickrDataModule(batch_size=2)
    data_module.setup()
    
    # 获取一个批次的数据
    train_loader = data_module.train_dataloader()
    images, captions = next(iter(train_loader))
    
    print(f"Image batch shape: {images.shape}")
    print(f"Caption batch shape: {captions.shape}")
    
    # 解码一个描述示例
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    decoded_caption = tokenizer.decode(captions[0], skip_special_tokens=True)
    print(f"Sample caption: {decoded_caption}")

if __name__ == "__main__":
    # 验证数据集
    validate_dataset()
    
    # 训练模型
    train_model()
