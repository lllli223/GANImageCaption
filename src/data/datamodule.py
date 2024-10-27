import pytorch_lightning as pl
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from torchvision import transforms
from datasets import load_dataset
from .dataset import FlickrDataset

class FlickrDataModule(pl.LightningDataModule):
    def __init__(self, batch_size=32):
        super().__init__()
        self.batch_size = batch_size
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def prepare_data(self):
        # 下载数据集
        load_dataset("nlphuji/flickr8k")

    def setup(self, stage=None):
        # 加载数据集
        dataset = load_dataset("nlphuji/flickr8k")
        
        if stage == 'fit' or stage is None:
            self.train_dataset = dataset['train']
            self.val_dataset = dataset['validation']
        
        if stage == 'test':
            self.test_dataset = dataset['test']

    def train_dataloader(self):
        return DataLoader(
            FlickrDataset(self.train_dataset, self.transform, self.tokenizer),
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )

    def val_dataloader(self):
        return DataLoader(
            FlickrDataset(self.val_dataset, self.transform, self.tokenizer),
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )

    def test_dataloader(self):
        return DataLoader(
            FlickrDataset(self.test_dataset, self.transform, self.tokenizer),
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )