from datasets import load_dataset
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from PIL import Image
import requests
from io import BytesIO
from dataset import FlickrDataModule
from model import ImageCaptioningGAN

def train_model():
    # 初始化数据模块
    data_module = FlickrDataModule(batch_size=32)
    
    # 初始化模型
    model = ImageCaptioningGAN(
        latent_dim=100,
        learning_rate=0.0002
    )
    
    # 设置训练器
    trainer = pl.Trainer(
        max_epochs=100,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1,
        precision=16,
        callbacks=[
            pl.callbacks.ModelCheckpoint(
                monitor='g_loss',
                dirpath='checkpoints',
                filename='gan-{epoch:02d}-{g_loss:.2f}',
                save_top_k=3,
                mode='min',
            ),
            pl.callbacks.EarlyStopping(
                monitor='g_loss',
                patience=10,
                mode='min'
            )
        ],
        logger=pl.loggers.TensorBoardLogger('logs/', name='flickr_captioning_gan')
    )
    
    # 训练模型
    trainer.fit(model, data_module)