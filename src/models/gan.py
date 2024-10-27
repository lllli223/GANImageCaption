import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
from transformers import ViTFeatureExtractor, ViTModel
from transformers import BertTokenizer, BertModel
from torchvision import transforms
import torch.optim as optim
from PIL import Image
import numpy as np

class ImageCaptioningGAN(pl.LightningModule):
    def __init__(self, latent_dim=100, learning_rate=0.0002):
        super().__init__()
        self.save_hyperparameters()
        
        # 初始化 tokenizer 和 feature extractor
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224')
        
        # 创建生成器和判别器
        self.generator = Generator(latent_dim)
        self.discriminator = Discriminator()
        
        # 损失函数
        self.criterion = nn.BCELoss()
        
    def forward(self, z, images):
        return self.generator(z, images)
    
    def training_step(self, batch, batch_idx, optimizer_idx):
        images, captions = batch
        batch_size = images.size(0)
        
        # 生成随机噪声
        z = torch.randn(batch_size, self.hparams.latent_dim).to(self.device)
        
        # 训练判别器
        if optimizer_idx == 0:
            # 真实样本的标签为1
            valid = torch.ones(batch_size, 1).to(self.device)
            # 生成的假样本的标签为0
            fake = torch.zeros(batch_size, 1).to(self.device)
            
            # 计算真实样本的损失
            real_loss = self.criterion(
                self.discriminator(images, captions), valid
            )
            
            # 生成假样本
            fake_captions = self.generator(z, images)
            
            # 计算假样本的损失
            fake_loss = self.criterion(
                self.discriminator(images, fake_captions.detach()), fake
            )
            
            # 总损失
            d_loss = (real_loss + fake_loss) / 2
            
            self.log('d_loss', d_loss)
            return d_loss
            
        # 训练生成器
        if optimizer_idx == 1:
            valid = torch.ones(batch_size, 1).to(self.device)
            
            # 生成假样本
            generated_captions = self.generator(z, images)
            
            # 计算生成器损失
            g_loss = self.criterion(
                self.discriminator(images, generated_captions), valid
            )
            
            self.log('g_loss', g_loss)
            return g_loss
    
    def configure_optimizers(self):
        opt_g = torch.optim.Adam(self.generator.parameters(), lr=self.hparams.learning_rate)
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=self.hparams.learning_rate)
        return [opt_d, opt_g], []

class Generator(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        
        # 图像特征提取器
        self.vit = ViTModel.from_pretrained('google/vit-base-patch16-224')
        
        # 将噪声和图像特征转换为序列
        self.fc = nn.Linear(latent_dim + 768, 768)
        
        # Transformer 解码器层
        decoder_layer = nn.TransformerDecoderLayer(d_model=768, nhead=8)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)
        
        # 输出层
        self.output_layer = nn.Linear(768, 30522)  # BERT词表大小
        
    def forward(self, z, images):
        # 提取图像特征
        image_features = self.vit(images).last_hidden_state
        
        # 合并噪声和图像特征
        z = z.unsqueeze(1).expand(-1, image_features.size(1), -1)
        combined = torch.cat([z, image_features], dim=-1)
        
        # 转换为序列
        sequence = self.fc(combined)
        
        # 通过 Transformer 解码器
        output = self.transformer_decoder(sequence, image_features)
        
        # 生成词的概率分布
        logits = self.output_layer(output)
        
        return logits

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        
        # 图像编码器
        self.vit = ViTModel.from_pretrained('google/vit-base-patch16-224')
        
        # 文本编码器
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        
        # 判别器网络
        self.classifier = nn.Sequential(
            nn.Linear(768 * 2, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        
    def forward(self, images, captions):
        # 提取图像特征
        image_features = self.vit(images).last_hidden_state.mean(dim=1)
        
        # 提取文本特征
        caption_features = self.bert(captions).last_hidden_state.mean(dim=1)
        
        # 合并特征
        combined_features = torch.cat([image_features, caption_features], dim=1)
        
        # 判别真伪
        validity = self.classifier(combined_features)
        
        return validity

