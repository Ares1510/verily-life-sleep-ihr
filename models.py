import math
import numpy as np
import torch
from torch import nn, optim
from torch.optim.lr_scheduler import LambdaLR
import pytorch_lightning as pl
from torchmetrics import MetricCollection, Accuracy, AUROC, F1Score
import torch
from torch import nn

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3)
        self.relu = nn.LeakyReLU()
        self.maxpool = nn.MaxPool1d(kernel_size=2, stride=2)
        
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.relu(out)
        out = self.maxpool(out)
        out += residual
        return out

class DilatedConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DilatedConvBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=7, dilation=2)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=7, dilation=4)
        self.conv3 = nn.Conv1d(out_channels, out_channels, kernel_size=7, dilation=8)
        self.conv4 = nn.Conv1d(out_channels, out_channels, kernel_size=7, dilation=16)
        self.conv5 = nn.Conv1d(out_channels, out_channels, kernel_size=7, dilation=32)
        self.relu = nn.LeakyReLU()
        self.dropout = nn.Dropout()
        
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.relu(out)
        out = self.conv4(out)
        out = self.relu(out)
        out = self.conv5(out)
        out = self.relu(out)
        out = self.dropout(out)
        out += residual
        return out

class HeartRateNetwork(pl.LightningModule):
    def __init__(self, lr=0.0001, weight_decay=0.25):
        super().__init__()
        self.input_conv = nn.Conv1d(1, 16, kernel_size=1),
        self.cnn_blocks = nn.Sequential(
            ConvBlock(16, 16),
            ConvBlock(16, 32),
            ConvBlock(32, 64)
        )
        self.embedding_layer = nn.Linear(64 * 256, 1200 * 128)
        self.dilated_blocks = nn.Sequential(
            DilatedConvBlock(128, 128),
            DilatedConvBlock(128, 128)
        )
        self.output_conv = nn.Conv1d(128, 4, kernel_size=1)

        self.criterion = nn.CrossEntropyLoss()
        self.lr = lr
        self.weight_decay = weight_decay

        metrics = MetricCollection([Accuracy(task='binary'), 
                                               AUROC(task='binary'), 
                                               F1Score(task='binary', average='macro')])
        self.train_metrics = metrics.clone(prefix='train_')
        self.val_metrics = metrics.clone(prefix='val_')
        self.test_metrics = metrics.clone(prefix='test_')
        
    def forward(self, x):
        out = self.input_conv(x)
        out = self.cnn_blocks(x)
        out = out.view(out.size(0), -1)  # Flatten the output
        out = self.embedding_layer(out)
        out = out.view(out.size(0), 1200, 128)  # Reshape to (batch_size, 1200, 128)
        out = out.permute(0, 2, 1)  # Reshape to (batch_size, 128, 1200)
        out = self.dilated_blocks(out)
        out = self.output_conv(out)
        return out
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.train_metrics(y_hat, y, on_epoch=True)
        self.log('train_loss', loss, on_step=True, on_epoch=True, logger=True)
        self.log_dict(self.train_metrics, on_step=False, on_epoch=True, logger=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.val_metrics(y_hat, y, on_epoch=True)
        self.log('val_loss', loss, on_step=True, on_epoch=True, logger=True)
        self.log_dict(self.val_metrics, on_step=False, on_epoch=True, logger=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.test_metrics(y_hat, y, on_epoch=True)
        self.log('test_loss', loss, on_step=True, on_epoch=True, logger=True)
        self.log_dict(self.test_metrics, on_step=False, on_epoch=True, logger=True)
        return loss
    
    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)


