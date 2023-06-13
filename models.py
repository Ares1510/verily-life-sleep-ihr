import torch
import pytorch_lightning as pl
from torch import nn, optim
from torchmetrics import MetricCollection, Accuracy, CohenKappa


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()

        self.conv_block = nn.Sequential(nn.Conv1d(in_channels, out_channels, kernel_size=3, padding='same'),
                                        nn.Conv1d(out_channels, out_channels, kernel_size=3, padding='same'),
                                        nn.LeakyReLU(0.15),
                                        nn.MaxPool1d(kernel_size=2, stride=2))

        self.downsample = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=1),
            nn.MaxPool1d(kernel_size=2, stride=2))
        
    def forward(self, x):
        residual = self.downsample(x)
        out = self.conv_block(x)
        out += residual
        return out


class DilatedConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DilatedConvBlock, self).__init__()

        self.dil_conv_block = nn.Sequential(nn.Conv1d(in_channels, out_channels, kernel_size=7, dilation=2, padding='same'),
                                            nn.Conv1d(out_channels, out_channels, kernel_size=7, dilation=4, padding='same'),
                                            nn.Conv1d(out_channels, out_channels, kernel_size=7, dilation=8, padding='same'),
                                            nn.Conv1d(out_channels, out_channels, kernel_size=7, dilation=16, padding='same'),
                                            nn.Conv1d(out_channels, out_channels, kernel_size=7, dilation=32, padding='same'),
                                            nn.LeakyReLU(0.15),
                                            nn.Dropout(0.2))

        self.downsample = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        
    def forward(self, x):
        residual = self.downsample(x)
        out = self.dil_conv_block(x)
        out += residual
        return out


class HeartRateNetwork(pl.LightningModule):
    def __init__(self, lr=0.0001):
        super().__init__()

        self.input_conv = nn.Conv1d(1, 8, kernel_size=1)

        self.cnn_blocks = nn.Sequential(
            ConvBlock(8, 16),
            ConvBlock(16, 32),
            ConvBlock(32, 64))
        
        self.embedding_layer = nn.Linear(64 * 32, 128)

        self.dilated_blocks = nn.Sequential(
            DilatedConvBlock(128, 128),
            DilatedConvBlock(128, 128))
        
        self.output_conv = nn.Conv1d(128, 4, kernel_size=1)

        self.criterion = nn.CrossEntropyLoss()
        self.lr = lr

        metrics = MetricCollection([Accuracy(task='multiclass', num_classes=4), 
                                    CohenKappa(task='multiclass', num_classes=4)])
        self.train_metrics = metrics.clone(prefix='train_')
        self.val_metrics = metrics.clone(prefix='val_')
        self.test_metrics = metrics.clone(prefix='test_')
        
    def forward(self, x):
        x = x.unsqueeze(1)
        out = self.input_conv(x)
        out = self.cnn_blocks(out)
        out = out.view(out.size(0), -1) # Flatten the output
        out = self.embedding_layer(out)
        #reshape batch dimension to time dimension
        out = out.unsqueeze(0)
        out = out.permute(0, 2, 1)
        out = self.dilated_blocks(out)
        out = self.output_conv(out)
        out = out.squeeze(0)
        out = out.permute(1, 0)
        return out
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        #ignore values that were padded from metric calculation
        mask = torch.any(x != 0, dim=1).nonzero().squeeze()
        y_hat = torch.index_select(y_hat, 0, mask)
        y = torch.index_select(y, 0, mask)
        self.train_metrics(y_hat, y, on_epoch=True)
        self.log('train_loss', loss, on_step=True, on_epoch=True, logger=True)
        self.log_dict(self.train_metrics, on_step=False, on_epoch=True, logger=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        mask = torch.any(x != 0, dim=1).nonzero().squeeze()
        y_hat = torch.index_select(y_hat, 0, mask)
        y = torch.index_select(y, 0, mask)
        self.val_metrics(y_hat, y, on_epoch=True)
        self.log('val_loss', loss, on_step=True, on_epoch=True, logger=True)
        self.log_dict(self.val_metrics, on_step=False, on_epoch=True, logger=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        mask = torch.any(x != 0, dim=1).nonzero().squeeze()
        y_hat = torch.index_select(y_hat, 0, mask)
        y = torch.index_select(y, 0, mask)
        self.test_metrics(y_hat, y, on_epoch=True)
        self.log('test_loss', loss, on_step=True, on_epoch=True, logger=True)
        self.log_dict(self.test_metrics, on_step=False, on_epoch=True, logger=True)
        return loss
    
    def configure_optimizers(self):
        return optim.AdamW(self.parameters(), lr=self.lr, weight_decay=0.25)


