import os
from torch.nn import init
import torch.nn.functional as func
from torch.utils.data import DataLoader, Dataset, random_split
import torchaudio
from torchaudio.transforms import MelSpectrogram, AmplitudeToDB
import torch
import pandas as pd

import pytorch_lightning as pl
from torch import nn, optim
from torchmetrics.classification import Accuracy
from torch.optim.lr_scheduler import ReduceLROnPlateau

import lightning as light
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping

import torchmetrics
from matplotlib import pyplot as plt
import seaborn as sns

# from main import dataset


class CNNAudioClassifier(nn.Module):
    def __init__(self, output_dim):
        super().__init__()
        conv_layers = []

        # First Convolution Block with Relu and Batch Norm. Use Kaiming Initialization
        self.conv1 = nn.Conv2d(1, 8, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2))
        self.relu1 = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(8)
        init.kaiming_normal_(self.conv1.weight, a=0.1)
        self.conv1.bias.data.zero_()
        conv_layers += [self.conv1, self.relu1, self.bn1]

        # Second Convolution Block
        self.conv2 = nn.Conv2d(8, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.relu2 = nn.ReLU()
        self.bn2 = nn.BatchNorm2d(16)
        init.kaiming_normal_(self.conv2.weight, a=0.1)
        self.conv2.bias.data.zero_()
        conv_layers += [self.conv2, self.relu2, self.bn2]

        # Second Convolution Block
        self.conv3 = nn.Conv2d(16, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.relu3 = nn.ReLU()
        self.bn3 = nn.BatchNorm2d(32)
        init.kaiming_normal_(self.conv3.weight, a=0.1)
        self.conv3.bias.data.zero_()
        conv_layers += [self.conv3, self.relu3, self.bn3]

        # Second Convolution Block
        self.conv4 = nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.relu4 = nn.ReLU()
        self.bn4 = nn.BatchNorm2d(64)
        init.kaiming_normal_(self.conv4.weight, a=0.1)
        self.conv4.bias.data.zero_()
        conv_layers += [self.conv4, self.relu4, self.bn4]

        # Linear Classifier
        self.ap = nn.AdaptiveAvgPool2d(output_size=1)
        self.lin = nn.Linear(in_features=64, out_features=output_dim)

        # Wrap the Convolutional Blocks
        self.conv = nn.Sequential(*conv_layers)

    # Forward pass computations
    def forward(self, x):
        # Run the convolutional blocks
        x = self.conv(x)

        # Adaptive pool and flatten for input to linear layer
        x = self.ap(x)
        x = x.view(x.shape[0], -1)

        # Linear layer
        x = self.lin(x)

        # Final output
        return x



class CNNAudioClassifierLightning(light.LightningModule):
    def __init__(self, output_dim, lr=0.001):
        super().__init__()

        # Verwende das bestehende CNNAudioClassifier-Modell
        self.num_classes = 7
        self.model = CNNAudioClassifier(self.num_classes)
        self.criterion = nn.CrossEntropyLoss()
        self.accuracy = Accuracy(task="multiclass", num_classes=self.num_classes)

        # Speichere Hyperparameter
        self.save_hyperparameters(ignore=['output_dim'])
        self.conf_matrix_metric = torchmetrics.classification.confusion_matrix.MulticlassConfusionMatrix(
            self.num_classes)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        loss = self.criterion(outputs, labels)

        # Berechne und logge die Trainingsgenauigkeit
        acc = self.accuracy(outputs.softmax(dim=-1), labels)
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_accuracy", acc, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        # labels = labels.argmax(dim=1)
        outputs = self(inputs)
        loss = self.criterion(outputs, labels)

        # Berechne und logge Accuracy und Loss
        acc = self.accuracy(outputs.softmax(dim=-1), labels)
        # self.log("val_loss", loss, prog_bar=True)
        self.log('val_loss', loss, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_accuracy", acc, prog_bar=True)

    # def on_validation_epoch_end(self):
    #     current_lr = self.trainer.optimizers[0].param_groups[0]['lr']
    #     print(f"Current Learning Rate: {current_lr}")

    def test_step(self, batch, batch_idx):
        inputs, labels = batch
        # labels = labels.argmax(dim=1)
        outputs = self(inputs)
        loss = self.criterion(outputs, labels)

        # confusion matrix
        preds = torch.argmax(outputs, dim=1)
        self.conf_matrix_metric.update(preds, labels)

        # Berechne und logge Accuracy und Loss
        acc = self.accuracy(outputs.softmax(dim=-1), labels)
        self.log("test_loss", loss)
        self.log("test_accuracy", acc)

    def on_test_epoch_end(self):
        # Berechne die finale Confusion Matrix
        conf_matrix = self.conf_matrix_metric.compute()

        # Log die Confusion Matrix als Bild in TensorBoard
        # fig = self.plot_confusion_matrix(conf_matrix)
        # self.logger.experiment.add_figure("Confusion Matrix", fig, global_step=self.current_epoch)
        # plt.close(fig)  # Speicher freigeben

        # Metric zurücksetzen
        self.conf_matrix_metric.reset()

    # @staticmethod
    # def plot_confusion_matrix(conf_matrix):
    #     """Erstellt ein Confusion-Matrix-Diagramm als Matplotlib-Figur."""
    #     conf_matrix = conf_matrix.cpu().numpy()
    #     fig, ax = plt.subplots(figsize=(8, 8))
    #     sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=dataset.label_dict.keys(),
    #                 yticklabels=dataset.label_dict.keys(), ax=ax)
    #     ax.set_xlabel("Predicted")
    #     ax.set_ylabel("True")
    #     ax.set_title("Confusion Matrix")
    #     return fig

    # def configure_optimizers(self):
    #     # Optimierer und optionalen Scheduler zurückgeben
    #     optimizer = optim.Adam(self.parameters(), lr=self.hparams.lr)
    #     return optimizer

    def on_train_epoch_end(self):
        current_lr = self.trainer.optimizers[0].param_groups[0]['lr']
        print(f"Current learning rate: {current_lr}")
        self.log('learning_rate', current_lr, prog_bar=True, logger=True)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.model.parameters(), lr=self.hparams.lr)  # Beispiel-Optimizer
        scheduler = {
            'scheduler': ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, min_lr=1e-6),
            'monitor': 'val_loss',  # Überwachte Metrik für den Scheduler
        }
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}
