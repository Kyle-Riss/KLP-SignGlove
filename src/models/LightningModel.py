import numpy as np
import torch
import lightning as L
import wandb
from lightning.pytorch.loggers.wandb import WandbLogger
from torcheval.metrics.functional import multiclass_f1_score
from sklearn.metrics import confusion_matrix

from src.models.generalModels import ModelInfo

from typing import Tuple, Dict
from torch import Tensor


class LitModel(L.LightningModule, ModelInfo):

    def training_step(self, batch: Dict[str, Tensor], batch_idx: int) -> Tensor:
        """training step for the model with regularization"""

        x, x_padding, y = batch

        logits, loss = self(batch[x], batch[x_padding], batch[y])  # forward pass
        
        # 정확도 계산 및 로깅
        predictions = torch.argmax(logits, dim=-1)
        accuracy = (predictions == batch[y]).float().mean()
        
        self.log("train/loss", loss, prog_bar=True)
        self.log("train/accuracy", accuracy, prog_bar=True)

        return loss

    def configure_optimizers(self):
        """
        과적합 방지를 위한 강화된 옵티마이저 설정
        - Weight Decay: L2 정규화
        - Learning Rate Scheduler: 더 안정적인 학습
        - Gradient Clipping: 그래디언트 폭발 방지
        """
        # Weight Decay를 통한 L2 정규화 강화
        self.optimizer = torch.optim.AdamW(
            self.parameters(), 
            lr=self.lr,
            weight_decay=0.001,  # L2 정규화 강화
            betas=(0.9, 0.999),
            eps=1e-8
        )

        # 더 안정적인 학습률 스케줄러
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 
            mode='min',
            factor=0.5, 
            min_lr=1e-6,  # 더 낮은 최소 학습률
            patience=15,  # 더 빠른 감소
            verbose=True
        )

        optim = {
            "optimizer": self.optimizer,
            "lr_scheduler": scheduler,
            "monitor": "val/loss"
        }

        return optim

    def validation_step(self, batch: Dict[str, Tensor], batch_idx: int) -> Tensor:

        x, x_padding, y = batch

        logits, loss = self(batch[x], batch[x_padding], batch[y])  # forward pass
        logits_argmax = torch.argmax(logits, dim=-1)  # get argmax of logits
        # confusion matrix
        cm = confusion_matrix(
            batch[y].cpu(),
            logits.cpu().argmax(axis=1).numpy(),
            labels=np.arange(self.classes).tolist(),
        )
        true_acc, cat_acc = self.get_accuracy(cm)  # call accuracy from misc.py
        val_f1 = multiclass_f1_score(
            logits, batch[y], num_classes=self.classes, average=None
        )  # f1-score

        # log metrics
        self.log("val/loss", loss, prog_bar=True)
        self.log("val/accuracy", torch.tensor(cat_acc).mean().item(), prog_bar=True)
        self.log("val/f1_score", val_f1.mean().item(), prog_bar=True)
        self.log("learning_rate", self.optimizer.param_groups[0]['lr'], prog_bar=True)

        if isinstance(self.logger, WandbLogger):
            plot = wandb.plot.confusion_matrix(
                probs=None,
                y_true=np.array(batch[y].cpu()),
                preds=np.array(logits_argmax.cpu()),
                class_names=np.arange(self.classes).tolist(),
            )
            self.logger.experiment.log({"confusion_matrix": plot})

        val_loss = loss
        return val_loss

    def test_step(self, batch: Dict[str, Tensor], batch_idx: int) -> Tensor:
        """test step for the model"""
        
        x, x_padding, y = batch
        
        logits, loss = self(batch[x], batch[x_padding], batch[y])  # forward pass
        logits_argmax = torch.argmax(logits, dim=-1)  # get argmax of logits
        
        # confusion matrix
        cm = confusion_matrix(
            batch[y].cpu(),
            logits.cpu().argmax(axis=1).numpy(),
            labels=np.arange(self.classes).tolist(),
        )
        true_acc, cat_acc = self.get_accuracy(cm)  # call accuracy from misc.py
        test_f1 = multiclass_f1_score(
            logits, batch[y], num_classes=self.classes, average=None
        )  # f1-score
        
        # log metrics
        self.log("test/loss", loss, prog_bar=True)
        self.log("test/accuracy", torch.tensor(cat_acc).mean().item(), prog_bar=True)
        self.log("test/f1_score", test_f1.mean().item(), prog_bar=True)
        
        test_loss = loss
        return test_loss
