import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchmetrics
import torchvision
import lightning as pl
from lightning.pytorch import LightningModule
from utils import CosineAnnealingWarmRestarts
import finetuning_scheduler as fts

class StanfordCarsNet(pl.LightningModule):
    def __init__(self, lr=1e-2, min_lr=1e-6, weight_decay=1e-4, is_finetuned = False):
        super().__init__()
        self.save_hyperparameters()
        self.learning_rate = lr
        self.min_lr = min_lr
        self.weight_decay = weight_decay
        self.is_finetuned = is_finetuned
        
        backbone = torchvision.models.resnet50(pretrained=True)
        layers = list(backbone.children())[:-1]
        self.feature_extractor = nn.Sequential(*layers)
        num_features = backbone.fc.in_features
        num_target_classes = 196
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, num_target_classes)    
        )
        
        self.train_acc = torchmetrics.Accuracy(task="multiclass",
                                               num_classes=num_target_classes)
        self.val_acc = torchmetrics.Accuracy(task="multiclass",
                                             num_classes=num_target_classes)
        self.test_acc = torchmetrics.Accuracy(task="multiclass",
                                             num_classes=num_target_classes)
        
    @property
    def finetuningscheduler_callback(self) -> fts.FinetuningScheduler:
        fts_callback = [c for c in self.trainer.callbacks if isinstance(c, fts.FinetuningScheduler)]
        return fts_callback[0] if fts_callback else None
        
    def forward(self, x):
        if self.is_finetuned:
            representations = self.feature_extractor(x).flatten(1)
        else:
            self.feature_extractor.eval()
            with torch.no_grad():
                representations = self.feature_extractor(x).flatten(1)
        
        logits = self.classifier(representations)
        return logits
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        self.train_acc(logits, y)

        tensorboard_logs = {"train_loss":loss, "train_acc": self.train_acc}
        self.log_dict(tensorboard_logs, on_step=False, on_epoch=True)
        return loss
    
    def on_train_epoch_end(self):
        if self.finetuningscheduler_callback:
            self.log("finetuning_schedule_depth", float(self.finetuningscheduler_callback.curr_depth))
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        self.val_acc(logits, y)

        tensorboard_logs = {"val_loss":loss, "val_acc":self.val_acc}
        self.log_dict(tensorboard_logs, prog_bar=True, on_step=False, on_epoch=True)
        return loss
    
    def predict_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        preds = torch.argmax(logits, dim=1)
        return preds
    
    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay = self.weight_decay)
        scheduler = {
            "scheduler": CosineAnnealingWarmRestarts(optimizer, 20, 1, 1e-6),
            "interval": "epoch",
        }
        return [optimizer], [scheduler]
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        self.test_acc(logits, y)
        self.log('test_acc', self.test_acc, on_step=False, on_epoch=True)