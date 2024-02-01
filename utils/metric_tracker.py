from lightning.pytorch.callbacks import Callback
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor

class MetricTracker(Callback):
    
    def __init__(self):
        self.val_loss = []
        self.val_acc = []
        self.train_loss = []
        self.train_acc = []
        self.lr = []

    def on_validation_epoch_end(self, trainer, module):
        val_loss = trainer.logged_metrics['val_loss'].item()
        val_acc = trainer.logged_metrics['val_acc'].item()
        self.val_loss.append(val_loss)
        self.val_acc.append(val_acc)
        
    def on_train_epoch_end(self, trainer, module):
        train_loss = trainer.logged_metrics['train_loss'].item()
        train_acc = trainer.logged_metrics['train_acc'].item()
        lr = module.optimizers().param_groups[0]['lr']
        self.train_loss.append(train_loss)
        self.train_acc.append(train_acc)
        self.lr.append(lr)

metric_tracker = MetricTracker()
checkpoint_callback = ModelCheckpoint(monitor='val_loss')
lr_monitor = LearningRateMonitor(logging_interval='epoch')