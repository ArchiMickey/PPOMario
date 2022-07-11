import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from ppo import PPO
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, RichProgressBar
from datetime import datetime

checkpoint_callback = ModelCheckpoint(
    save_top_k=3,
    monitor="test_reward",
    mode="max",
    dirpath="model/",
    filename="qmario-{test_reward:.2f}-{step}",
    every_n_train_steps=1000,
)

# 1 training step = 2.5 global step
model = PPO(
    world=1,
    stage=1,
    nb_optim_iters=1,
    batch_size=64,
)

now_dt = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
wandb_logger = WandbLogger(name=f"PPOMario-{now_dt}")

trainer = pl.Trainer(
    accelerator="gpu",
    devices = 1 if torch.cuda.is_available() else None,
    max_steps=2000000,
    logger=wandb_logger,
    log_every_n_steps=5,
    gradient_clip_val= 5000.0,
    auto_lr_find=True,
    callbacks=[checkpoint_callback, LearningRateMonitor(logging_interval='step')],
)

# trainer.tune(model)
trainer.fit(model)