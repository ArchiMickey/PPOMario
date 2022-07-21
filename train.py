import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from src.ppo import PPO
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, RichProgressBar, TQDMProgressBar
from datetime import datetime


checkpoint_callback = ModelCheckpoint(
    save_top_k=3,
    monitor="test_score",
    mode="max",
    dirpath="model/",
    filename="ppomario-{test_score:.2f}-{step}",
    every_n_train_steps=5000,
    save_last=True,
)

def main(world: int = 1, stage: int = 1, ckpt_path: str = None):
    if ckpt_path is not None:
        model = PPO.load_from_checkpoint(ckpt_path)
    
    else:
        model = PPO(
            world=world,
            stage=stage,
            lr=1e-3,
            nb_optim_iters=1,
            batch_epoch=10,
            batch_size=64,
            num_workers=6,
            hidden_size=512,
            steps_per_epoch=1024,
            render_freq=10000,
        )

    wandb_logger = WandbLogger(name=f"PPOMario-{world}-{stage}")

    trainer = pl.Trainer(
        accelerator="gpu",
        devices = 1 if torch.cuda.is_available() else None,
        max_steps=2000000,
        logger=wandb_logger,
        log_every_n_steps=10,
        default_root_dir="model",
        gradient_clip_val= 100.0,
        auto_lr_find=True,
        callbacks=[checkpoint_callback, LearningRateMonitor(logging_interval='epoch')],
    )
    
    trainer.fit(model)

if __name__ == "__main__":
    main(world=2, stage=1)