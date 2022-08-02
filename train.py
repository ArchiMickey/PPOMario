import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from src.ppomario import PPOMario
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, ModelSummary
from datetime import datetime

def main(world: int = 1, stage: int = 1, max_episodes: int = 100000, ckpt_path: str = None):
    
    checkpoint_callback = ModelCheckpoint(
        save_top_k=3,
        monitor="avg_score",
        mode="max",
        dirpath=f"model/{world}-{stage}/",
        filename="ppomario-{avg_score:.2f}",
        every_n_epochs=50,
        save_last=True,
    )
    
    model = PPOMario(
        world=world,
        stage=stage,
        lr=2.5e-4,
        lr_decay_ratio=0,
        lr_decay_epoch=max_episodes,
        nb_optim_iters=1,
        batch_epoch=10,
        batch_size=64,
        num_workers=6,
        hidden_size=512,
        steps_per_epoch=512,
        val_episodes=5,
    )

    wandb_logger = WandbLogger(name=f"PPOMario-{world}-{stage}")

    trainer = pl.Trainer(
        accelerator="gpu",
        devices = 1 if torch.cuda.is_available() else None,
        max_epochs=max_episodes,
        logger=wandb_logger,
        default_root_dir=f"model/{world}-{stage}",
        check_val_every_n_epoch=20,
        gradient_clip_val= 100,
        auto_lr_find=True,
        callbacks=[checkpoint_callback, LearningRateMonitor(logging_interval='epoch'), ModelSummary(max_depth=5)],
    )
    
    if ckpt_path is not None:
        trainer.fit(model, ckpt_path=ckpt_path)
    
    else:
        trainer.fit(model)

if __name__ == "__main__":
    main(world=2, stage=2, max_episodes=100000)