from icecream import install
install()

import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from src.ppomario import PPOMario
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, ModelSummary
from datetime import datetime

def main(world: int = 1, stage: int = 1, ckpt_path: str = None, use_ppg: bool = False):
    
    checkpoint_callback = ModelCheckpoint(
        save_top_k=3,
        monitor="avg_score",
        mode="max",
        dirpath=f"model/{world}-{stage}/",
        filename="ppomario-{avg_score:.2f}",
        every_n_train_steps=5000,
        save_last=True,
    )
    
    model = PPOMario(
        world=world,
        stage=stage,
        lr=2.5e-4,
        lr_decay_ratio=0,
        lr_decay_epoch=100000,
        nb_optim_iters=1,
        batch_epoch=1,
        batch_size=4,
        num_workers=6,
        hidden_size=512,
        steps_per_epoch=128,
        val_episodes=3,
        render=True,
        use_ppg=use_ppg,
        aux_batch_epoch=9,
        aux_interval=16,
    )

    wandb_logger = WandbLogger(name=f"PPOMario-{world}-{stage}", offline=True)

    trainer = pl.Trainer(
        accelerator="gpu",
        devices = 1 if torch.cuda.is_available() else None,
        max_epochs=100000,
        logger=wandb_logger,
        default_root_dir=f"model/{world}-{stage}",
        check_val_every_n_epoch=15,
        auto_lr_find=True,
        callbacks=[checkpoint_callback, LearningRateMonitor(logging_interval='epoch'), ModelSummary(max_depth=5)],
        # enable_progress_bar=False,
        # num_sanity_val_steps=0,
    )
    
    if ckpt_path is not None:
        trainer.fit(model, ckpt_path=ckpt_path)
    
    else:
        trainer.fit(model)

if __name__ == "__main__":
    main(world=1, stage=1, use_ppg=True)