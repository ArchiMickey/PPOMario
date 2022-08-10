from icecream import install
install()

import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from src.ppomario import PPOMario
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, ModelSummary
from datetime import datetime

def main(world: int = 1, stage: int = 1, ckpt_path: str = None, use_ppg: bool = False):
    if use_ppg:
        run_name = f"PPOMario-PPG-{world}-{stage}"
        ckpt_save_path = f"model/ppg/{world}-{stage}/"
    else:
        run_name = f"PPOMario-PPO-{world}-{stage}"
        ckpt_save_path = f"model/ppo/{world}-{stage}/"
        
    checkpoint_callback = ModelCheckpoint(
        save_top_k=3,
        monitor="avg_score",
        mode="max",
        dirpath=ckpt_save_path,
        filename="ppomario-{avg_score:.2f}",
        every_n_epochs=50,
        save_last=True,
        verbose=True,
    )
    
    model = PPOMario(
        world=world,
        stage=stage,
        lr=2.5e-4,
        lr_decay_ratio=0,
        lr_decay_epoch=10000,
        batch_epoch=10,
        batch_size=512,
        num_workers=4,
        num_envs=8,
        hidden_size=512,
        steps_per_epoch=512,
        val_episodes=3,
        render=False,
        use_ppg=use_ppg,
        aux_batch_size=16,
        aux_batch_epoch=6,
        aux_interval=2,
    )

    wandb_logger = WandbLogger(name=run_name, offline=True)

    trainer = pl.Trainer(
        accelerator="gpu",
        devices = 1 if torch.cuda.is_available() else None,
        max_epochs=10000,
        logger=wandb_logger,
        default_root_dir=f"model/{world}-{stage}",
        check_val_every_n_epoch=2 * model.batch_epoch,
        auto_lr_find=True,
        log_every_n_steps=20,
        callbacks=[checkpoint_callback, LearningRateMonitor(logging_interval='epoch'), ModelSummary(max_depth=5)],
        num_sanity_val_steps=0,
        # enable_progress_bar=False,
        reload_dataloaders_every_n_epochs=model.batch_epoch,
    )
    
    if ckpt_path is not None:
        trainer.fit(model, ckpt_path=ckpt_path)
    
    else:
        trainer.fit(model)

if __name__ == "__main__":
    main(world=1, stage=1, use_ppg=False)