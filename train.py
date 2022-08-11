import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from src.ppomario import PPOMario
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, ModelSummary

def main(world: int = 1, stage: int = 1, max_steps: int = 10000, ckpt_path: str = None, use_ppg: bool = False,):
    if use_ppg:
        run_name = f"PPOMario-PPG-{world}-{stage}"
        ckpt_save_path = f"model/ppg/{world}-{stage}/"
    else:
        run_name = f"PPOMario-PPO-{world}-{stage}"
        ckpt_save_path = f"model/ppo/{world}-{stage}/"
    
    model = PPOMario(
        world=world,
        stage=stage,
        lam=1.0,
        lr=1e-3,
        lr_decay_ratio=0,
        # lr_decay_epoch=max_episodes,
        batch_epoch=10,
        batch_size=64,
        num_workers=4,
        num_envs=8,
        hidden_size=512,
        steps_per_epoch=512,
        val_episodes=5,
        render=True,
        use_ppg=use_ppg,
        aux_batch_size=16,
        aux_batch_epoch=6,
        aux_interval=16,
    )
    
    checkpoint_callback = ModelCheckpoint(
        monitor="benchmark/avg_score",
        mode="max",
        save_top_k=3,
        dirpath=ckpt_save_path,
        filename="ppomario-{epoch}-{step}",
        every_n_epochs=50 * model.batch_epoch,
        save_last=True,
        save_on_train_epoch_end=True,
        verbose=True,
    )

    wandb_logger = WandbLogger(name=run_name)

    trainer = pl.Trainer(
        accelerator="gpu",
        devices = 1 if torch.cuda.is_available() else None,
        max_steps=max_steps,
        logger=wandb_logger,
        default_root_dir=f"model/{world}-{stage}",
        log_every_n_steps=100,
        check_val_every_n_epoch=20 * model.batch_epoch,
        reload_dataloaders_every_n_epochs=model.batch_epoch,
        num_sanity_val_steps=0,
        auto_lr_find=True,
        callbacks=[checkpoint_callback, LearningRateMonitor(logging_interval='step'), ModelSummary(max_depth=5)],
    )
    
    if ckpt_path is not None:
        trainer.fit(model, ckpt_path=ckpt_path)
    
    else:
        trainer.fit(model)

if __name__ == "__main__":
    main(world=1, stage=3, max_steps=2000000, use_ppg=False)