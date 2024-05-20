import logging
import os

import hydra
import torch
from avg_ckpts import ensemble
from datamodule.data_module import DataModule
from lightning import ModelModule
from pytorch_lightning import seed_everything, Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.plugins import DDPPlugin


# Set environment variables and logger level
os.environ["WANDB_SILENT"] = "true"
logging.basicConfig(level=logging.WARNING)


@hydra.main(config_path="conf", config_name="config")
def main(cfg):
    seed_everything(42, workers=True)
    #import pdb;pdb.set_trace()
    #cfg.slurm_job_id = os.environ["SLURM_JOB_ID"]
    cfg.gpus = torch.cuda.device_count()

    checkpoint = ModelCheckpoint(
        monitor="monitoring_step",
        mode="max",
        dirpath=os.path.join(cfg.exp_dir, cfg.exp_name) if cfg.exp_dir else None,
        save_last=True,
        filename="{epoch}",
        save_top_k=cfg.checkpoint.save_top_k,
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")
    callbacks = [checkpoint, lr_monitor]
    #callbacks = [checkpoint]
    # Configure logger
    wandb_logger = hydra.utils.instantiate(cfg.logger) if cfg.log_wandb else None

    # Set modules and trainer
    modelmodule = ModelModule(cfg)
    datamodule = DataModule(cfg)
    trainer = Trainer(
        **cfg.trainer,
        logger=wandb_logger,
        callbacks=callbacks,
        strategy=DDPPlugin(find_unused_parameters=False) if cfg.gpus > 1 else None
    )
    
    modelmodule.model.load_state_dict(
        torch.load(cfg.ckpt_path, map_location=lambda storage, loc: storage)
    )
    trainer.test(model=modelmodule, datamodule=datamodule)


if __name__ == "__main__":
    main()
