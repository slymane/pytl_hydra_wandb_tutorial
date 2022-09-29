# Local imports
from data import MNISTDataModule
from models import DNNClassifier

# Configs
from omegaconf import OmegaConf
import os
import hydra

# Pytorch Lightning
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

# Used by Hydra to alway execute config relative to train.py
DEFAULT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config')

@hydra.main(config_path=DEFAULT_PATH, config_name='default')
def main(config):
    # PyTL has various utility function like seed_everything which
    # Seed random number generators in : PyTorch, Numpy, Pytho.random and env var PL_GLOBAL_SEED
    pl.seed_everything(config.seed)

    # Integrating with W&B
    logger = True
    if config.use_wandb:
        logger = WandbLogger(**config.wandb)
        logger.log_hyperparams(config)

    # Instantiate model and data
    data = hydra.utils.instantiate({**config.dataset, **config.dataloader})
    model = hydra.utils.instantiate({**config.model, **config.optimizer}, shape=data.dims)
    
    # Train the model
    trainer = pl.Trainer(logger=logger, **config.trainer)
    trainer.fit(model, data)
    
    # Testing
    trainer.test()

if __name__ == '__main__':
    main()