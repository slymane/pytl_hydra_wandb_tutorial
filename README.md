# Simplifying Code With PyTorch Lightning, Hydra, and Weights & Biases 

## Overview

This repository hosts a sample project for MNIST digit classification using PyTorch Lightning and Hydra atop a PyTorch core. [config](config/) defines experiment configuration for Hydra as a structured hierarchy of .yaml files. When executing [train.py](train.py), Hydra will load these files and pass the result to main as the variable *config*. Hydra is able to instantiate objects from these configs, which can be seen in lines *30* and *31* to create the data and model objects. These two objects are both PyTorch Lightning modules, defined in [data.py](data.py) and [model.py](model.py) respectively. By creating these as PyTorch Lightning modules, we are then able to train the model by simply calling `trainer.fit(model, data)`.

## Hydra

[Hydra](https://hydra.cc/docs/intro/) is an open-source Python framework that simplifies the development of research and other complex applications. The key feature is the ability to dynamically create a hierarchical configuration by composition and override it through config files and the command line.

### Key Features

- Hierarchical configuration composable from multiple sources
- Configuration can be specified or overridden from the command line
- Dynamic command line tab completion
- Run your application locally or launch it to run remotely
- Run multiple jobs with different arguments with a single command

## PyTorch Lightning

[PyTorch Lightning](https://pytorch-lightning.readthedocs.io/en/stable/) is a open-source Python library providing a high-level interface for PyTorch. The key idea is to remove boilerplate and engineering code so that researchers can focus on what matters.


### Key Features
- Scale your models to run on any hardware (CPU, GPUs, TPUs) without changing your model
- Making code more readable by decoupling the research code from the engineering
Easier to reproduce
- Less error prone by automating most of the training loop and tricky engineering
- Keeps all the flexibility (LightningModules are still PyTorch modules), but removes a ton of boilerplate
- Lightning has out-of-the-box integration with the popular logging/visualizing frameworks (Tensorboard, MLFlow, Neptune.ai, Comet.ml, Wandb).
- Minimal running speed overhead (about 300 ms per epoch compared with pure PyTorch).
- Automated features such as: Distributed training, early stopping, checkpointing, and more

## Weights and Biases

[Weights & Biases](https://wandb.ai/site) is a experiment tracking, dataset versioning, and model management platform with built in web-based visualization dashboard.
