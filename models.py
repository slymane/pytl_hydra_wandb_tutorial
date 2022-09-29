from abc import ABC, abstractmethod
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.metrics import Accuracy
import torch
import torch.nn as nn
from torchvision.datasets.mnist import MNIST
from torchvision import transforms

# Abstract class to inherit from for all classifiers
class Classifier(pl.LightningModule, ABC):
	def __init__(self, lr, shape):
		super(Classifier, self).__init__()
		self.lr = lr
		self.accuracy = Accuracy()
		self.example_input_array = torch.rand(shape)

	@abstractmethod
	def forward(self, x):
		pass

	def eval_batch(self, batch):
		# Make predictions
		x, y = batch
		y_hat = self(x)

		# Evaluate predictions
		loss = nn.functional.cross_entropy(y_hat, y)
		acc = self.accuracy(y_hat, y)

		return loss, acc

	def training_step(self, batch, batch_idx):
		# Evaluate batch
		loss, acc = self.eval_batch(batch)

		# Configure result
		result = pl.TrainResult(loss)
		result.log('train_loss', loss)
		result.log('train_acc', acc)
		return result

	def validation_step(self, batch, batch_idx):
		# Evaluate batch
		loss, acc = self.eval_batch(batch)

		# Configure result
		result = pl.EvalResult(checkpoint_on=loss, early_stop_on=loss)
		result.log('val_loss', loss)
		result.log('val_acc', acc)
		return result

	def test_step(self, batch, batch_idx):
		# Evaluate batch
		loss, acc = self.eval_batch(batch)

		# Configure result
		result = pl.EvalResult()
		result.log('test_loss', loss)
		result.log('test_acc', acc)
		return result

	def configure_optimizers(self):
		return torch.optim.Adam(self.parameters(), lr=self.lr)


class DNNClassifier(Classifier):
	def __init__(self, hidden_layers, lr, shape):
		super(DNNClassifier, self).__init__(lr, shape)

		self.i2h = nn.Linear(np.product(shape), hidden_layers[0])
		self.h2h = nn.ModuleList(
			[nn.Linear(l1, l2) for l1, l2 in zip(hidden_layers[:-1], hidden_layers[1:])]
		)
		self.h2o = nn.Linear(hidden_layers[-1], 10)
		
	def forward(self, x):
		x = x.flatten(start_dim=1)
		x = torch.relu(self.i2h(x))
		for l in self.h2h:
			x = torch.relu(l(x))
		x = self.h2o(x)
		return x


class CNNClassifier(Classifier):
	def __init__(self, hidden_layers, lr, shape):
		super(CNNClassifier, self).__init__(lr, shape)

		self.i2h = nn.Conv2d(1, hidden_layers[0], 3)
		self.h2h = nn.ModuleList(
			[nn.Conv2d(l1, l2, 3) for l1, l2 in zip(hidden_layers[:-1], hidden_layers[1:])]
		)
		self.h2o = nn.Linear(hidden_layers[-1] * (23 - len(hidden_layers))**2, 10)
		
	def forward(self, x):
		x = torch.relu(self.i2h(x))

		for l in self.h2h:
			x = torch.relu(l(x))
		x = x.flatten(start_dim=1)
		x = self.h2o(x)
		return x
