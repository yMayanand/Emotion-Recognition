import torch
import argparse
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import transforms, models
import pytorch_lightning as pl
from dataset import EmotionDataset
from utils import freeze

class EmotionModel(pl.LightningModule):
	def __init__(self, num_classes=10, finetune=False, lr=1e-3):
		super().__init__()
		model = models.resnet18(weights=models.ResNet18_Weights)
		# unfreeze param for freeze
		unfreeze = not finetune
		freeze(model, unfreeze=unfreeze)
		self.lr = lr
		model.fc = nn.Linear(512, 10)
		self.model = model
		self.loss_func = nn.BCEWithLogitsLoss()
	  
	def forward(self, x):
		out = self.model(x)
		return out

	def configure_optimizers(self):
		optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
		return optimizer

	def training_step(self, train_batch, batch_idx):
		x, y = train_batch
		out = self.model(x)  
		loss = self.loss_func(out, y)  
		self.log('train_loss', loss)
		return loss

	def validation_step(self, val_batch, batch_idx):
		x, y = val_batch
		out = self.model(x)
		loss = self.loss_func(out, y)  
		self.log('val_loss', loss)


def main():
	# args
	parser = argparse.ArgumentParser()
	parser.add_argument(
        "--root_dir", 
        type = str,
        help = "Base folder containing the training, validation and testing folder.", 
	)

	parser.add_argument(
        "--finetune", 
        type = bool,
        help = "flag for finetuning model.", 
	)

	parser.add_argument(
        "--lr", 
		default=1e-3,
		type=float,
        help = "learning rate for model training", 
	)

	args = parser.parse_args()

	# transforms
	mean = [0.485, 0.456, 0.406]
	std = [0.229, 0.224, 0.225]

	transform = transforms.Compose([
		transforms.ToTensor(),
		transforms.Normalize(mean, std)
	])


	# data
	train_ds = EmotionDataset(
		root=args.root_dir, 
		split='train', 
		transform=transform
	)

	val_ds = EmotionDataset(
		root=args.root_dir, 
		split='valid',
		transform=transform
	)

	# dataloader
	train_loader = DataLoader(train_ds, batch_size=32)
	val_loader = DataLoader(val_ds, batch_size=32)

	# model
	model = EmotionModel(finetune=args.finetune, lr=args.lr)

	# training
	trainer = pl.Trainer(max_epochs=10, limit_train_batches=0.5)
	trainer.fit(model, train_loader, val_loader)

if __name__ == "__main__":
	main()
		
