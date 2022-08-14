import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import transforms, models
import pytorch_lightning as pl
from dataset import EmotionDataset

class EmotionModel(pl.LightningModule):
	def __init__(self, num_classes=10):
		super().__init__()
		model = models.resnet18(weights=models.ResNet18_Weights)
		model.fc = nn.Linear(512, 10)
		self.model = model
		self.loss_func = nn.BCEWithLogitsLoss()
	  
	def forward(self, x):
		out = self.model(x)
		return out

	def configure_optimizers(self):
		optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
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
	# transforms
	mean = [0.485, 0.456, 0.406]
	std = [0.229, 0.224, 0.225]

	transform = transforms.Compose([
		transforms.ToTensor(),
		transforms.Normalize(mean, std)
	])


	# data
	train_ds = EmotionDataset(
		root='/content/Dataset/FERPlus/data', 
		split='train', 
		transform=transform
	)

	val_ds = EmotionDataset(
		root='/content/Dataset/FERPlus/data', 
		split='valid',
		transform=transform
	)

	# dataloader
	train_loader = DataLoader(train_ds, batch_size=32)
	val_loader = DataLoader(val_ds, batch_size=32)

	# model
	model = EmotionModel()

	# training
	trainer = pl.Trainer(limit_train_batches=0.5)
	trainer.fit(model, train_loader, val_loader)

if __name__ == "__main__":
	main()
		
