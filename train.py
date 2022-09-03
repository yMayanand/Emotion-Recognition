import torch
import timm
import argparse
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import transforms, models
import pytorch_lightning as pl
from dataset import EmotionDataset
from utils import accuracy

class EmotionModel(pl.LightningModule):
	def __init__(self, num_classes=10, finetune=False, lr=1e-3, wd=0., label_smoothing=0.):
		super().__init__()
		# model = models.resnet18(weights=models.ResNet18_Weights)
		model = timm.create_model('efficientnet_b0', pretrained=True, num_classes=10)
	
		self.lr = lr
		self.wd = wd
		model.fc = nn.Linear(512, 10)
		self.model = model
		self.loss_func = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
	  
	def forward(self, x):
		out = self.model(x)
		return out

	def configure_optimizers(self):
		optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.wd)
		#sched = torch.optim.lr_scheduler.StepLR(optimizer, 10, gamma=0.95)
		#opt_dict = {"optimizer": optimizer, "lr_scheduler": sched}
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
		acc = accuracy(out, y)
		self.log('val_loss', loss)
		self.log('val_acc', acc)


def main():
	# args
	parser = argparse.ArgumentParser()
	parser.add_argument(
		"--root_dir", 
		type = str,
		help = "Base folder containing the training, validation and testing folder.", 
	)

	parser.add_argument(
		"--lr", 
		default=1e-3,
		type=float,
		help = "learning rate for model training", 
	)

	parser.add_argument(
		"--epochs", 
		default=10,
		type=int,
		help = "number of epochs for which the training will be done", 
	)

	parser.add_argument(
		"--wd", 
		default=0.,
		type=float,
		help = "weight decay", 
	)

	parser.add_argument(
		"--label_smoothing", 
		default=0.,
		type=float,
		help = "label smoothing", 
	)


	args = parser.parse_args()

	# transforms
	mean = [0.485, 0.456, 0.406]
	std = [0.229, 0.224, 0.225]

	val_transform = transforms.Compose([
		transforms.ToTensor(),
		transforms.Normalize(mean, std)
	])
	
	train_transform = transforms.Compose([
		transforms.ToTensor(),
		transforms.RandomHorizontalFlip(),
		transforms.Resize(size=(96, 96)), 
		transforms.RandomResizedCrop(48, scale=(0.75, 1.)),
		transforms.Normalize(mean, std)
	])


	# data
	train_ds = EmotionDataset(
		root=args.root_dir, 
		split='train', 
		transform=train_transform
	)

	val_ds = EmotionDataset(
		root=args.root_dir, 
		split='valid',
		transform=val_transform
	)

	# dataloader
	train_loader = DataLoader(train_ds, shuffle=True, batch_size=64)
	val_loader = DataLoader(val_ds, batch_size=64)

	# model
	model = EmotionModel(finetune=args.finetune, lr=args.lr)
	# model.load_from_checkpoint("/content/Emotion-Recognition/lightning_logs/version_2/checkpoints/epoch=19-step=4460.ckpt")
	# training
	trainer = pl.Trainer(accelerator='gpu', devices=1, max_epochs=args.epochs, limit_train_batches=0.5)
	trainer.fit(model, train_loader, val_loader)

if __name__ == "__main__":
	main()

