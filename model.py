import timm
import torch.nn as nn
from torchvision import models

class ResnetModel(nn.Module):
	def __init__(self, num_classes=10):
		super().__init__()
		model = models.resnet18()
		model.fc = nn.Linear(512, 10)
		self.model = model

	def forward(self, x):
		out = self.model(x)
		return out

class EffnetModel(nn.Module):
	def __init__(self, num_classes=10) -> None:
		super().__init__()
		model = timm.create_model('efficientnet_b0', num_classes=10)
		self.model = model

	def forward(self, x):
		out = self.model(x)
		return out
		