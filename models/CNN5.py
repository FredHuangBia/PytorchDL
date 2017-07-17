import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch
import torch.autograd as autograd  # test purpose
import torch.nn.functional as F

class myModel(nn.Module):
	def __init__(self, opt):
		nn.Module.__init__(self)
		self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(3,5), stride=(1,1), padding=(0,2))
		self.bn1 = nn.BatchNorm2d(32)
		self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(1,5), stride=(1,1), padding=(0,2))
		self.bn2 = nn.BatchNorm2d(64)
		self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(1,3), stride=(1,1), padding=(0,1))
		self.bn3 = nn.BatchNorm2d(128)
		self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(1,3), stride=(1,1), padding=(0,1))
		self.bn4 = nn.BatchNorm2d(256)
		self.conv5 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(1,3), stride=(1,1), padding=(0,1))
		self.bn5 = nn.BatchNorm2d(256)
		self.fc1 = nn.Linear(384*2, 4)
		self.dropout = opt.dropout

	def forward(self, x):
		x = self.conv1(x)
		x = self.bn1(x)
		x = F.relu(x)

		x = self.conv2(x)
		x = self.bn2(x)
		x = F.avg_pool2d(x, kernel_size=(1,2), stride=(1,2), padding=(0,1))
		x = F.relu(x)

		x = self.conv3(x)
		x = self.bn3(x)
		x = F.avg_pool2d(x, kernel_size=(1,2), stride=(1,2), padding=(0,1))
		x = F.relu(x)

		x = self.conv4(x)
		x = self.bn4(x)
		x = F.relu(x)

		x = self.conv5(x)
		x = self.bn5(x)
		x = F.avg_pool2d(x, kernel_size=(1,2), stride=(1,2), padding=0)
		x = F.relu(x)

		x = x.view(-1,384*2)
		x = F.dropout(x, p=self.dropout)
		x = self.fc1(x)

		return x


class myParallelModel(nn.Module):
	def __init__(self, opt):
		super().__init__()
		self.opt = opt

		self.model = myModel(opt)
		self.model = nn.DataParallel(self.model, opt.GPUs)

	def forward(self, x):
		x = self.model(x)
		return x


def createModel(opt):
	if opt.GPU:
		if opt.nGPUs > 1:
			model = myParallelModel(opt)
		else:
			model = myModel(opt)
		model = model.cuda()
	return model