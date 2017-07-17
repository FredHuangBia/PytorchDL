import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch
import torch.autograd as autograd  # test purpose
import torch.nn.functional as F
from torch.autograd import Variable

class myBranch(nn.Module):
	def __init__(self, opt):
		nn.Module.__init__(self)
		self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(1,5), stride=(1,1), padding=(0,2))
		self.bn1 = nn.BatchNorm2d(16)
		self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(1,5), stride=(1,1), padding=(0,2))
		self.bn2 = nn.BatchNorm2d(32)
		self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(1,3), stride=(1,1), padding=(0,1))
		self.bn3 = nn.BatchNorm2d(64)
		self.conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(1,3), stride=(1,1), padding=(0,1))
		self.bn4 = nn.BatchNorm2d(64)


	def forward(self, x):
		x = self.conv1(x)
		x = F.relu(x)
		x = self.bn1(x)
		x = self.conv2(x)
		x = self.bn2(x)
		x = F.max_pool2d(x, kernel_size=(1,2), stride=(1,2), padding=(0,1))
		x = F.relu(x)
		x = self.conv3(x)
		x = self.bn3(x)
		x = F.max_pool2d(x, kernel_size=(1,2), stride=(1,2), padding=(0,1))
		x = F.relu(x)
		x = self.conv4(x)
		x = self.bn4(x)
		x = F.avg_pool2d(x, kernel_size=(1,2), stride=(1,2), padding=0)
		x = F.relu(x)
		x = x.view(-1,192)

		return x	

class myModel(nn.Module):
	def __init__(self, opt):
		nn.Module.__init__(self)
		self.br1 = myBranch(opt)
		self.br2 = myBranch(opt)
		self.br3 = myBranch(opt)
		self.fc1 = nn.Linear(192*3, 64)	
		self.fc2 = nn.Linear(64, 4)
		self.bn1 = nn.BatchNorm1d(192*3)
		self.bn2 = nn.BatchNorm1d(64)
		self.dropout = opt.dropout

	def forward(self, x):
		batchSize = x.size()[0]
		x1 = torch.zeros(batchSize, 1, 1, 21)
		x2 = torch.zeros(batchSize, 1, 1, 21)
		x3 = torch.zeros(batchSize, 1, 1, 21)
		for b in range(batchSize):
			for t in range(21):
				x1[b,0,0,t] = x.data[b,0,0,t]
				x2[b,0,0,t] = x.data[b,0,1,t]
				x3[b,0,0,t] = x.data[b,0,2,t]
		x1, x2, x3 = Variable(x1), Variable(x2), Variable(x3)
		x1, x2, x3 = self.br1.forward(x1), self.br2.forward(x2), self.br2.forward(x3)

		x = torch.cat([x1, x2, x3], 1)
		x = self.bn1(x)
		x = F.dropout(x, p=self.dropout)
		x = self.fc1(x)
		x = self.bn2(x)
		x = F.dropout(x, p=self.dropout)
		x = self.fc2(x)

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