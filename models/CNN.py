import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch
import torch.autograd as autograd  # test purpose
import torch.nn.functional as F

class myModel(nn.Module):
	def __init__(self, opt):
		nn.Module.__init__(self)
		self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(3,5), stride=(1,1), padding=(0,2))
		self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(1,5), stride=(1,1), padding=(0,2))
		self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(1,3), stride=(1,1), padding=(0,1))
		self.conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(1,3), stride=(1,1), padding=(0,1))
		self.fc = nn.Linear(192, 4)	
		self.dropout = opt.dropout

	def forward(self, x):
		x = self.conv1(x)
		x = F.relu(x)
		x = self.conv2(x)
		x = F.max_pool2d(x, kernel_size=(1,2), stride=(1,2), padding=(0,1))
		x = F.relu(x)
		x = self.conv3(x)
		x = F.max_pool2d(x, kernel_size=(1,2), stride=(1,2), padding=(0,1))
		x = F.relu(x)
		x = self.conv4(x)
		x = F.avg_pool2d(x, kernel_size=(1,2), stride=(1,2), padding=0)
		x = F.relu(x)
		x = x.view(-1,192)
		x = F.dropout(x, p=self.dropout)
		x = self.fc(x)
		#TODO softmax or sth
		return x

	# def _initialize_weights(self):
	# 	self.conv1.weight.data.copy_(_get_orthogonal_init_weights(self.conv1.weight) * sqrt(2))
	# 	self.conv2.weight.data.copy_(_get_orthogonal_init_weights(self.conv2.weight) * sqrt(2))

def createModel(opt):
	model = myModel(opt)
	ipt = autograd.Variable(torch.randn(16,1,3,21))
	opt = model.forward(ipt)
	print(opt)
	return model