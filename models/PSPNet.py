"code borrow and modified from piwise on GitHub and modified"
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch
import torch.autograd as autograd  # test purpose
import torch.nn.functional as F
from torchvision import models

class PSPDec(nn.Module):
	def __init__(self, in_features, out_features, downsize):
		super().__init__()
		self.downsize = downsize
		self.conv = nn.Conv2d(in_features, out_features, 1, bias=False)
		self.bn = nn.BatchNorm2d(out_features, momentum=.95)
		self.relu = nn.ReLU(inplace=True)

	def forward(self, x):
		if self.downsize == None:
			downsize = (1,1)
		else:
			downsize = ( int(x.size()[2]/self.downsize[0]), int(x.size()[3]/self.downsize[1]) )
		upsize = (x.size()[2], x.size()[3])

		output = F.avg_pool2d(x, downsize, stride=downsize)
		output = self.conv(output)
		output = self.bn(output)
		output = self.relu(output)
		output = F.upsample_bilinear(output, upsize)
		return output

class myModel(nn.Module):
	def __init__(self, opt):
		super().__init__()
		self.opt = opt

		if opt.netSpec == 'resnet101':
			resnet = models.resnet101(pretrained=opt.pretrain)
		elif opt.netSpec == 'resnet34':
			resnet = models.resnet34(pretrained=opt.pretrain)

		self.conv1 = resnet.conv1
		self.layer1 = resnet.layer1
		self.layer2 = resnet.layer2
		self.layer3 = resnet.layer3
		self.layer4 = resnet.layer4

		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				m.stride = 1
				m.requires_grad = False
			if isinstance(m, nn.BatchNorm2d):
				m.requires_grad = False

		self.layer5a = PSPDec(512, 128, (1,1))
		self.layer5b = PSPDec(512, 128, (3,3))
		self.layer5c = PSPDec(512, 128, (8,16))
		self.layer5d = PSPDec(512, 128, (32,64))
		self.layer5e = PSPDec(512, 128, None)

		self.final = nn.Sequential(
			nn.Conv2d(128*5, 128, 3, padding=1, bias=False),
			nn.BatchNorm2d(128, momentum=.95),
			nn.ReLU(inplace=True),
			nn.Dropout(.1),
			nn.Conv2d(128, opt.numClasses, 1),
		)

	def forward(self, x):
		x = self.conv1(x)
		x = self.layer1(x)
		x = self.layer2(x)
		x = self.layer3(x)
		x = self.layer4(x)
		x = torch.cat([
			self.layer5a(x),
			self.layer5b(x),
			self.layer5c(x),
			self.layer5d(x),
			self.layer5e(x),
		], 1)
		final = self.final(x)
		upsize = ( int(x.size()[2]/self.opt.downRate), int(x.size()[3]/self.opt.downRate) )
		final = F.upsample_bilinear(final, upsize )
		return final

def createModel(opt):
	model = myModel(opt)
	if opt.GPU:
		model = model.cuda()
	return model