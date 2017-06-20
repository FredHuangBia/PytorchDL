import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch
import torch.autograd as autograd  # test purpose
import torch.nn.functional as F
from torchvision import models

class PSPDec(nn.Module):
	def __init__(self, in_features, out_features, downsize, upsize):
		super().__init__()

		self.features = nn.Sequential(
			nn.AvgPool2d(downsize, stride=downsize),
			nn.Conv2d(in_features, out_features, 1, bias=False),
			nn.BatchNorm2d(out_features, momentum=.95),
			nn.ReLU(inplace=True),
			nn.UpsamplingBilinear2d(upsize)
		)

	def forward(self, x):
		return self.features(x)

class myModel(nn.Module):
	def __init__(self, opt):
		super().__init__()

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

		height = int(opt.cropedSize[0]*opt.downRate)
		width = int(opt.cropedSize[1]*opt.downRate)
		hw = (height, width)

		self.layer5a = PSPDec(512, 128, (int(hw[0]/1),  int(hw[1]/1)), hw)
		self.layer5b = PSPDec(512, 128, (int(hw[0]/4),  int(hw[1]/8)), hw)
		self.layer5c = PSPDec(512, 128, (int(hw[0]/16), int(hw[1]/32)), hw)
		self.layer5d = PSPDec(512, 128, (int(hw[0]/64), int(hw[1]/128)), hw)
		self.layer5e = PSPDec(512, 128, (1, 1), hw)

		self.final = nn.Sequential(
			nn.Conv2d(128*5, 128, 3, padding=1, bias=False),
			nn.BatchNorm2d(128, momentum=.95),
			nn.ReLU(inplace=True),
			nn.Dropout(.1),
			nn.Conv2d(128, opt.numClasses, 1),
		)

	def forward(self, x):
		# print('x', x.size())
		x = self.conv1(x)
		# print('conv1', x.size())
		x = self.layer1(x)
		# print('layer1', x.size())
		x = self.layer2(x)
		# print('layer2', x.size())
		x = self.layer3(x)
		# print('layer3', x.size())
		x = self.layer4(x)
		# print('layer4', x.size())
		x = torch.cat([
			self.layer5a(x),
			self.layer5b(x),
			self.layer5c(x),
			self.layer5d(x),
			self.layer5e(x),
		], 1)
		# print('cated', x.size())
		final = self.final(x)
		# print('final', final.size())
		# final = F.upsample_bilinear(final, x.size()[2:])
		final = F.upsample_bilinear(final, (1024,2048))
		# print('upsample', final.size())

		return final

def createModel(opt):
	model = myModel(opt)
	if opt.GPU:
		model = model.cuda()
	return model