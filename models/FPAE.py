import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch
import torch.autograd as autograd  # test purpose
import torch.nn.functional as F
from torchvision import models
import os

class nonBt1dMain(nn.Module):
	def __init__(self, inChannel, outChannel, kSize, dropout, prelus, dilated):
		super().__init__()
		pad = int((kSize-1)/2)

		self.conv1a = nn.Conv2d(inChannel, outChannel, (kSize,1), stride=1, padding=(pad,0))
		self.nonLinear1a = prelus and nn.PReLU(outChannel) or nn.ReLU(True)
		self.conv1b = nn.Conv2d(outChannel, outChannel, (1,kSize), stride=1, padding=(0,pad))
		self.bn1 = nn.BatchNorm2d(outChannel, eps=1e-3)
		self.nonLinear1b = prelus and nn.PReLU(outChannel) or nn.ReLU(True)

		self.conv2a = nn.Conv2d(inChannel, outChannel, (kSize,1), stride=1, padding=(pad*dilated,0), dilation=(dilated,1))
		self.nonLinear2 = prelus and nn.PReLU(outChannel) or nn.ReLU(True)
		self.conv2b = nn.Conv2d(outChannel, outChannel, (1,kSize), stride=1, padding=(0,pad*dilated), dilation=(1,dilated))
		self.bn2 = nn.BatchNorm2d(outChannel, eps=1e-3)
		self.dropout = nn.Dropout(dropout)

	def forward(self, x):
		y = self.conv1a(x)
		y = self.nonLinear1a(y)
		y = self.conv1b(y)
		y = self.bn1(y)
		y = self.nonLinear1b(y)

		y = self.conv2a(y)
		y = self.nonLinear2(y)
		y = self.conv2b(y)
		y = self.bn2(y)
		y = self.dropout(y)

		return y


class nonBt1d(nn.Module):
	def __init__(self, inChannel, outChannel, kSize, dropout, prelus, dilated):
		super().__init__()

		self.main = nonBt1dMain(inChannel, outChannel, kSize, dropout, prelus, dilated)
		self.nonLinear = prelus and nn.PReLU(outChannel) or nn.ReLU(True)

	def forward(self, x):
		y = self.main(x)
		y += x
		y = self.nonLinear(y)

		return y


class downsampler(nn.Module):
	def __init__(self, inChannel, outChannel, kSize, dropout, prelus):
		super().__init__()
		pad = int((kSize-1)/2)

		self.main = nn.Conv2d(inChannel, outChannel-inChannel, kSize, stride=2, padding=pad)
		self.other = nn.MaxPool2d(2, stride=2)
		self.bn = nn.BatchNorm2d(outChannel, eps=1e-3)
		self.dropout = nn.Dropout(dropout)
		self.nonLinear = prelus and nn.PReLU(outChannel) or nn.ReLU(True)

	def forward(self, x):
		main = self.main(x)
		other = self.other(x)
		y = torch.cat([main, other], 1)
		y = self.bn(y)
		y = self.dropout(y)
		y = self.nonLinear(y)

		return y


class upsamplerA(nn.Module):
	def __init__(self, inChannel, outChannel):
		super().__init__()
		# inputStride = upsample and 2 or 1
		self.deconv = nn.ConvTranspose2d(inChannel, outChannel, 3, stride=2, padding=1, output_padding=1)
		self.bn = nn.BatchNorm2d(outChannel, eps=1e-3)
		self.nonLinear = nn.ReLU(True)
		self.scalar1 = nn.Parameter(torch.rand(1))
		self.scalar2 = nn.Parameter(torch.rand(1))
		
		self.conv1a = nn.Conv2d(outChannel, outChannel, (3,1), stride=1, padding=(1,0))
		self.nonLinear1a = nn.ReLU(True)
		self.conv1b = nn.Conv2d(outChannel, outChannel, (1,3), stride=1, padding=(0,1))
		self.bn1 = nn.BatchNorm2d(outChannel, eps=1e-3)
		self.nonLinear1b = nn.ReLU(True)

	def forward(self, x, mid):
		y = self.deconv(x)
		y = y*self.scalar1.expand_as(y) + mid*self.scalar2.expand_as(y)
		y = self.bn(y)
		y = self.nonLinear(y)
		y = self.conv1a(y)
		y = self.nonLinear1a(y)
		y = self.conv1b(y)
		y = self.bn1(y)
		y = self.nonLinear1b(y)

		return y

class upsamplerB(nn.Module):
	def __init__(self, inChannel, outChannel):
		super().__init__()
		# inputStride = upsample and 2 or 1
		self.deconv = nn.ConvTranspose2d(inChannel, outChannel, 3, stride=2, padding=1, output_padding=1)
		self.bn = nn.BatchNorm2d(outChannel, eps=1e-3)
		self.nonLinear = nn.ReLU(True)

	def forward(self, x):
		y = self.deconv(x)
		y = self.bn(y)
		y = self.nonLinear(y)

		return y

class encoderA(nn.Module):
	def __init__(self, prelus=False, dropout=0.3):
		super().__init__()

		self.downsampler1 = downsampler(3, 16, 3, 0, False)
		self.downsampler2 = downsampler(16, 64, 3, dropout/10.0, False)

		self.conv1a = nonBt1d(64, 64, 3, dropout/10.0, prelus, 1)
		self.conv1b = nonBt1d(64, 64, 3, dropout/10.0, prelus, 2)
		self.conv1c = nonBt1d(64, 64, 3, dropout/10.0, prelus, 4)
		self.conv1d = nonBt1d(64, 64, 3, dropout/10.0, prelus, 8)
		self.conv1e = nonBt1d(64, 64, 3, dropout/10.0, prelus, 16)

	def forward(self, x):
		y = self.downsampler1(x)
		y = self.downsampler2(y)
		y = self.conv1a(y)
		y = self.conv1b(y)
		y = self.conv1c(y)
		y = self.conv1d(y)
		y = self.conv1e(y)

		return y

class encoderB(nn.Module):
	def __init__(self, prelus=False, dropout=0.3):
		super().__init__()

		self.downsampler3 = downsampler(64, 128, 3, dropout, False)

		self.conv2a = nonBt1d(128, 128, 3, dropout, prelus, 2)
		self.conv2b = nonBt1d(128, 128, 3, dropout, prelus, 4)
		self.conv2c = nonBt1d(128, 128, 3, dropout, prelus, 8)
		self.conv2d = nonBt1d(128, 128, 3, dropout, prelus, 16)

		self.conv3a = nonBt1d(128, 128, 3, dropout, prelus, 2)
		self.conv3b = nonBt1d(128, 128, 3, dropout, prelus, 4)
		self.conv3c = nonBt1d(128, 128, 3, dropout, prelus, 8)
		self.conv3d = nonBt1d(128, 128, 3, dropout, prelus, 16)

	def forward(self, x):
		y = self.downsampler3(x)
		y = self.conv2a(y)
		y = self.conv2b(y)
		y = self.conv2c(y)
		y = self.conv2d(y)
		y = self.conv3a(y)
		y = self.conv3b(y)
		y = self.conv3c(y)
		y = self.conv3d(y)

		return y

class encoderC(nn.Module):
	def __init__(self, prelus=False, dropout=0.3):
		super().__init__()

		self.downsampler4 = nn.MaxPool2d(2, stride=2)

		self.conv4a = nonBt1d(128, 128, 3, dropout, prelus, 1)
		self.conv4b = nonBt1d(128, 128, 3, dropout, prelus, 1)

	def forward(self, x):
		y = self.downsampler4(x)
		y = self.conv4a(y)
		y = self.conv4b(y)

		return y

class encoder(nn.Module):
	def __init__(self, prelus=False, dropout=0.3):
		super().__init__()

		self.encoder1 = encoderA(prelus, dropout)
		self.encoder2 = encoderB(prelus, dropout)
		self.encoder3 = encoderC(prelus, dropout)
		self.encoder4 = encoderC(prelus, dropout)
		self.encoder5 = encoderC(prelus, dropout)
		self.encoder6 = encoderC(prelus, dropout)
		self.encoder7 = encoderC(prelus, dropout)

	def forward(self, x):
		y = self.encoder1(x)
		y1 = y.clone()
		y = self.encoder2(y)
		y2 = y.clone()
		y = self.encoder3(y)
		y3 = y.clone()
		y = self.encoder4(y)
		y4 = y.clone()
		y = self.encoder5(y)
		y5 = y.clone()
		y = self.encoder6(y)
		y6 = y.clone()
		y = self.encoder7(y)

		return y1, y2, y3, y4, y5, y6, y

class decoder(nn.Module):
	def __init__(self, numClasses, prelus=False):
		super().__init__()

		self.upsampler7 = upsamplerA(128, 128)
		self.conv7 = nonBt1d(128, 128, 3, 0.1, prelus, 1)
		self.upsampler6 = upsamplerA(128, 128)
		self.conv6 = nonBt1d(128, 128, 3, 0.1, prelus, 1)
		self.upsampler5 = upsamplerA(128, 128)
		self.conv5 = nonBt1d(128, 128, 3, 0.1, prelus, 1)
		self.upsampler4 = upsamplerA(128, 128)
		self.conv4 = nonBt1d(128, 128, 3, 0.1, prelus, 1)
		self.upsampler3 = upsamplerA(128, 128)
		self.conv3 = nonBt1d(128, 128, 3, 0.1, prelus, 1)

		self.upsampler2 = upsamplerA(128, 64)
		self.conv2a = nonBt1d(64, 64, 3, 0.1, prelus, 2)
		self.conv2b = nonBt1d(64, 64, 3, 0.1, prelus, 4)

		self.upsampler1 = upsamplerB(64, numClasses)
		self.conv1a = nonBt1d(numClasses, numClasses, 3, 0.1, prelus, 2)
		self.conv1b = nonBt1d(numClasses, numClasses, 3, 0.1, prelus, 4)
		self.conv1c = nonBt1d(numClasses, numClasses, 3, 0.1, prelus, 8)

		self.convFinal = nn.ConvTranspose2d(numClasses, numClasses, 2, stride=2)

	def forward(self, y1, y2, y3, y4, y5, y6, y7):
		y = self.upsampler7(y7, y6)
		y = self.conv7(y)
		y = self.upsampler6(y, y5)
		y = self.conv6(y)
		y = self.upsampler5(y, y4)
		y = self.conv5(y)
		y = self.upsampler4(y, y3)
		y = self.conv4(y)
		y = self.upsampler3(y, y2)
		y = self.conv3(y)
		y = self.upsampler2(y, y1)
		y = self.conv2a(y)
		y = self.conv2b(y)
		y = self.upsampler1(y)
		y = self.conv1a(y)
		y = self.conv1b(y)
		y = self.conv1c(y)
		y = self.convFinal(y)

		return y


class myModel(nn.Module):
	def __init__(self, opt):
		super().__init__()
		self.Encoder = encoder(opt.prelus, opt.dropout)
		self.Decoder = decoder(opt.numClasses, opt.prelus)	

	def forward(self, x):
		y1, y2, y3, y4, y5, y6, y7 = self.Encoder(x)
		y = self.Decoder(y1, y2, y3, y4, y5, y6, y7)
		return y


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