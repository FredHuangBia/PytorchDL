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


class upsampler(nn.Module):
	def __init__(self, inChannel, outChannel):
		super().__init__()
		# inputStride = upsample and 2 or 1
		self.deconv = nn.ConvTranspose2d(inChannel, outChannel, 3, stride=2, padding=1, output_padding=1)
		self.bn = nn.BatchNorm2d(outChannel, eps=1e-3)
		self.nonLinear = nn.ReLU(True)

		self.conv1 = nn.Conv2d(outChannel*2, outChannel, 3, stride=1, padding=1)
		self.bn1 = nn.BatchNorm2d(outChannel, eps=1e-3)
		self.nonLinear1 = nn.ReLU(True)
		self.conv2 = nn.Conv2d(outChannel, outChannel, 3, stride=1, padding=2, dilation=2)
		self.bn2 = nn.BatchNorm2d(outChannel, eps=1e-3)
		self.nonLinear2 = nn.ReLU(True)

	def forward(self, x, mid):
		y = self.deconv(x)
		y = self.bn(y)
		y = self.nonLinear(y)
		y = torch.cat([y, mid], 1)
		y = self.conv1(y)
		y = self.bn1(y)
		y = self.nonLinear1(y)
		y = self.conv2(y)
		y = self.bn2(y)
		y = self.nonLinear2(y)

		return y


class encoder(nn.Module):
	def __init__(self, prelus=False, dropout=0.3):
		super().__init__()

		self.downsampler1 = downsampler(3, 16, 3, 0, False)
		self.downsampler2 = downsampler(16, 64, 3, dropout/10.0, False)

		self.conv1a = nonBt1d(64, 64, 3, dropout/10.0, prelus, 1)
		self.conv1b = nonBt1d(64, 64, 3, dropout/10.0, prelus, 1)
		self.conv1c = nonBt1d(64, 64, 3, dropout/10.0, prelus, 1)
		self.conv1d = nonBt1d(64, 64, 3, dropout/10.0, prelus, 1)
		self.conv1e = nonBt1d(64, 64, 3, dropout/10.0, prelus, 1)

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
		y = self.downsampler1(x)
		mid1 = y.clone()
		y = self.downsampler2(y)
		y = self.conv1a(y)
		y = self.conv1b(y)
		y = self.conv1c(y)
		y = self.conv1d(y)
		y = self.conv1e(y)
		mid2 = y.clone()
		y = self.downsampler3(y)
		y = self.conv2a(y)
		y = self.conv2b(y)
		y = self.conv2c(y)
		y = self.conv2d(y)
		y = self.conv3a(y)
		y = self.conv3b(y)
		y = self.conv3c(y)
		y = self.conv3d(y)

		return y, mid1, mid2


class encoderPred(nn.Module):
	def __init__(self, numClasses, prelus=False, dropout=0.3):
		super().__init__()

		self.encoder = encoder(prelus, dropout)
		self.convFinal = nn.Conv2d(128, numClasses, 1, 1)

	def forward(self, x):
		y,_,_ = self.encoder(x)
		y = self.convFinal(y)

		return y, None, None


class decoder(nn.Module):
	def __init__(self, numClasses, prelus=False):
		super().__init__()

		self.upsampler1 = upsampler(128, 64)
		self.conv1a = nonBt1d(64, 64, 3, 0, prelus, 1)
		self.conv1b = nonBt1d(64, 64, 3, 0, prelus, 1)

		self.upsampler2 = upsampler(64, 16)
		self.conv2a = nonBt1d(16, 16, 3, 0, prelus, 1)
		self.conv2b = nonBt1d(16, 16, 3, 0, prelus, 1)

		self.convFinal = nn.ConvTranspose2d(16, numClasses, 2, stride=2)

	def forward(self, x, mid1, mid2):
		y = self.upsampler1(x, mid2)
		y = self.conv1a(y)
		y = self.conv1b(y)
		y = self.upsampler2(y, mid1)
		y = self.conv2a(y)
		y = self.conv2b(y)
		y = self.convFinal(y)

		return y


class myModel(nn.Module):
	def __init__(self, opt):
		super().__init__()
		self.encoderOnly = opt.encoderOnly

		if self.encoderOnly:
			self.Encoder = encoderPred(opt.numClasses, opt.prelus, opt.dropout)

		elif opt.epochNum==0: #retrain combined model
			EncoderPred = torch.load(opt.encoderPath)
			self.Encoder = EncoderPred.Encoder.encoder
			self.Decoder = decoder(opt.numClasses, opt.prelus)

		elif not self.encoderOnly and not opt.epochNum==0:
			self.Encoder = encoder(opt.prelus, opt.dropout)
			self.Decoder = decoder(opt.numClasses, opt.prelus)			

	def forward(self, x):
		y, mid1, mid2 = self.Encoder(x)
		if not self.encoderOnly:
			y = self.Decoder(y, mid1, mid2)
			return y
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