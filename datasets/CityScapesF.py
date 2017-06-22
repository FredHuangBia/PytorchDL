# This code defines a customized dataset and can generate an instance of it
from torch.utils.data.dataset import *
import torch
import os
import numpy as np
import utils.transforms as t
from scipy import misc

colorMap = {
						0:(128, 64,128), 
						1:(244, 35,232), 
						2:( 70, 70, 70), 
						3:(102,102,156), 
						4:(190,153,153), 
						5:(153,153,153), 
						6:(250,170, 30), 
						7:(220,220,  0), 
						8:(107,142, 35), 
						9:(152,251,152), 
						10:(70,130,180), 
						11:(220, 20, 60), 
						12:( 19,  0,  0), 
						13:(  0,  0,142), 
						14:(  0,  0, 70), 
						15:(  0, 60,100), 
						16:(  0, 80,100), 
						17:(  0,  0,230), 
						18:(119, 11, 32), 
						19:(81,  0, 81), 
}

class myDataset(Dataset):
	def __init__(self, info, opt, split):
		self.split = split
		self.dataInfo = info[split]
		self.numEntry = len(self.dataInfo['xmlPath'])
		self.opt = opt
		self.dir = info['basedir']

	def __getitem__(self, index):
		dataPath = self.dataInfo['dataPath'][index]
		dataRaw = misc.imread(dataPath)
		dataRaw = misc.imresize(dataRaw, self.opt.downRate)
		dataRaw = np.asarray(dataRaw, dtype=np.uint8)

		xmlPath = self.dataInfo['xmlPath'][index]
		xmlRaw = misc.imread(xmlPath)
		xmlRaw = np.asarray(xmlRaw, dtype=np.int64)

		dataRaw, xmlRaw = self.preprocess(dataRaw, xmlRaw)

		dataRaw = np.swapaxes(dataRaw, 0, 2)
		dataRaw = np.swapaxes(dataRaw, 1, 2)
		data = torch.from_numpy(dataRaw)

		xml = torch.from_numpy(xmlRaw)

		return data, xml

	def __len__(self):
		return self.numEntry

	def preprocess(self, data, xml):
		if self.split == 'train':
			data, xml = t.randomSizeCrop(data, xml, 0.8)
			data, xml = t.randomFlip(data, xml)
			data = t.scaleRGB(data)
			data = t.addNoise(data, 0, 0.005)
			data = t.normalize(data, self.opt.ImNetMean, self.opt.ImNetStd)
			return data, xml
		else:
			data = t.scaleRGB(data)
			data = t.normalize(data, self.opt.ImNetMean, self.opt.ImNetStd)
			return data, xml			

	def postprocessData(self):
		def process(ipt):
			processed = np.swapaxes(ipt, 0, 2)
			processed = np.swapaxes(processed, 0, 1)
			processed = t.unNormalize(processed, self.opt.ImNetMean, self.opt.ImNetStd)
			processed = t.unScaleRGB(processed)
			return processed
		return process

	def postprocessXml(self):
		def process(ipt):
			processed = np.zeros((ipt.shape[0], ipt.shape[1], 3))
			for i in range(ipt.shape[0]):
				for j in range(ipt.shape[1]):
					processed[i,j,0] = colorMap[ipt[i,j]][0]
					processed[i,j,1] = colorMap[ipt[i,j]][1]
					processed[i,j,2] = colorMap[ipt[i,j]][2]
			return processed			
		return process

def getInstance(info, opt, split):
	myInstance = myDataset(info, opt, split)
	return myInstance