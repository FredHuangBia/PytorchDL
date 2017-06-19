# This code defines a customized dataset and can generate an instance of it
from torch.utils.data.dataset import *
import torch
import os
import numpy as np
import utils.transforms as t
from scipy import misc

def normalizeRaw(ipt):
	return (ipt - 255/2) / (255/2)

class myDataset(Dataset):
	def __init__(self, info, opt, split):
		self.split = split
		self.dataInfo = info[split]
		self.numEntry = len(self.dataInfo['xmlPath'])
		self.opt = opt
		self.dir = info['basedir']

	def __getitem__(self, index):
		dataPath = self.dataInfo['dataPath'][index]
		# data = torch.load(dataPath)
		dataRaw = misc.imread(dataPath)
		dataRaw = misc.imresize(dataRaw, 0.125)
		npRaw = np.asarray(dataRaw, dtype=np.float32)
		npRaw = np.swapaxes(npRaw, 0, 2)
		npRaw = np.swapaxes(npRaw, 1, 2)
		normRaw = normalizeRaw(npRaw)
		data = torch.from_numpy(normRaw)

		xmlPath = self.dataInfo['xmlPath'][index]
		# xml = torch.load(xmlPath)
		xmlRaw = misc.imread(xmlPath)
		npRaw = np.asarray(xmlRaw, dtype=np.int64)
		xml = torch.from_numpy(npRaw)

		return self.preprocessData(data), self.preprocessXml(xml)

	def __len__(self):
		return self.numEntry

	def preprocessData(self, ipt):
		if self.split == 'train':
			processed = t.addNoise(ipt, 0, 0.001)
			return processed
		elif self.split == 'val':
			processed = ipt
			return processed

	def preprocessXml(self, ipt):
		processed = ipt
		return processed

	def postprocessData(self):
		def process(ipt):
			processed = ipt
			return processed
		return process

	def postprocessXml(self):
		def process(ipt):
			processed = ipt
			return processed
		return process

def getInstance(info, opt, split):
	myInstance = myDataset(info, opt, split)
	return myInstance