# This code defines a customized dataset and can generate an instance of it
from torch.utils.data.dataset import *
import torch
import os
import numpy as np
import utils.transforms as t

class myDataset(Dataset):
	def __init__(self, info, opt, split):
		self.split = split
		self.dataInfo = info[split]
		self.numEntry = len(self.dataInfo['xml'])
		self.opt = opt
		self.dir = info['basedir']

	def __getitem__(self, index):
		path = self.dataInfo['dataPath'][index]
		data = torch.load(os.path.join(self.dir, path, 'merged.pth'))
		xml = self.dataInfo['xml'][index]
		xmlLen = self.dataInfo['xmlLen'][index] # may not be always useful
		return self.preprocessData(data), self.preprocessXml(xml)

	def __len__(self):
		return self.numEntry

	def preprocessData(self, ipt):
		if self.split == 'train':
			processed = t.addNoise(ipt, 0, 0.01)
			return processed
		elif self.split == 'val':
			processed = ipt
			return processed

	def preprocessXml(self, ipt):
		processed = torch.zeros(self.opt.outputSize)
		processed[0] = ipt[0]
		processed[1] = ipt[1]
		processed[2] = ipt[2]
		processed[3] = ipt[3]
		return processed

	def postprocessData(self):
		def process(ipt):
			processed = ipt
			return processed
		return process

	def postprocessXml(self):
		def process(ipt):
			processed = ipt * (self.opt.lpMax - self.opt.lpMin) + self.opt.lpMin
			return processed
		return process

def getInstance(info, opt, split):
	myInstance = myDataset(info, opt, split)
	# myInstance.__getitem__(0)
	return myInstance