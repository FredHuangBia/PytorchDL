# This code defines a customized dataset and can generate an instance of it
from torch.utils.data.dataset import *
import torch
import os

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
		return data, preprocessXml(xml)

	def __len__(self):
		return self.numEntry

	def preprocessData(ipt):
		if self.split == 'train':
			processed = ipt
			return processed
		elif self.split == 'val':
			processed = ipt
			return processed

	def preprocessXml(ipt):
		processed = ipt
		return processed

	def postprocessData():
		def process(ipt):
			processed = ipt
			return processed
		return process

	def postprocessXml():
		def process(ipt):
			processed = ipt
			return processed
		return process


def getInstance(info, opt, split):
	myInstance = myDataset(info, opt, split)
	# myInstance.__getitem__(0)
	return myInstance