# This code defines a customized dataset and can generate an instance of it
from torch.utils.data.dataset import *
import torch
import os
import torchvision.transforms as t

class exampleDataset(Dataset):
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
		return data, xml

	def __len__(self):
		return self.numEntry

	def preprocessData():
		if self.split == 'train':
			# return t.Compose([ t.Normalize(0,1), t.... ])
			pass
		elif self.split == 'val':
			# return t.Compose([ t.Normalize(0,1), t.... ])
			pass

	def postprocessData():
		pass

	def preprocessXml():
		def process(input):
			processed = torch.zeros(self.opt.outputSize)
			# do sth
			return processed
		return preprocess

	def postprocessXml():
		def process(input):
			processed = torch.zeros(self.opt.outputSize)
			# do sth
			return processed
		return preprocess


def getInstance(info, opt, split):
	myInstance = exampleDataset(info, opt, split)
	# myInstance.__getitem__(0)
	return myInstance