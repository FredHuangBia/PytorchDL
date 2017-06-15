import sys
sys.path.append("..")
import datasets.init as datasets
from torch.utils.data.dataloader import *

class myDataLoader(DataLoader):
	def __init__(self, dataset, batch_size=1, shuffle=False, sampler=None, num_workers=0, collate_fn=default_collate, pin_memory=False, drop_last=False):
		DataLoader.__init__(self, dataset, batch_size, shuffle, sampler, num_workers, collate_fn, pin_memory, drop_last)


def create(opt):
	loaders = []
	
	for split in ['train','val']:
		dataset = datasets.create(opt, split) # generate the serialized files, then return an instance of the dataset
		loaders.append(	myDataLoader(dataset=dataset, batch_size=opt.batchSize, shuffle=True, num_workers=opt.nThreads))
	# print(loaders[0].__iter__()._next_indices())
	return loaders[0], loaders[1]