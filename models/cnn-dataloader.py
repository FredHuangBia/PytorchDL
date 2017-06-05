import sys
sys.path.append("..")
import datasets.init as datasets


def create(opt):
	loaders = []
	
	for split in ['train','val']:
		dataset = datasets.create(opt, split)
		loaders.append(DataLoader(dataset, opt, split))
	return loaders