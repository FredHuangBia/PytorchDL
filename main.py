from opts import *

if __name__=='__main__':
	opt = opts()

	models = importlib.import_module('models.init')
	#criterion
	DataLoader = importlib.import_module('models.' + opt.args.netType + '-dataloader')

	print('=> Setting up data loader')
	DataLoaders = DataLoader.create(opt)
	# trainLoader = DataLoaders[0]
	# valLoader = DataLoaders[1]