from opts import *

if __name__=='__main__':
	opt = opts().args

	models = importlib.import_module('models.init')
	DataLoader = importlib.import_module('models.' + opt.netType + '-dataloader')
	checkpoints = importlib.import_module('checkpoints')
	# criterions = importlib.import_module('criterions.init')
	# Trainer = importlib.import_module('models.' + opt.netType + '-train')

	print('=> Setting up data loader')
	trainLoader, valLoader = DataLoader.create(opt)

	print('=> Checking checkpoints')
	checkpoint = checkpoints.load(opt)



	# for i, (data, xml) in enumerate(trainLoader):
	# 	print(data)