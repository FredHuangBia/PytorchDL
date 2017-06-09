from opts import *
import sys
import math

if __name__=='__main__':
	opt = opts().args

	models = importlib.import_module('models.init')
	DataLoader = importlib.import_module('models.' + opt.netType + '-dataloader')
	checkpoints = importlib.import_module('checkpoints')
	criterions = importlib.import_module('criterions.init')
	Trainer = importlib.import_module('models.' + opt.netType + '-train')

	print('=> Setting up data loader')
	trainLoader, valLoader = DataLoader.create(opt)

	print('=> Checking checkpoints')
	checkpoint = checkpoints.load(opt)

	print('=> Setting up model and criterion')
	model, optimState = models.setup(opt, checkpoint, valLoader)
	criterion = criterions.setup(opt, checkpoint, model)

	print('=> Loading trainer')
	trainer = Trainer.createTrainer(model, criterion, opt, optimState)

	bestLoss = math.inf
	startEpoch = max([1, opt.epochNum])
	if checkpoint != None:
		startEpoch = checkpoint['epoch'] + 1
		bestLoss = checkpoint['loss']

	if opt.valOnly:
		trainer.val(valLoader, startEpoch-1)
		sys.exit()

	if opt.testOnly:
		trainer.test(valLoader, startEpoch-1)
		sys.exit()

	bestModel = False
	for epoch in range(startEpoch, opt.nEpochs+1):
		trainLoss = trainer.train(trainLoader, epoch)
		valLoss = trainer.val(valLoader, epoch)

		if valLoss < bestLoss:
			bestModel = True
			print(' * Best model: \033[1;36m%.6f\033[0m * ' %valLoss)
			bestLoss = valLoss

		checkpoints.save(epoch, trainer.model, criterion, trainer.optimState, bestModel, valLoss ,opt)
