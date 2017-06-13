import torch
import importlib
import os

def setup(opt, checkpoint, model):
	criterion = None
	if opt.customLoss == 0:
		criterionHandler = importlib.import_module('criterions.criterion')
	else:
		criterionHandler = importlib.import_module('models.' + opt.netType + '-criterion')

	if checkpoint != None:
		criterionPath = os.path.join(opt.resume, checkpoint['criterionFile'])
		assert os.path.exists(criterionPath), 'Saved criterion not found: ' + criterionPath
		print('=> Resuming criterion from ' + criterionPath)
		criterion = torch.load(criterionPath)
		criterionHandler.initCriterion(criterion, model)
	else:
		print('=> Creating new criterion')
		criterion = criterionHandler.createCriterion(opt, model)

	return criterion