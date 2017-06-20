import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch
import os
import importlib
from torch.nn.parallel.data_parallel import DataParallel

def setup(opt, checkpoint, valLoader):
	model = None
	if checkpoint != None:
		modelPath = os.path.join(opt.resume, checkpoint['modelFile'])
		assert os.path.exists(modelPath), 'Saved model not found: '+modelPath
		print('=> Resuming model from ' + modelPath)
		model = torch.load(modelPath)
	else:
		print('=> Creating new model')
		models = importlib.import_module('models.' + opt.netType)
		model = models.createModel(opt)

	if isinstance(model, nn.DataParallel):
		model = model.get(0)

	if opt.resetClassifier and not checkpoint:
		pass
		#TODO

	if opt.cudnn == 'fastest':
		cudnn.fastest = True
		cudnn.benchmark = True
	elif opt.cudnn == 'deterministic':
		pass
		#TODO

	if opt.nGPUs > 1:
		gpus = opt.GPUs
		fastest, benchmark = cudnn.fastest, cudnn.benchmark
		# TODO  make a dataparallel to split data on different GPUs

	optimState = None
	if checkpoint != None:
		optimPath = os.path.join(opt.resume, checkpoint['optimFile'])
		assert os.path.exists(optimPath), 'Saved optimState not found: ' + optimPath
		print('=> Resuming optimState from ' + optimPath)
		optimState = torch.load(optimPath)

	return model, optimState