import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch
import os
import importlib

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

	# if torch.type(model) == 'nn.DataParallelTable':
	# 	model = model.get(0)

	if opt.shareGradInput:
		pass 
		#TODO

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
		# TODO

	optimState = None
	if checkpoint != None:
		optimPath = os.path.join(opt.resume, checkpoint['optimFile'])
		assert os.path.exists(optimPath), 'Saved optimState not found: ' + optimPath
		print('=> Resuming optimState from ' + optimPath)
		optimState = torch.load(optimPath)

	return model, optimState