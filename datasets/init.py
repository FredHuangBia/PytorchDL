import torch
import os
import subprocess
import importlib

def isvalid(opt, cachePath):
	info = torch.load(cachePath)
	if info.basedir != opt.args.data:
		return False
	return True

def create(opt, split):
	cachePath = os.path.join(opt.args.gen, opt.args.dataset+'.tar')
	if not os.path.exists(cachePath) or not isvalid(cachePath):
		print(cachePath, "not found. Generating it.")
		script = opt.args.dataset+'-gen'
		gen = importlib.import_module('datasets.'+script)
		gen.exec(opt, cachePath)  #TODO
	info = torch.load(cachePath)
	dataset = importlib.import_module('datasets.'+opt.args.dataset)
	return dataset.run(info, opt, split)  # TODO

		