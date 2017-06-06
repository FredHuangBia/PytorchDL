# This code initializes the data set by generating serialized data and label information. 
# The create function will return a instance of the specified dataset.
import torch
import os
import subprocess
import importlib

def isvalid(opt, cachePath):
	info = torch.load(cachePath)
	if info['basedir'] != opt.data:
		return False
	return True

def create(opt, split):
	cachePath = os.path.join(opt.gen, opt.dataset+'.pth')
	if not os.path.exists(cachePath) or not isvalid(opt, cachePath):
		print(cachePath, "not found. Generating it.")
		script = opt.dataset+'-gen'
		gen = importlib.import_module('datasets.'+script)
		gen.exec(opt, cachePath) 
	info = torch.load(cachePath)
	dataset = importlib.import_module('datasets.'+opt.dataset)
	return dataset.getInstance(info, opt, split)  

		