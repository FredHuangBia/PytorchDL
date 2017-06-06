import argparse
import torch
import os
import math
import importlib

class opts:
	def parse(self):
		parser = argparse.ArgumentParser()
		# General options
		parser.add_argument('--debug',           default=False,          type=bool,  help='Debug mode' )
		parser.add_argument('--manualSeed',      default=0,              type=int,   help='manual seed')
		parser.add_argument('--GPUs',            default='0',            type=str,   help='ID of GPUs to use, seeperate by ,')
		parser.add_argument('--backend',         default='cudnn',        type=str,   help='cudnn|cunn', choices=['cudnn', 'cunn'])
		parser.add_argument('--cudnn',           default='fastest' ,     type=str,   help='fastest|default|deterministic', choices=['fastest', 'default', 'deterministic'])
	    # Path options
		parser.add_argument('--data',            default='../../data',   type=str,   help='Path to dataset' )
		parser.add_argument('--gen',             default='../../gen',    type=str,   help='Path to generated files' )
		parser.add_argument('--resume',          default='../../models', type=str,   help='Path to checkpoint' )
		parser.add_argument('--www',             default='../../www',    type=str,   help='Path to visualization' )
	 	# Data options
		parser.add_argument('--nThreads',        default=4,              type=int,   help='Number of data loading threads' )
		parser.add_argument('--dataset',         default='pku',          type=str,   help='Name of dataset' ,choices=['pku'])
		parser.add_argument('--maxImgs',         default=10000,          type=int,   help='Number of images in train+val')
		parser.add_argument('--trainPctg',       default=0.95,           type=float, help='Percentage of training images')
	    # Training/testing options
		parser.add_argument('--nEpochs',         default=50,             type=int,   help='Number of total epochs to run')
		parser.add_argument('--epochNum',        default=0,              type=int,   help='0=retrain|-1=latest|-2=best', choices=[0,-1,-2])
		parser.add_argument('--saveEpoch',       default=10,             type=int,   help='saving at least # epochs')
		parser.add_argument('--batchSize',       default=3,              type=int,   help='mini-batch size')
		parser.add_argument('--testOnly',        default=False,          type=bool,  help='Run on validation set only')
		parser.add_argument('--visEpoch',        default=10,             type=int,   help='Visualizing every n epochs')
		parser.add_argument('--visTrain',        default=3,              type=int,   help='Visualizing training examples in unit of batchsize')
		parser.add_argument('--visTest',         default=3,              type=int,   help='Visualizing testing examples in unit of batchsize')
		parser.add_argument('--visWidth',        default=-1,             type=int,   help='# images per row for visualization')
		parser.add_argument('--tenCrop',         default=False,          type=bool,  help='Ten-crop testing')
	    # Optimization options
		parser.add_argument('--LR',              default=0.001,          type=float, help='initial learning rate')
		parser.add_argument('--LRDecay',         default='stepwise',     type=str,   help='LRDecay', choices=['anneal','stepwise','pow','none'])
		parser.add_argument('--LRDParam',        default=200,            type=float, help='param for learning rate decay')
		parser.add_argument('--momentum',        default=0.9,            type=float, help='momentum')
		parser.add_argument('--weightDecay',     default=1e-4,           type=float, help='weight decay')
	    # Model options
		parser.add_argument('--netType',         default='CNN',          type=str,   help='ANN type', choices=['CNN','MCCNN'])
		parser.add_argument('--netSpec',         default='CNN',          type=str,   help='ANN Spec', choices=['CNN'])
		parser.add_argument('--pretrain',        default='none',         type=str,   help='pretrain', choices=['none','default'])
		parser.add_argument('--absLoss',         default=0,              type=float, help='Weight for abs derender criterion')
		parser.add_argument('--mseLoss',         default=1,              type=float, help='Weight for mse derender criterion')
		parser.add_argument('--gdlLoss',         default=0,              type=float, help='Weight for gdl derender criterion')
		parser.add_argument('--customLoss',      default=0,              type=float, help='Weight for custom derender criterion')
	    # SVM options
		parser.add_argument('--featLayer',       default=18,             type=int,   help='# layer for features')
		parser.add_argument('--svmTrain',        default='',             type=str,   help='SVM training options')
		parser.add_argument('--svmTest',         default='',             type=str,   help='SVM testing options')
	    # Other model options
		parser.add_argument('--nmsThres',        default=0.5,            type=float, help='Threshold for non-max suppression')
		parser.add_argument('--shareGradInput',  default=False,          type=bool,  help='Share gradInput tensors to reduce memory usage')
		parser.add_argument('--resetClassifier', default=False,          type=bool,  help='Reset the fully connected layer for fine-tuning')
		parser.add_argument('--nClasses',        default=0,              type=int,   help='Number of classes in the dataset')
		parser.add_argument('--suffix',          default='none',         type=str,   help='Suffix to hashKey')
		self.args = parser.parse_args()

	def __init__(self):
		self.parse()
		self.args.GPUs = self.args.GPUs.split(',')
		self.args.nGPUs = len(self.args.GPUs)

		torch.set_default_tensor_type('torch.FloatTensor')
		torch.manual_seed(self.args.manualSeed)

		if self.args.dataset == 'pku':
			self.args.numEntry = 10
			self.args.dataSize = [4]
			self.args.maxXmlLen = 4 # heading dir, lat pos, lat speed, ID
		# criterions = importlib.import_module('datasets.'+self.args.dataset+'-criterion')
		elif self.args.dataset == 'EKF2':
			self.args.numEntry = 34947
			self.args.dataSize = [3,21]
			self.args.maxXmlLen = 5 # lat pos*4, ID
		# criterions = importlib.import_module('datasets.'+self.args.dataset+'-criterion')
		elif self.args.dataset == 'EKF4':
			self.args.numEntry = 34927
			self.args.dataSize = [3,41]
			self.args.maxXmlLen = 5 # lat pos*4, ID
		# criterions = importlib.import_module('datasets.'+self.args.dataset+'-criterion')

		if self.args.netType == 'cnn':
			self.args.outputSize = 4
		
		if self.args.netType == 'cnnSVM':
			opt.nEpochs = 1


		self.args.visPerInst = 4
		if self.args.visWidth == -1:
			self.args.visWidth = math.floor(8 / self.args.visPerInst) * self.args.visPerInst

		if self.args.debug:
			self.args.nEpochs = 1
			self.args.nThreads = 1

		if self.args.resetClassifier:
			assert self.args.nClasses != 0 , 'nClasses required when resetClassifier is set'

		self.args.hashKey = self.args.dataset+'_'+self.args.netType+'_'+self.args.netSpec+'_'+'pretrain='+self.args.pretrain+'_'+ \
			'Loss='+str(self.args.absLoss)+'-'+str(self.args.mseLoss)+'-'+str(self.args.gdlLoss)+'-'+str(self.args.customLoss)+'_'+\
			'LR='+str(self.args.LR)+'_'+'Suffix='+self.args.suffix

		self.args.dataRoot = self.args.data
		self.args.data = os.path.join(self.args.data, self.args.dataset)
		self.args.gen = os.path.join(self.args.gen, self.args.dataset)
		self.args.resume = os.path.join(self.args.resume, self.args.hashKey)
		self.args.www = os.path.join(self.args.www, self.args.hashKey)

		if not os.path.exists(self.args.gen):
			os.makedirs(self.args.gen)
		if not os.path.exists(self.args.resume):
			os.makedirs(self.args.resume)
		if not os.path.exists(self.args.www):
			os.makedirs(self.args.www)