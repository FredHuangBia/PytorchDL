import argparse
import torch
import os
import math
import importlib

class opts:
	def parse(self):
		parser = argparse.ArgumentParser()
		# General options
		parser.add_argument('--debug',           default=False,          type=bool,  help='Debug mode, only run 2 epochs' )
		parser.add_argument('--manualSeed',      default=0,              type=int,   help='manual seed')
		parser.add_argument('--GPU',             default=True,           type=bool,  help='Use GPU' )
		parser.add_argument('--GPUs',            default='0',            type=str,   help='ID of GPUs to use, seeperate by ,')
		parser.add_argument('--backend',         default='cudnn',        type=str,   help='backend', choices=['cudnn', 'cunn'])
		parser.add_argument('--cudnn',           default='fastest',      type=str,   help='cudnn setting', choices=['fastest', 'deterministic', 'default'])
	    # Path options
		parser.add_argument('--data',            default='../../data',   type=str,   help='Path to dataset' )
		parser.add_argument('--gen',             default='../../gen',    type=str,   help='Path to generated files' )
		parser.add_argument('--resume',          default='../../models', type=str,   help='Path to checkpoint' )
		parser.add_argument('--www',             default='../../www',    type=str,   help='Path to visualization' )
	 	# Data options
		parser.add_argument('--nThreads',        default=2,              type=int,   help='Number of data loading threads' )
		parser.add_argument('--dataset',         default='CityScapesF',  type=str,   help='Name of dataset' ,choices=['EKF2','CityScapesF'])
		parser.add_argument('--maxImgs',         default=100000,         type=int,   help='Number of images in train+val')
		parser.add_argument('--trainPctg',       default=1.00,           type=float, help='Percentage of training images')
	    # Training/testing options
		parser.add_argument('--nEpochs',         default=120,            type=int,   help='Number of total epochs to run')
		parser.add_argument('--epochNum',        default=0,              type=int,   help='0=retrain|-1=latest|-2=best', choices=[0,-1,-2])
		parser.add_argument('--saveEpoch',       default=10,             type=int,   help='saving at least # epochs')
		parser.add_argument('--saveOne',         default=True,           type=bool,  help='Only preserve one saved model')
		parser.add_argument('--batchSize',       default=2,              type=int,   help='mini-batch size')
		parser.add_argument('--dropout',         default=0.5,            type=float, help='zero rate of dropout')
		parser.add_argument('--valOnly',         default=False,          type=bool,  help='Run on validation set only')
		parser.add_argument('--testOnly',        default=False,          type=bool,  help='Run the test to see the performance')
		parser.add_argument('--visEpoch',        default=10,             type=int,   help='Visualizing every n epochs')
		parser.add_argument('--visTrain',        default=1,              type=int,   help='Visualizing training examples in unit of batchsize')
		parser.add_argument('--visVal',          default=1,              type=int,   help='Visualizing validation examples in unit of batchsize')
		parser.add_argument('--visWidth',        default=2,              type=int,   help='# images per row for visualization')
	    # Optimization options
		parser.add_argument('--LR',              default=0.01,           type=float, help='initial learning rate')
		parser.add_argument('--LRDecay',         default='none',         type=str,   help='LRDecay', choices=['anneal','stepwise','pow','none'])
		parser.add_argument('--LRDParam',        default=0,              type=float, help='param for learning rate decay')
		parser.add_argument('--momentum',        default=0.9,            type=float, help='momentum')
		parser.add_argument('--weightDecay',     default=1e-4,           type=float, help='weight decay')
		parser.add_argument('--dampening',       default=0,              type=float, help='dampening')
		parser.add_argument('--optimizer',       default='Adam',         type=str,   help='optimizer type, more choices available', choices=['SGD','Adam'])
	    # Model options
		parser.add_argument('--netType',         default='ENet',         type=str,   help='Your defined model name', choices=['CNN5','MCCNN','PSPNet','ENet'])
		parser.add_argument('--netSpec',         default='resnet34',     type=str,   help='Other model to be loaded', choices=['custom','resnet101','resnet34'])
		parser.add_argument('--pretrain',        default=True,           type=bool,  help='Pretrained or not')
		parser.add_argument('--L1Loss',          default=0,              type=float, help='Weight for abs derender criterion')
		parser.add_argument('--mseLoss',         default=0,              type=float, help='Weight for mse derender criterion')
		parser.add_argument('--ceLoss',          default=0,              type=float, help='Weight for cross-entrophy derender criterion')
		parser.add_argument('--gdlLoss',         default=0,              type=float, help='Weight for gdl derender criterion')
		parser.add_argument('--customLoss',      default=1,              type=float, help='Weight for custom derender criterion')
	    # Other model options
		parser.add_argument('--nmsThres',        default=0.5,            type=float, help='Threshold for non-max suppression')
		parser.add_argument('--resetClassifier', default=False,          type=bool,  help='Reset the fully connected layer for fine-tuning')
		parser.add_argument('--numClasses',      default=20,             type=int,   help='Number of classes in the dataset')
		parser.add_argument('--suffix',          default='none',         type=str,   help='Suffix for saving the model')
		self.args = parser.parse_args()

	def __init__(self):
		self.parse()
		self.args.GPUs = self.args.GPUs.split(',')
		self.args.nGPUs = len(self.args.GPUs)

		torch.set_default_tensor_type('torch.FloatTensor')
		torch.manual_seed(self.args.manualSeed)

		# Customized parameters for each dataset and net
		if self.args.dataset == 'EKF2':
			self.args.numEntry = 28779
			self.args.dataSize = [1,3,21]
			self.args.maxXmlLen = 5
			self.args.lpMin = -1.791070
			self.args.lpMax = 8.823801
			self.args.lspMin = -1.892717
			self.args.lspMax = 1.976027
			self.args.ehMin = -0.067614
			self.args.ehMax = 0.064971
		elif self.args.dataset == 'CityScapesF':
			self.args.downRate = 1.0

		if self.args.netType == 'CNN5':
			self.args.outputSize = 4

		elif self.args.netType == 'MCCNN':
			self.args.outputSize = 4

		elif self.args.netType == 'PSPNet':
			self.args.cropedSize = (1024, 2048)
			self.args.branchOut = 128

		self.args.visPerInst = 4
		if self.args.visWidth == -1:
			self.args.visWidth = math.floor(8 / self.args.visPerInst) * self.args.visPerInst

		if self.args.debug:
			self.args.nEpochs = 1
			self.args.nThreads = 1

		if self.args.resetClassifier:
			assert self.args.numClasses != 0 , 'numClasses required when resetClassifier is set'

		self.args.hashKey = self.args.dataset+'_'+self.args.netType+'_'+self.args.netSpec+'_'+'pretrain='+str(self.args.pretrain)+'_'+ \
			'Loss='+str(self.args.L1Loss)+'-'+str(self.args.mseLoss)+'-'+str(self.args.gdlLoss)+'-'+str(self.args.customLoss)+'_'+\
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