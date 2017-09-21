import argparse
import torch
import os
import math
import importlib

class opts:
	def parse(self):
		parser = argparse.ArgumentParser()
		# General options
		parser.add_argument('--GPU',             default=True,           type=bool,  help='Use GPU' )
		parser.add_argument('--GPUs',            default='0',            type=str,   help='ID of GPUs to use, seperate by ,')
		parser.add_argument('--backend',         default='cudnn',        type=str,   help='backend', choices=['cudnn', 'cunn'])
		parser.add_argument('--cudnn',           default='fastest',      type=str,   help='cudnn setting', choices=['fastest', 'deterministic', 'default'])
		parser.add_argument('--debug',           default=False,          type=bool,  help='Debug mode, only run 2 epochs' )
		parser.add_argument('--manualSeed',      default=0,              type=int,   help='manual seed')
		# Path options
		parser.add_argument('--data',            default='../../data',   type=str,   help='Path to dataset' )
		parser.add_argument('--gen',             default='../../gen',    type=str,   help='Path to generated files' )
		parser.add_argument('--resume',          default='../../models', type=str,   help='Path to checkpoint' )
		parser.add_argument('--www',             default='../../www',    type=str,   help='Path to visualization' )
		# Data options
		parser.add_argument('--dataset',         default='EKF2',         type=str,   help='Name of dataset' ,choices=['EKF2','CityScapesF'])
		parser.add_argument('--nThreads',        default=4,              type=int,   help='Number of data loading threads' )
		parser.add_argument('--trainPctg',       default=0.95,           type=float, help='Percentage of training images')
		# Training/testing options
		parser.add_argument('--nEpochs',         default=120,            type=int,   help='Number of total epochs to run')
		parser.add_argument('--epochNum',        default=0,              type=int,   help='0=retrain|-1=latest|-2=best', choices=[0,-1,-2])
		parser.add_argument('--batchSize',       default=16,             type=int,   help='mini-batch size')
		parser.add_argument('--saveEpoch',       default=20,             type=int,   help='saving at least # epochs')
		parser.add_argument('--saveOne',         default=True,           type=bool,  help='Only preserve one saved model')
		parser.add_argument('--valOnly',         default=False,          type=bool,  help='Run on validation set only')
		parser.add_argument('--testOnly',        default=False,          type=bool,  help='Run the test to see the performance')
		parser.add_argument('--visEpoch',        default=20,             type=int,   help='Visualizing every n epochs')
		parser.add_argument('--visBatch',        default=1,              type=int,   help='Number of examples to visualize, in unit of batchsize')
		parser.add_argument('--visWidth',        default=1,              type=int,   help='Number of images per row for visualization')
		# Optimization options
		parser.add_argument('--LR',              default=0.01,           type=float, help='initial learning rate')
		parser.add_argument('--LRDecay',         default='none',         type=str,   help='LRDecay method', choices=['anneal','stepwise','pow','none'])
		parser.add_argument('--LRDParam',        default=0,              type=float, help='param for learning rate decay')
		parser.add_argument('--momentum',        default=0.9,            type=float, help='momentum')
		parser.add_argument('--weightDecay',     default=1e-4,           type=float, help='weight decay')
		parser.add_argument('--dampening',       default=0,              type=float, help='dampening')
		parser.add_argument('--dropout',         default=0.1,            type=float, help='zero rate of dropout')
		parser.add_argument('--optimizer',       default='Adam',         type=str,   help='optimizer type, more choices available', choices=['SGD','Adam'])
		# Model options
		parser.add_argument('--netType',         default='CNN5',         type=str,   help='Your defined model name', choices=['CNN5','MCCNN','PSPNet','ENet','ERFNet'])
		parser.add_argument('--netSpec',         default='custom',       type=str,   help='Other model to be loaded', choices=['custom','resnet101','resnet50'])
		parser.add_argument('--pretrain',        default=False,          type=bool,  help='Pretrained or not')
		parser.add_argument('--L1Loss',          default=1,              type=float, help='Weight for abs criterion')
		parser.add_argument('--mseLoss',         default=0,              type=float, help='Weight for mse criterion')
		parser.add_argument('--gdlLoss',         default=0,              type=float, help='Weight for gdl criterion')
		parser.add_argument('--customLoss',      default=0,              type=float, help='Weight for custom criterion')
		# Other model options
		parser.add_argument('--resetClassifier', default=False,          type=bool,  help='Reset the fully connected layer for fine-tuning')
		parser.add_argument('--numClasses',      default=20,             type=int,   help='Number of classes in the dataset')
		parser.add_argument('--suffix',          default='none',         type=str,   help='Suffix for saving the model')
		self.args = parser.parse_args()

	def __init__(self):
		self.parse()
		self.args.GPUs = [int(i) for i in self.args.GPUs.split(',')]
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
			self.args.classRates = [0.326340837398497, 0.053859626385344174, 0.20191867267384248, 0.0058039854354217275, 0.007771590096609933, 0.010862099463198365, 0.001844829751663849, 0.0048919963035262935, 0.14084477047960298, 0.010252431821422417, 0.03549583755621389, 0.010771288671413389, 0.001193280580664883, 0.061949212050237575, 0.0023681590937766708, 0.0020829861103987494, 0.002061850283326221, 0.00087287710494354, 0.0036623760832457984, 0.11515129265665006]
			self.args.CSFMean = (0.28689553743650931, 0.32513302306918557, 0.28389177263033455)
			self.args.CSFStd = (0.17613641034404787, 0.18099167376255773, 0.17772230936699546)

		if self.args.netType == 'CNN5':
			self.args.outputSize = 4

		elif self.args.netType == 'MCCNN':
			self.args.outputSize = 4

		elif self.args.netType == 'PSPNet':
			self.args.downRate = 1.0
			self.args.branchOut = 128

		elif self.args.netType == 'ERFNet':
			self.args.prelus = False
			self.args.downRate = 0.5
			self.args.encoderOnly = False
			self.args.encoderPath = '/home/titan/Fred/segment/models/CityScapesF_ERFNet_custom_pretrain=False_Loss=0-0-0-1_LR=0.0005_Suffix=encoder/model_best.pth'

		elif self.args.netType == 'ERFNet2':
			self.args.prelus = False
			self.args.downRate = 0.5
			self.args.encoderOnly = False
			self.args.encoderPath = '/home/titan/Fred/segment/models/CityScapesF_ERFNet2_custom_pretrain=False_Loss=0-0-0-1_LR=0.0005_Suffix=encoder/model_best.pth'

		elif self.args.netType == 'FPAE':
			self.args.prelus = False
			self.args.downRate = 0.5
			self.args.encoderOnly = False

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