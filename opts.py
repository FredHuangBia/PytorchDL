import argparse

def parse():
	parser = argparse.ArgumentParser()

	# General options
	parser.add_argument('--debug',           default=False,          type=bool,  help='Debug mode' )
	parser.add_argument('--seed',            default=0,              type=int,   help='manual seed')
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
	parser.add_argument('--batchSize',       default=16,             type=int,   help='mini-batch size')
	parser.add_argument('--testOnly',        default=False,          type=bool,  help='Run on validation set only')
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
	parser.add_argument('--netType',         default='CNN',          type=str,   help='ANN type', choices=['CNN'])
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
	parser.add_argument('--sufix',           default='',             type=str,   help='Suffix to hashKey')
 
	opt = parser.parse_args()
	return opt

def init():
	opt = parse()
	opt.GPUs = opt.GPUs.split(',')

	return opt