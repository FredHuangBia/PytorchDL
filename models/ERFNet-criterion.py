import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossEntropyLoss2d(nn.Module):
	def __init__(self, weight=None):
		super().__init__()
		self.loss = nn.NLLLoss2d(weight)

	def forward(self, outputs, targets):
		return self.loss(F.log_softmax(outputs), targets)

def initCriterion(criterion, model):
	# if isinstance(criterion, nn.MultiCriterion) or isinstance(criterion, nn.ParallelCriterion):
	# 	for i in range(len(criterion.criterions)):
	# 		initCriterion(criterion.criterions[i], model)
	pass

def createCriterion(opt, model):
	weight = torch.ones(opt.numClasses)
	weight[19] = 0
	# weight[5] = 2
	# weight[6] = 2
	# weight[7] = 2
	# weight[11] = 2
	# weight[12] = 2
	# weight[17] = 2
	# weight[18] = 2
	criterion = CrossEntropyLoss2d(weight)

	return criterion
