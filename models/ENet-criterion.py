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
	criterion = CrossEntropyLoss2d()

	return criterion
