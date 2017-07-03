"This code is imported by default if custom criterion is not necessary."

import torch.nn as nn

def initCriterion(criterion, model):
	# if isinstance(criterion, nn.MultiCriterion) or isinstance(criterion, nn.ParallelCriterion):
	# 	for i in range(len(criterion.criterions)):
	# 		initCriterion(criterion.criterions[i], model)
	pass

def createCriterion(opt, model):
	"Criterion is still a legacy of pytorch."
	# criterion = nn.MultiCriterion()
	# if opt.absLoss != 0:
	# 	criterion.add(nn.AbsCriterion(), weight=opt.absLoss)
	# if opt.mseLoss != 0:
	# 	criterion.add(nn.MSECriterion(), weight=opt.absLoss)
	# if opt.gdlLoss != 0:
	# 	criterion.add(nn.GDLCriterion(), weight=opt.absLoss)
	# if opt.customLoss != 0:
	# 	criterion.add(customCriterion(), weight=opt.customLoss)

	if opt.L1Loss != 0:
		criterion = nn.L1Loss()
	elif opt.mseLoss != 0:
		criterion = nn.MSELoss()
	elif opt.gdlLoss != 0:
		criterion = nn.GDLLoss()

	return criterion