import torch.nn as nn
# # Criterion is still a legacy of pytorch

def initCriterion(criterion, model):
	# if isinstance(criterion, nn.MultiCriterion) or isinstance(criterion, nn.ParallelCriterion):
	# 	for i in range(len(criterion.criterions)):
	# 		initCriterion(criterion.criterions[i], model)
	pass

def createCriterion(opt, model):
	# criterion = nn.MultiCriterion()
	# if opt.absLoss != 0:
	# 	criterion.add(nn.AbsCriterion(), weight=opt.absLoss)
	# if opt.mseLoss != 0:
	# 	criterion.add(nn.MSECriterion(), weight=opt.absLoss)
	# if opt.gdlLoss != 0:
	# 	criterion.add(nn.GDLCriterion(), weight=opt.absLoss)
	# if opt.customLoss != 0:
	# 	criterion.add(customCriterion(), weight=opt.customLoss)

	if opt.absLoss != 0:
		criterion = nn.AbsLoss()
	elif opt.mseLoss != 0:
		criterion = nn.MSELoss()
	elif opt.ceLoss != 0:
		criterion = nn.CrossEntropyLoss()
	elif opt.gdlLoss != 0:
		criterion = nn.GDLLoss()
	elif opt.customLoss != 0:
		criterion = customLoss()

	return criterion

# class customCriterion(Criterion):
# 	def __init__():
# 		nn.Criterion.__init__(self)