import os
from torch.autograd import Variable
import torch.optim as optim
from tqdm import tqdm
import utils.visualize

colorH = '\[\033[1;36m\]'
colorT = '\[\033[0m\]'

class myTrainer():
	def __init__(self, model, criterion, opt, optimState):
		self.model = model
		self.criterion = criterion
		self.optimState = optimState
		if self.optimState == None:
			self.optimState = { 'learningRate' : opt.LR,
								'learningRateDecay' : opt.LRDParam,
								'momentum' : opt.momentum,
								'nesterov' : True,
								'dampening'  : opt.dampening,
								'weightDecay' : opt.weightDecay
							}
		self.opt = opt
		self.optimizer = optim.SGD(model.parameters(), lr = opt.LR, momentum=opt.momentum)
		self.logger = { 'train' : open(os.path.join(opt.resume, 'train.log'), 'a+'), 
						'val' : open(os.path.join(opt.resume, 'val.log'), 'a+')
					}

	def train(self, trainLoader, epoch):
		self.model.train()
		print('\n')
		print('==> Start train epoch: ' + str(epoch))
		loss = None
		for i, (ipt, targetXml) in enumerate(tqdm(trainLoader)):
			ipt, targetXml = Variable(ipt), Variable(targetXml)
			self.optimizer.zero_grad()
			opt = self.model.forward(ipt)
			loss = self.criterion(opt, targetXml)
			loss.backward()
			self.optimizer.step()
		
		print('Finish train epoch: %d' %epoch)
		print('Finish loss: \033[1;36m%.6f\033[0m <==' %loss.data[0])
		return loss.data[0]


	def val(self, valLoader, epoch):
		self.model.eval()
		valLoss = 0
		print('\n')
		print('==> Start val epoch: ' + str(epoch))
		for i, (ipt, targetXml) in enumerate(tqdm(valLoader)):
			ipt, targetXml = Variable(ipt), Variable(targetXml)
			opt = self.model.forward(ipt)
			loss = self.criterion(opt, targetXml)
			valLoss += loss.data[0]
		avgLoss = valLoss/len(valLoader.dataset)

		print('Finish val epoch: %d' %epoch)
		print('Average loss: \033[1;36m%.6f\033[0m <==' %avgLoss)
		return avgLoss

	def visualize(self):
		pass

def createTrainer(model, criterion, opt, optimState):
	trainer = myTrainer(model, criterion, opt, optimState)
	return trainer