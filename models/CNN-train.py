import os
from torch.autograd import Variable
import torch.optim as optim
from tqdm import tqdm
import utils.visualize as vis

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
		trainLoss = 0
		print('\n')
		print('==> Start train epoch: ' + str(epoch))

		targetXmls = []
		outputs = []

		for i, (ipt, targetXml) in enumerate(tqdm(trainLoader)):
			ipt, targetXml = Variable(ipt), Variable(targetXml)
			self.optimizer.zero_grad()
			opt = self.model.forward(ipt)
			loss = self.criterion(opt, targetXml)
			trainLoss += loss.data[0]
			loss.backward()
			self.optimizer.step()

			if len(targetXmls) < self.opt.visTrain:
				targetXmls.append(targetXml)
				outputs.append(opt)
		if epoch % self.opt.visEpoch == 0:
			self.visualize(targetXmls, outputs, epoch, 'train', trainLoader.dataset.postprocessXml())

		avgLoss = trainLoss/len(trainLoader.dataset)
		self.logger['train'].write('%d %f\n' %(epoch, avgLoss))		
		print('==> Finish train epoch: %d' %epoch)
		print('Average loss: \033[1;36m%.6f\033[0m' %avgLoss)
		return avgLoss


	def val(self, valLoader, epoch):
		self.model.eval()
		valLoss = 0
		print('\n')
		print('==> Start val epoch: ' + str(epoch))
		targetXmls = []
		outputs = []
		for i, (ipt, targetXml) in enumerate(tqdm(valLoader)):
			ipt, targetXml = Variable(ipt), Variable(targetXml)
			opt = self.model.forward(ipt)
			loss = self.criterion(opt, targetXml)
			valLoss += loss.data[0]

			if len(targetXmls) < self.opt.visVal:
				targetXmls.append(targetXml)
				outputs.append(opt)
		if epoch % self.opt.visEpoch == 0:
			self.visualize(targetXmls, outputs, epoch, 'val', valLoader.dataset.postprocessXml())

		avgLoss = valLoss/len(valLoader.dataset)
		self.logger['val'].write('%d %f\n' %(epoch, avgLoss))
		print('==> Finish val epoch: %d' %epoch)
		print('Average loss: \033[1;36m%.6f\033[0m' %avgLoss)
		return avgLoss

	def visualize(self, targetXmls, outputs, epoch, split, postprocessXml):
		targetCaps = []
		outputCaps = []
		for i in range(len(targetXmls)):
			XmlBatch = postprocessXml(targetXmls[i])
			OptBatch = postprocessXml(outputs[i])
			for j in range(self.opt.batchSize):
				targetCaps.append('target: %.2f %.2f %.2f %.2f' %(XmlBatch[j][0].data[0], XmlBatch[j][1].data[0], XmlBatch[j][2].data[0], XmlBatch[j][3].data[0]))
				outputCaps.append('output: %.2f %.2f %.2f %.2f' %(OptBatch[j][0].data[0], OptBatch[j][1].data[0], OptBatch[j][2].data[0], OptBatch[j][3].data[0]))
		vis.writeHTML(targetCaps, outputCaps, epoch, split, self.opt)

def createTrainer(model, criterion, opt, optimState):
	trainer = myTrainer(model, criterion, opt, optimState)
	return trainer