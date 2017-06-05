import torch
import os
import math

def findData(opt):
	maxLength = -1
	dataPaths = []

	xmlRaw = open(os.path.join(opt.args.dataRoot, opt.args.dataset + '.txt'))
	numData = opt.args.numEntry
	xml = torch.zeros(numData, opt.args.maxXmlLen)
	xmlLen = torch.zeros(numData)

	for i, line in enumerate([itm for itm in xmlRaw]):
		pieces = line.split()
		xml[i][0] = float(pieces[1])
		xml[i][1] = float(pieces[2])
		xml[i][2] = float(pieces[3])
		xml[i][3] = float(pieces[0]) # data ID
		xmlLen[i] = 4

		dataPaths.append(os.path.join('./','none'))
		if len(dataPaths[i]) + 1 > maxLength:
			maxLength = len(dataPaths[i]) + 1

	dataPath = torch.CharTensor(numData, maxLength)
	#TODO: assign values to dataPath
	return [dataPath, xml, xmlLen]

def mergeData(dataPath, opt):
	for i in range(opt.args.numEntry):
		pass
		#TODO 

def exec(opt, cacheFile):
	print("=> Generating list of audios")
	infos = findData(opt)
	dataPath = infos[0]
	xml = infos[1]
	xmlLen = infos[2]
	mergeData(dataPath, opt)

	numData = len(dataPath)
	numTrainData = math.floor(numData * opt.args.trainPctg)

	print("=> Shuffling")
	shuffle = torch.randperm(numData)
	trainDataPath = [dataPath[shuffle[i]] for i in range(numTrainData) ]
	trainXml = [xml[shuffle[i]] for i in range(numTrainData) ]
	trainXmlLen = [xmlLen[shuffle[i]] for i in range(numTrainData) ]
	valDataPath = [dataPath[shuffle[i]] for i in range(numTrainData, numData, 1) ]
	valXml = [xml[shuffle[i]] for i in range(numTrainData, numData, 1) ]
	valXmlLen = [xmlLen[shuffle[i]] for i in range(numTrainData, numData, 1) ]

	info = {'basedir' : opt.args.data,\
			'train' : {\
				'dataPath' : trainDataPath,\
				'xml' : trainXml,\
				'xmlLen' : trainXmlLen
			},\
			'val' : {\
				'dataPath' : valDataPath,\
				'xml' : valXml,\
				'xmlLen' : valXmlLen\
			}\
			}

	torch.save(info,cacheFile)
	return info