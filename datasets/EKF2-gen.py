# Generate the serialized data and label files
import torch
import os
import math
from tqdm import tqdm

overwrite = False

def normalizeRaw(type, ipt, opt):
	if type == 0: #lp
		return (ipt - (opt.lpMax + opt.lpMin)/2 ) / (opt.lpMax - opt.lpMin) * 2
	elif type == 1: #lsp
		return (ipt - (opt.lspMax + opt.lspMin)/2 ) / (opt.lspMax - opt.lspMin) * 2
	if type == 2: #eh
		return (ipt - (opt.ehMax + opt.ehMin)/2 ) / (opt.ehMax - opt.ehMin) * 2

def findData(opt):
	maxLength = -1
	dataPaths = [] #dataPaths is a py list

	xmlRaw = open(os.path.join(opt.dataRoot, opt.dataset + '.txt'))
	numData = opt.numEntry
	xml = torch.zeros(numData, opt.maxXmlLen)
	xmlLen = torch.zeros(numData)

	for i, line in enumerate([itm for itm in xmlRaw]):
		pieces = line.split()
		xml[i][0] = normalizeRaw(0, float(pieces[1]), opt)
		xml[i][1] = normalizeRaw(0, float(pieces[2]), opt)
		xml[i][2] = normalizeRaw(0, float(pieces[3]), opt)
		xml[i][3] = normalizeRaw(0, float(pieces[4]), opt)
		xml[i][4] = int(pieces[0])
		xmlLen[i] = 4

		dataPaths.append(os.path.join('./', str(math.floor(int(pieces[0])/100)), pieces[0]))
		if len(dataPaths[i]) + 1 > maxLength:
			maxLength = len(dataPaths[i]) + 1

	# dataPath = torch.CharTensor(numData, maxLength)
	#TODO: Maybe try assign values to dataPath, which is using CharTensor instead of py list
	return dataPaths, xml, xmlLen

def mergeData(dataPath, opt):
	for i in tqdm(range(opt.numEntry)):
		path = dataPath[i]
		mergedName = os.path.join(opt.data, path, 'merged.pth')

		if not os.path.exists(mergedName) or overwrite==True:
			merged = torch.zeros(opt.dataSize)
			filename = os.path.join(opt.data, path, 'data.raw')
			dataRaw = open(filename,'r')
			lines = [line for line in dataRaw] 
			dataRaw.close()
			for channel in range(opt.dataSize[0]):
				for row in range(opt.dataSize[1]):
					pieces = lines[row].split()
					for coln in range(opt.dataSize[2]):
						merged[channel][row][coln] = normalizeRaw(row, float(pieces[coln]), opt)
			torch.save(merged, mergedName)

def exec(opt, cacheFile):
	print("=> Generating list of audios")
	dataPath, xml, xmlLen = findData(opt)
	mergeData(dataPath, opt)

	numData = len(dataPath)
	numTrainData = math.floor(numData * opt.trainPctg)

	# can also choose to shuffle in netType-dataloader.py file
	print("=> Shuffling")
	shuffle = torch.randperm(numData)
	trainDataPath = [dataPath[shuffle[i]] for i in range(numTrainData) ]
	trainXml = [xml[shuffle[i]] for i in range(numTrainData) ]
	trainXmlLen = [xmlLen[shuffle[i]] for i in range(numTrainData) ]
	valDataPath = [dataPath[shuffle[i]] for i in range(numTrainData, numData, 1) ]
	valXml = [xml[shuffle[i]] for i in range(numTrainData, numData, 1) ]
	valXmlLen = [xmlLen[shuffle[i]] for i in range(numTrainData, numData, 1) ]

	info = {'basedir' : opt.data,
			'train' : {
				'dataPath' : trainDataPath,
				'xml' : trainXml,
				'xmlLen' : trainXmlLen
				},
			'val' : {
				'dataPath' : valDataPath,
				'xml' : valXml,
				'xmlLen' : valXmlLen
				}
			}

	torch.save(info, cacheFile)
	return info