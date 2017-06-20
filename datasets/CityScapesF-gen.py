# Generate the serialized data and label files
import torch
import os
import glob
import math
from tqdm import tqdm
from PIL import Image
import numpy as np
import subprocess

overwrite = False

def normalizeRaw(ipt):
	return (ipt - 255/2) / (255/2)

def findData(opt):
	cityscapesDataPath = os.path.join('../../data', 'cityScapes/leftImg8bit')
	searchData = os.path.join(cityscapesDataPath, "*" , "*" , "*.png" )
	dataPaths = glob.glob( searchData )
	dataPaths.sort()

	cityscapesXmlPath = os.path.join('../../data', 'label/gtFine')
	searchXml = os.path.join(cityscapesXmlPath, "*" , "*" , "*_labelTrainIds.png" )
	xmlPaths = glob.glob( searchXml )
	xmlPaths.sort()

	# dataPath = torch.CharTensor(numData, maxLength)
	#TODO: Maybe try assign values to dataPath, which is using CharTensor instead of py list
	return dataPaths, xmlPaths

def mergeData(dataPaths, xmlPaths, opt):
	for i in tqdm(range(len(dataPaths))):
		path = dataPaths[i]
		mergedName = path.replace('.png', '.pth')
		# if not os.path.exists(mergedName) or overwrite==True:
		# 	dataRaw = Image.open(path)
		# 	npRaw = np.asarray(dataRaw.convert('L'))
		# 	normRaw = normalizeRaw(npRaw)
		# 	merged = torch.from_numpy(normRaw)
		# 	dataRaw.close()
		# 	torch.save(merged, mergedName)

		path = xmlPaths[i]
		mergedName = path.replace('.png', '.pth')
		# if not os.path.exists(mergedName) or overwrite==True:
		# 	xmlRaw = Image.open(path)
		# 	npRaw = np.asarray(xmlRaw.convert('L'))
		# 	merged = torch.from_numpy(npRaw)
		# 	xmlRaw.close()
		# 	torch.save(merged, mergedName)

def exec(opt, cacheFile):
	print("=> Generating list of data")
	dataPaths, xmlPaths = findData(opt)
	mergeData(dataPaths, xmlPaths, opt)

	numData = len(dataPaths)

	trainDataPath = []
	trainXmlPath = []
	valDataPath = []
	valXmlPath = []
	testDataPath = []
	testXmlPath = []

	for i in range(numData):
		split = dataPaths[i].split('/')[5]
		if split == 'train':
			trainDataPath.append(dataPaths[i])
			trainXmlPath.append(xmlPaths[i])
		elif split == 'val':
			valDataPath.append(dataPaths[i])
			valXmlPath.append(xmlPaths[i])
		elif split == 'test':
			testDataPath.append(dataPaths[i])
			testXmlPath.append(xmlPaths[i])
		else:
			print('Unknown data split!!!')	

	print("=> Shuffling")
	numTrainData = len(trainDataPath)
	shuffle = torch.randperm(numTrainData)
	trainDataPath = [trainDataPath[shuffle[i]] for i in range(numTrainData) ]
	trainXmlPath = [trainXmlPath[shuffle[i]] for i in range(numTrainData) ]

	numValData = len(valDataPath)
	shuffle = torch.randperm(numValData)
	valDataPath = [valDataPath[shuffle[i]] for i in range(numValData) ]
	valXmlPath = [valXmlPath[shuffle[i]] for i in range(numValData) ]

	numTestData = len(testDataPath)
	shuffle = torch.randperm(numTestData)
	testDataPath = [testDataPath[shuffle[i]] for i in range(numTestData) ]
	testXmlPath = [testXmlPath[shuffle[i]] for i in range(numTestData) ]

	info = {'basedir' : opt.data,
			'train' : {
				'dataPath' : trainDataPath,
				'xmlPath' : trainXmlPath,
				},
			'val' : {
				'dataPath' : valDataPath,
				'xmlPath' : valXmlPath,
				},
			'test' : {
				'dataPath' : testDataPath,
				'xmlPath' : testXmlPath,
				}
			}

	torch.save(info, cacheFile)
	return info