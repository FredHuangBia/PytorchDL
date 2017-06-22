import os
import math
import numpy as np
from PIL import Image

def renderTable(targetCaps, outputCaps, ncoln):
	assert len(targetCaps) == len(outputCaps), 'Target and output Caps dimension mismatch.'
	htmlStr = '<table border="1">\n'
	for row in range(math.ceil(len(targetCaps)/ncoln)):
		htmlStr +=  '<tr>'
		for coln in range(ncoln):
			htmlStr += '<td>'
			ID = row*ncoln + coln
			if ID < len(targetCaps):
				htmlStr += '<br>'
				htmlStr += targetCaps[ID]
				htmlStr += '<br>'
				htmlStr += outputCaps[ID]
				htmlStr += '<br>'
			htmlStr += '</td>'
		htmlStr += '</tr>\n'
	htmlStr += '</table>\n'
	return htmlStr

def renderImgTable(inputImgs, targetImgs, outputImgs, ncoln, imgDir):
	assert len(targetImgs) == len(outputImgs), 'Target and output Imgs dimension mismatch.'
	htmlStr = '<table border="1">\n'
	for row in range(math.ceil(len(targetImgs)/ncoln)):
		htmlStr +=  '<tr>'
		for coln in range(ncoln):
			ID = row*ncoln + coln
			htmlStr += '<td>'
			img = Image.fromarray(np.uint8(inputImgs[ID]), 'RGB')
			img.save(os.path.join(imgDir, 'input-%d.png'%ID))
			img = Image.fromarray(np.uint8(targetImgs[ID]), 'RGB')
			img.save(os.path.join(imgDir, 'target-%d.png'%ID))
			img = Image.fromarray(np.uint8(outputImgs[ID]), 'RGB')
			img.save(os.path.join(imgDir, 'output-%d.png'%ID))
			htmlStr += '<img src="imgs/input-%d.png" style="width:800px;height:400px;">' %ID
			htmlStr += '<img src="imgs/target-%d.png" style="width:800px;height:400px;">' %ID
			htmlStr += '<img src="imgs/output-%d.png" style="width:800px;height:400px;">' %ID
			htmlStr += '</td>'
		htmlStr += '</tr>\n'
	htmlStr += '</table>\n'
	return htmlStr

def writeHTML(targetCaps, outputCaps, epoch, split, opt):
	print('==> Writing visualize html ...')
	htmlFile = 'index.html'
	rootDir = os.path.join(opt.www, split + '_' + str(epoch))
	if not os.path.exists(rootDir):
		os.makedirs(rootDir)
	f = open(os.path.join(rootDir, htmlFile), 'w')
	f.write('<html>\n<body>\n')
	table = renderTable(targetCaps, outputCaps, opt.visWidth)
	f.write(table)
	f.write('</body>\n</html>\n')
	f.close()

def writeImgHTML(inputImgs, targetImgs, outputImgs, epoch, split, opt):
	print('==> Writing visualize html ...')
	htmlFile = 'index.html'

	rootDir = os.path.join(opt.www, split + '_' + str(epoch))
	if not os.path.exists(rootDir):
		os.makedirs(rootDir)

	imgDir = os.path.join(rootDir, 'imgs')
	if not os.path.exists(imgDir):
		os.makedirs(imgDir)

	f = open(os.path.join(rootDir, htmlFile), 'w')
	f.write('<html>\n<body>\n')
	table = renderImgTable(inputImgs, targetImgs, outputImgs, opt.visWidth, imgDir)
	f.write(table)
	f.write('</body>\n</html>\n')
	f.close()

